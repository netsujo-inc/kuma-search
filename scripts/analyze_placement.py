#!/usr/bin/env python3
"""
ステップ2: センサー配置候補を算出
建物データを活用し、以下の建物をセンサー設置候補とする:
1. 建物群の外周に位置する建物（熊が最初に遭遇する可能性が高い）
2. 川に面した建物（熊の移動経路）
"""

import json
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.ndimage import binary_dilation
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.ops import unary_union
from shapely.prepared import prep


@dataclass
class Building:
    """建物データ"""
    id: int
    centroid_lat: float
    centroid_lon: float
    properties: dict


@dataclass
class SensorCandidate:
    """センサー設置候補地点"""
    lat: float
    lon: float
    score: float
    reasons: List[str]
    nearest_road_distance: float
    nearest_power_distance: float
    near_water: bool
    building_id: int


def load_geojson(filepath: Path) -> dict:
    """GeoJSONファイルを読み込み"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def calculate_centroid(coordinates: List) -> Tuple[float, float]:
    """ポリゴンの重心を計算"""
    if not coordinates:
        return (0, 0)

    # ポリゴンの場合は最初のリング（外周）を使用
    ring = coordinates[0] if isinstance(coordinates[0][0], list) else coordinates

    lats = [coord[1] for coord in ring]
    lons = [coord[0] for coord in ring]

    return (sum(lats) / len(lats), sum(lons) / len(lons))


def extract_buildings(geojson: dict) -> List[Building]:
    """GeoJSONから建物データを抽出"""
    buildings = []

    for i, feature in enumerate(geojson.get("features", [])):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            coords = geom.get("coordinates", [])
            if coords:
                lat, lon = calculate_centroid(coords)
                buildings.append(Building(
                    id=i,
                    centroid_lat=lat,
                    centroid_lon=lon,
                    properties=feature.get("properties", {})
                ))

    return buildings


def extract_line_points(geojson: dict, sample_interval: float = 50) -> List[Tuple[float, float]]:
    """GeoJSONからライン（河川・道路）上のポイントを抽出"""
    points = []

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            # ラインを一定間隔でサンプリング
            for i, coord in enumerate(coords):
                points.append((coord[1], coord[0]))  # lat, lon
        elif geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if len(coords) >= 2:
                points.append((coords[1], coords[0]))

    return points


def extract_line_segments(geojson: dict) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """GeoJSONからLineStringのセグメント（線分）を抽出"""
    segments = []

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "LineString":
            coords = geom.get("coordinates", [])
            for i in range(1, len(coords)):
                # (lat1, lon1), (lat2, lon2) の形式
                p1 = (coords[i-1][1], coords[i-1][0])
                p2 = (coords[i][1], coords[i][0])
                segments.append((p1, p2))

    return segments


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """2点間の距離をメートルで計算（ハヴァシン公式）"""
    R = 6371000  # 地球の半径（メートル）

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def point_to_segment_distance(
    point: Tuple[float, float],
    seg_start: Tuple[float, float],
    seg_end: Tuple[float, float]
) -> float:
    """
    点から線分への最短距離をメートルで計算
    平面近似を使用（局所的な範囲では十分な精度）
    """
    # 緯度経度をメートル単位の平面座標に変換（簡易投影）
    # 緯度37度付近: 緯度1度≈111km, 経度1度≈91km
    lat_to_m = 111000
    lon_to_m = 91000  # cos(37°) * 111000 ≈ 91000

    px = (point[1] - seg_start[1]) * lon_to_m
    py = (point[0] - seg_start[0]) * lat_to_m

    ax = 0
    ay = 0
    bx = (seg_end[1] - seg_start[1]) * lon_to_m
    by = (seg_end[0] - seg_start[0]) * lat_to_m

    # 線分ABに対する点Pの射影を計算
    ab_len_sq = bx * bx + by * by

    if ab_len_sq == 0:
        # 線分が点の場合
        return math.sqrt(px * px + py * py)

    # 射影パラメータ t (0-1の範囲でクランプ)
    t = max(0, min(1, (px * bx + py * by) / ab_len_sq))

    # 最近接点
    closest_x = ax + t * bx
    closest_y = ay + t * by

    # 距離を計算
    dx = px - closest_x
    dy = py - closest_y

    return math.sqrt(dx * dx + dy * dy)


def find_peripheral_buildings(
    buildings: List[Building],
    grid_resolution: float = 0.001,  # 約100m
    dilation_iterations: int = 2
) -> Set[int]:
    """
    建物群の外周に位置する建物を特定
    グリッドベースのアプローチで建物密集地帯の境界を検出
    """
    if len(buildings) < 10:
        # 建物が少ない場合は全て外周とみなす
        return set(b.id for b in buildings)

    # 建物座標を配列に変換
    coords = np.array([(b.centroid_lat, b.centroid_lon) for b in buildings])

    # グリッドの範囲を決定
    lat_min, lat_max = coords[:, 0].min(), coords[:, 0].max()
    lon_min, lon_max = coords[:, 1].min(), coords[:, 1].max()

    # グリッドサイズを計算
    lat_bins = max(10, int((lat_max - lat_min) / grid_resolution))
    lon_bins = max(10, int((lon_max - lon_min) / grid_resolution))

    # 建物の密度グリッドを作成
    grid = np.zeros((lat_bins, lon_bins), dtype=bool)

    building_grid_pos = {}  # 建物IDとグリッド位置のマッピング

    for b in buildings:
        lat_idx = min(lat_bins - 1, int((b.centroid_lat - lat_min) / (lat_max - lat_min + 1e-10) * lat_bins))
        lon_idx = min(lon_bins - 1, int((b.centroid_lon - lon_min) / (lon_max - lon_min + 1e-10) * lon_bins))
        grid[lat_idx, lon_idx] = True
        building_grid_pos[b.id] = (lat_idx, lon_idx)

    # グリッドを膨張させて建物エリアを形成
    dilated_grid = binary_dilation(grid, iterations=dilation_iterations)

    # 外周セルを特定（膨張後のグリッドの境界）
    peripheral_cells = set()

    for i in range(lat_bins):
        for j in range(lon_bins):
            if dilated_grid[i, j]:
                # 隣接セルのいずれかが空（建物エリア外）なら外周
                is_peripheral = False
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if ni < 0 or ni >= lat_bins or nj < 0 or nj >= lon_bins:
                            is_peripheral = True
                            break
                        if not dilated_grid[ni, nj]:
                            is_peripheral = True
                            break
                    if is_peripheral:
                        break
                if is_peripheral:
                    peripheral_cells.add((i, j))

    # 外周セルに位置する建物を特定
    peripheral_building_ids = set()

    for b in buildings:
        lat_idx, lon_idx = building_grid_pos[b.id]
        # 建物自体のセルまたは近傍セルが外周なら外周建物
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if (lat_idx + di, lon_idx + dj) in peripheral_cells:
                    peripheral_building_ids.add(b.id)
                    break

    return peripheral_building_ids


def load_forest_polygons(forest_geojson_path: Path) -> Optional[MultiPolygon]:
    """
    国土数値情報の森林データをShapelyのMultiPolygonとして読み込み
    """
    if not forest_geojson_path.exists():
        return None

    with open(forest_geojson_path, "r", encoding="utf-8") as f:
        forest_data = json.load(f)

    polygons = []
    for feature in forest_data.get("features", []):
        geom = feature.get("geometry", {})
        if geom.get("type") == "Polygon":
            try:
                poly = shape(geom)
                if not poly.is_valid:
                    # 自己交差などを修正（buffer(0)は自己交差を解消する標準的な方法）
                    poly = poly.buffer(0)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
            except Exception:
                pass

    if not polygons:
        return None

    # 全ての森林ポリゴンをマージ
    merged = unary_union(polygons)
    if isinstance(merged, Polygon):
        return MultiPolygon([merged])
    elif isinstance(merged, MultiPolygon):
        return merged
    else:
        return None


def interpolate_forest_boundary(
    forest_polygons: MultiPolygon,
    interval_meters: float = 20
) -> List[Tuple[float, float]]:
    """
    森林境界を一定間隔で補間してポイントを生成
    戻り値: [(lat, lon), ...] のリスト
    """
    from shapely.geometry import LineString, MultiLineString

    # 緯度37度付近の変換係数
    lat_to_m = 111000
    lon_to_m = 91000
    avg_m_per_deg = (lat_to_m + lon_to_m) / 2

    # 度単位での補間間隔
    interval_deg = interval_meters / avg_m_per_deg

    boundary_points = []

    # 森林境界線を取得
    boundary = forest_polygons.boundary

    # LineStringまたはMultiLineStringから座標を抽出
    if isinstance(boundary, LineString):
        lines = [boundary]
    elif isinstance(boundary, MultiLineString):
        lines = list(boundary.geoms)
    else:
        # GeometryCollectionの場合
        lines = []
        for geom in boundary.geoms if hasattr(boundary, 'geoms') else [boundary]:
            if isinstance(geom, LineString):
                lines.append(geom)
            elif isinstance(geom, MultiLineString):
                lines.extend(geom.geoms)

    for line in lines:
        # 線分の長さ
        line_length = line.length

        if line_length == 0:
            continue

        # 補間ポイント数を計算
        num_points = max(2, int(line_length / interval_deg) + 1)

        for i in range(num_points):
            # 0.0 ~ 1.0 の位置パラメータ
            t = i / (num_points - 1) if num_points > 1 else 0
            point = line.interpolate(t, normalized=True)
            # (lat, lon) の形式で追加
            boundary_points.append((point.y, point.x))

    return boundary_points


def find_forest_adjacent_buildings(
    buildings: List[Building],
    forest_polygons: MultiPolygon,
    interpolation_interval: float = 20,  # 20m間隔で補間
    distance_threshold_m: Optional[float] = None  # Noneの場合は閾値なし（最寄り建物を無条件で追加）
) -> Set[int]:
    """
    森林エリアの境界に近い建物を特定（境界起点アプローチ）

    アプローチ:
    1. 森林境界を20m間隔で補間してポイントを生成
    2. 各境界ポイントから最も近い建物（森林外）を検索
    3. すべての境界ポイントから検出された建物を返す

    これにより、境界全体をカバーし、距離があっても境界に面している建物を検出可能
    """
    if forest_polygons is None or len(buildings) == 0:
        return set()

    # 緯度37度付近での変換係数
    lat_to_m = 111000
    lon_to_m = 91000

    # 森林境界を補間
    boundary_points = interpolate_forest_boundary(forest_polygons, interpolation_interval)
    print(f"  森林境界補間ポイント: {len(boundary_points)} 点（{interpolation_interval}m間隔）")

    # 高速化のために prepared geometry を使用
    prepared_forest = prep(forest_polygons)

    # 森林外の建物のみをフィルタリング
    outside_buildings = []
    for b in buildings:
        point = Point(b.centroid_lon, b.centroid_lat)
        if not prepared_forest.contains(point):
            outside_buildings.append(b)

    if not outside_buildings:
        return set()

    # 建物座標をメートル単位の平面座標に変換してKD-Tree構築
    base_lat = outside_buildings[0].centroid_lat
    base_lon = outside_buildings[0].centroid_lon

    building_coords_m = np.array([
        ((b.centroid_lat - base_lat) * lat_to_m,
         (b.centroid_lon - base_lon) * lon_to_m)
        for b in outside_buildings
    ])
    building_tree = cKDTree(building_coords_m)

    forest_adjacent_ids = set()

    # 各境界ポイントから最も近い建物を検索
    for boundary_lat, boundary_lon in boundary_points:
        point_m = np.array([
            (boundary_lat - base_lat) * lat_to_m,
            (boundary_lon - base_lon) * lon_to_m
        ])

        # 最も近い建物を検索
        distance, idx = building_tree.query(point_m)

        # 距離閾値がない場合は無条件で追加、ある場合は閾値以内のみ追加
        if distance_threshold_m is None or distance <= distance_threshold_m:
            forest_adjacent_ids.add(outside_buildings[idx].id)

    return forest_adjacent_ids


def interpolate_line_points(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    interval_meters: float = 20
) -> List[Tuple[float, float]]:
    """
    線分を一定間隔で補間してポイントを生成
    interval_meters: 補間間隔（メートル）
    """
    points = []

    # 緯度37度付近の変換係数
    lat_to_m = 111000
    lon_to_m = 91000

    for seg_start, seg_end in segments:
        # セグメントの長さを計算
        dlat = (seg_end[0] - seg_start[0]) * lat_to_m
        dlon = (seg_end[1] - seg_start[1]) * lon_to_m
        seg_length = math.sqrt(dlat * dlat + dlon * dlon)

        if seg_length == 0:
            points.append(seg_start)
            continue

        # 補間ポイント数を計算
        num_points = max(2, int(seg_length / interval_meters) + 1)

        for i in range(num_points):
            t = i / (num_points - 1)
            lat = seg_start[0] + t * (seg_end[0] - seg_start[0])
            lon = seg_start[1] + t * (seg_end[1] - seg_start[1])
            points.append((lat, lon))

    return points


def interpolate_line_with_direction(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    interval_meters: float = 20
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    線分を一定間隔で補間し、各ポイントと進行方向ベクトルを返す
    戻り値: [(point, direction_vector), ...]
    direction_vector は正規化された (dy, dx) タプル
    """
    results = []

    # 緯度37度付近の変換係数
    lat_to_m = 111000
    lon_to_m = 91000

    for seg_start, seg_end in segments:
        # セグメントの方向と長さを計算
        dlat = (seg_end[0] - seg_start[0]) * lat_to_m
        dlon = (seg_end[1] - seg_start[1]) * lon_to_m
        seg_length = math.sqrt(dlat * dlat + dlon * dlon)

        if seg_length == 0:
            continue

        # 正規化された方向ベクトル
        dir_lat = dlat / seg_length
        dir_lon = dlon / seg_length

        # 補間ポイント数を計算
        num_points = max(2, int(seg_length / interval_meters) + 1)

        for i in range(num_points):
            t = i / (num_points - 1)
            lat = seg_start[0] + t * (seg_end[0] - seg_start[0])
            lon = seg_start[1] + t * (seg_end[1] - seg_start[1])
            results.append(((lat, lon), (dir_lat, dir_lon)))

    return results


def find_riverside_buildings(
    buildings: List[Building],
    water_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    interpolation_interval: float = 20  # 20m間隔で補間
) -> Set[int]:
    """
    川の両岸で最も近い建物を特定（河川起点アプローチ）

    1. 河川を密に補間してポイントと進行方向を生成
    2. 各ポイントで左岸・右岸それぞれで最も近い建物をピックアップ
    """
    if not water_segments or not buildings:
        return set()

    # 河川を密に補間（進行方向付き）
    water_points_with_dir = interpolate_line_with_direction(water_segments, interpolation_interval)
    print(f"    河川補間ポイント: {len(water_points_with_dir)} 点（{interpolation_interval}m間隔）")

    # 緯度37度付近の変換係数
    lat_to_m = 111000
    lon_to_m = 91000

    # 建物座標をメートル単位の平面座標に変換してKD-Tree構築
    base_lat = buildings[0].centroid_lat
    base_lon = buildings[0].centroid_lon

    building_coords_m = np.array([
        ((b.centroid_lat - base_lat) * lat_to_m,
         (b.centroid_lon - base_lon) * lon_to_m)
        for b in buildings
    ])
    building_tree = cKDTree(building_coords_m)

    riverside_ids = set()

    # 各河川ポイントで左岸・右岸の最寄り建物を検索
    for water_point, direction in water_points_with_dir:
        point_m = np.array([
            (water_point[0] - base_lat) * lat_to_m,
            (water_point[1] - base_lon) * lon_to_m
        ])

        # 進行方向に対する垂直ベクトル（左岸: 左90度、右岸: 右90度）
        # direction = (dir_lat, dir_lon) をメートル単位で
        # 左90度回転: (-dir_lon, dir_lat)
        # 右90度回転: (dir_lon, -dir_lat)
        perp_left = (-direction[1], direction[0])  # 左岸方向
        perp_right = (direction[1], -direction[0])  # 右岸方向

        # 近傍の建物を取得（まず広めに検索）
        # k個の最近傍を取得して、左岸・右岸に分類
        k = min(50, len(buildings))
        distances, indices = building_tree.query(point_m, k=k)

        left_bank_nearest = None
        left_bank_dist = float('inf')
        right_bank_nearest = None
        right_bank_dist = float('inf')

        for dist, idx in zip(distances, indices):
            if dist == float('inf'):
                continue

            # 建物の位置ベクトル（河川ポイントからの相対位置）
            building_vec = building_coords_m[idx] - point_m

            # 左岸・右岸の判定（内積で方向を判定）
            dot_left = building_vec[0] * perp_left[0] + building_vec[1] * perp_left[1]
            dot_right = building_vec[0] * perp_right[0] + building_vec[1] * perp_right[1]

            if dot_left > 0:  # 左岸側
                if dist < left_bank_dist:
                    left_bank_dist = dist
                    left_bank_nearest = idx
            elif dot_right > 0:  # 右岸側
                if dist < right_bank_dist:
                    right_bank_dist = dist
                    right_bank_nearest = idx

        # 左岸・右岸それぞれの最寄り建物を追加
        if left_bank_nearest is not None:
            riverside_ids.add(buildings[left_bank_nearest].id)
        if right_bank_nearest is not None:
            riverside_ids.add(buildings[right_bank_nearest].id)

    return riverside_ids


def calculate_sensor_scores(
    buildings: List[Building],
    peripheral_ids: Set[int],
    riverside_ids: Set[int],
    road_points: List[Tuple[float, float]],
    power_points: List[Tuple[float, float]],
    max_road_distance: float = 100,
    max_power_distance: float = 50,
    use_forest_data: bool = False
) -> List[SensorCandidate]:
    """各建物のセンサー設置スコアを計算"""
    candidates = []

    # KD-Treeを構築
    road_tree = cKDTree(np.array(road_points)) if road_points else None
    power_tree = cKDTree(np.array(power_points)) if power_points else None

    for b in buildings:
        # 外周でも川沿いでもない建物はスキップ
        is_peripheral = b.id in peripheral_ids
        is_riverside = b.id in riverside_ids

        if not is_peripheral and not is_riverside:
            continue

        score = 0
        reasons = []
        point = np.array([b.centroid_lat, b.centroid_lon])

        # 森林隣接/外周建物ボーナス
        if is_peripheral:
            score += 30
            if use_forest_data:
                reasons.append("森林隣接（熊出没の最前線）")
            else:
                reasons.append("建物エリア外周（熊侵入の最前線）")

        # 川沿いボーナス
        if is_riverside:
            score += 25
            reasons.append("河川近接（熊の移動経路）")

        # 道路からの距離
        if road_tree:
            road_dist_deg, _ = road_tree.query(point)
            road_dist_m = road_dist_deg * 111000
        else:
            road_dist_m = float('inf')

        # 電柱からの距離
        if power_tree:
            power_dist_deg, _ = power_tree.query(point)
            power_dist_m = power_dist_deg * 111000
        else:
            power_dist_m = float('inf')

        # 道路アクセス性
        if road_dist_m <= max_road_distance:
            score += 25
            reasons.append(f"道路アクセス良好（{road_dist_m:.0f}m）")
        elif road_dist_m <= max_road_distance * 2:
            score += 10
            reasons.append(f"道路アクセス可（{road_dist_m:.0f}m）")

        # 電力アクセス性
        if power_dist_m <= max_power_distance:
            score += 20
            reasons.append(f"電源確保容易（{power_dist_m:.0f}m）")
        elif power_dist_m <= max_power_distance * 3:
            score += 10
            reasons.append(f"電源確保可能（{power_dist_m:.0f}m）")
        else:
            score += 5
            reasons.append("ソーラー運用推奨")

        # 建物タイプによるボーナス（もしタグがあれば）
        building_type = b.properties.get("building", "")
        if building_type in ["public", "civic", "school", "hospital"]:
            score += 10
            reasons.append(f"公共施設（{building_type}）")

        candidates.append(SensorCandidate(
            lat=b.centroid_lat,
            lon=b.centroid_lon,
            score=score,
            reasons=reasons,
            nearest_road_distance=road_dist_m,
            nearest_power_distance=power_dist_m,
            near_water=is_riverside,
            building_id=b.id
        ))

    return candidates


def cluster_and_select(
    candidates: List[SensorCandidate],
    min_distance: float = 300,  # 最低300m間隔
    max_sensors: int = 50
) -> List[SensorCandidate]:
    """
    クラスタリングして最適なセンサー位置を選定
    近すぎる候補は統合し、スコアの高いものを優先
    """
    # スコアでソート
    sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)

    selected = []

    for candidate in sorted_candidates:
        if len(selected) >= max_sensors:
            break

        # 既選定地点との距離をチェック
        too_close = False
        for existing in selected:
            dist = haversine_distance(
                candidate.lat, candidate.lon,
                existing.lat, existing.lon
            )
            if dist < min_distance:
                too_close = True
                break

        if not too_close:
            selected.append(candidate)

    return selected


def export_to_geojson(candidates: List[SensorCandidate], filepath: Path):
    """選定結果をGeoJSONとしてエクスポート"""
    features = []

    for i, candidate in enumerate(candidates, 1):
        features.append({
            "type": "Feature",
            "properties": {
                "id": f"sensor-{i:03d}",
                "score": candidate.score,
                "reasons": candidate.reasons,
                "road_distance_m": round(candidate.nearest_road_distance, 1),
                "power_distance_m": round(candidate.nearest_power_distance, 1),
                "near_water": candidate.near_water,
                "building_id": candidate.building_id,
                "priority": "high" if candidate.score >= 70 else "medium" if candidate.score >= 50 else "low"
            },
            "geometry": {
                "type": "Point",
                "coordinates": [candidate.lon, candidate.lat]
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"センサー配置候補を保存: {filepath}")


def export_candidate_buildings(
    buildings: List[Building],
    peripheral_ids: Set[int],
    riverside_ids: Set[int],
    filepath: Path
):
    """候補建物（外周・川沿い）をGeoJSONとしてエクスポート"""
    features = []

    for b in buildings:
        is_peripheral = b.id in peripheral_ids
        is_riverside = b.id in riverside_ids

        if not is_peripheral and not is_riverside:
            continue

        # カテゴリを決定
        if is_peripheral and is_riverside:
            category = "both"
        elif is_peripheral:
            category = "peripheral"
        else:
            category = "riverside"

        features.append({
            "type": "Feature",
            "properties": {
                "building_id": b.id,
                "category": category,
                "is_peripheral": is_peripheral,
                "is_riverside": is_riverside
            },
            "geometry": {
                "type": "Point",
                "coordinates": [b.centroid_lon, b.centroid_lat]
            }
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"候補建物データを保存: {filepath}")


def main():
    data_dir = Path(__file__).parent.parent / "data" / "osm"
    output_dir = Path(__file__).parent.parent / "data" / "sensors"
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("OSMデータを読み込み中...")

    try:
        buildings_geojson = load_geojson(data_dir / "buildings.geojson")
        infrastructure = load_geojson(data_dir / "infrastructure.geojson")
        waterways = load_geojson(data_dir / "waterways.geojson")
    except FileNotFoundError as e:
        print(f"❌ データファイルが見つかりません: {e}")
        print("先に fetch_boundaries.py を実行してください")
        return

    # 建物データを抽出
    print("\n建物データを処理中...")
    buildings = extract_buildings(buildings_geojson)
    print(f"  建物数: {len(buildings)} 件")

    if not buildings:
        print("❌ 建物データがありません")
        return

    # インフラポイントを抽出
    road_points = extract_line_points(infrastructure)
    power_points = road_points  # 道路沿いに電柱があると仮定

    # 河川セグメント（線分）を抽出
    water_segments = extract_line_segments(waterways)

    print(f"  道路/インフラポイント: {len(road_points)} 箇所")
    print(f"  河川セグメント: {len(water_segments)} 本")

    # 森林隣接建物を特定（国土数値情報を使用）
    ksj_forest_path = data_dir / "ksj_forest.geojson"
    forest_polygons = load_forest_polygons(ksj_forest_path)

    if forest_polygons is not None:
        print("\n森林隣接建物を分析中（国土数値情報使用）...")
        print(f"  森林ポリゴン数: {len(forest_polygons.geoms)}")
        peripheral_ids = find_forest_adjacent_buildings(
            buildings,
            forest_polygons,
            interpolation_interval=20,  # 20m間隔で境界を補間
            distance_threshold_m=None   # 閾値なし（各境界ポイントの最寄り建物を無条件で追加）
        )
        print(f"  森林隣接建物: {len(peripheral_ids)} 件")
    else:
        # フォールバック: グリッドベースの外周検出
        print("\n建物エリアの外周を分析中（グリッドベース）...")
        print("  ⚠️ 国土数値情報の森林データがありません。")
        print("     fetch_ksj_forest.py を実行して森林データを取得してください。")
        peripheral_ids = find_peripheral_buildings(buildings)
        print(f"  外周建物: {len(peripheral_ids)} 件")

    # 川沿い建物を特定（線分との最短距離で判定）
    print("\n河川近接建物を分析中（線分距離計算）...")
    riverside_ids = find_riverside_buildings(buildings, water_segments)
    print(f"  河川近接建物: {len(riverside_ids)} 件")

    # 候補建物（外周 OR 川沿い）
    candidate_ids = peripheral_ids | riverside_ids
    print(f"\n候補建物（外周∪川沿い）: {len(candidate_ids)} 件")

    # スコア計算
    print("\nセンサー配置スコアを計算中...")
    scored_candidates = calculate_sensor_scores(
        buildings,
        peripheral_ids,
        riverside_ids,
        road_points,
        power_points,
        use_forest_data=(forest_polygons is not None)
    )
    print(f"  スコア計算完了: {len(scored_candidates)} 件")

    # クラスタリングと選定
    print("\n最適配置を選定中...")
    selected = cluster_and_select(
        scored_candidates,
        min_distance=300,
        max_sensors=50
    )

    print(f"\n✅ {len(selected)} 箇所のセンサー配置候補を選定")

    # 結果をエクスポート
    export_to_geojson(selected, output_dir / "sensor_locations.geojson")

    # 候補建物（外周・川沿い）もエクスポート
    export_candidate_buildings(
        buildings,
        peripheral_ids,
        riverside_ids,
        output_dir / "candidate_buildings.geojson"
    )

    # サマリーを表示
    print("\n--- 配置候補サマリー ---")
    high_priority = [c for c in selected if c.score >= 70]
    medium_priority = [c for c in selected if 50 <= c.score < 70]
    low_priority = [c for c in selected if c.score < 50]

    print(f"  高優先度（スコア70以上）: {len(high_priority)} 箇所")
    print(f"  中優先度（スコア50-69）: {len(medium_priority)} 箇所")
    print(f"  低優先度（スコア50未満）: {len(low_priority)} 箇所")

    # 森林隣接・川沿いの内訳
    peripheral_only = [c for c in selected if not c.near_water]
    riverside_only = [c for c in selected if c.near_water and c.building_id not in peripheral_ids]
    both = [c for c in selected if c.near_water and c.building_id in peripheral_ids]

    peripheral_label = "森林隣接" if forest_polygons is not None else "外周"
    print(f"\n  {peripheral_label}のみ: {len(peripheral_only)} 箇所")
    print(f"  川沿いのみ: {len(riverside_only)} 箇所")
    print(f"  {peripheral_label}かつ川沿い: {len(both)} 箇所")

    if selected:
        print("\n上位5箇所:")
        for i, c in enumerate(selected[:5], 1):
            print(f"  {i}. スコア{c.score}: ({c.lat:.5f}, {c.lon:.5f})")
            print(f"     理由: {', '.join(c.reasons)}")


if __name__ == "__main__":
    main()
