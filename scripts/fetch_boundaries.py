#!/usr/bin/env python3
"""
ステップ1: Overpass APIで森林と市街地の境界データを取得
南魚沼市を例として使用（bbox座標は適宜変更）
"""

import requests
import json
import time
from typing import Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BoundingBox:
    """対象地域のバウンディングボックス"""
    south: float
    west: float
    north: float
    east: float
    
    def to_overpass(self) -> str:
        return f"{self.south},{self.west},{self.north},{self.east}"


# 石打駅の座標（JR上越線、新潟県南魚沼市）
# 参照: https://www.mapion.co.jp/m2/36.98944663,138.80443676,16/poi=ST23895
ISHIUCHI_STATION_LAT = 36.98945
ISHIUCHI_STATION_LON = 138.80444

# 石打駅を中心とした10km四方の範囲
# 緯度1度 ≈ 111km、経度1度 ≈ 91km（緯度37度付近）
# 5km ≈ 緯度0.045度、経度0.055度
MINAMI_UONUMA_BBOX = BoundingBox(
    south=ISHIUCHI_STATION_LAT - 0.045,  # 36.94445
    west=ISHIUCHI_STATION_LON - 0.055,   # 138.74944
    north=ISHIUCHI_STATION_LAT + 0.045,  # 37.03445
    east=ISHIUCHI_STATION_LON + 0.055    # 138.85944
)

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"

# レートリミット設定
REQUEST_INTERVAL_SEC = 5  # リクエスト間隔（秒）
MAX_RETRIES = 3  # 最大リトライ回数
RETRY_WAIT_SEC = 30  # リトライ時の待機時間（秒）


def fetch_with_retry(query: str, description: str = "") -> dict:
    """レートリミット対応のOverpass APIリクエスト"""
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OVERPASS_API_URL, data={"data": query}, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code in (429, 504, 502, 503):
                wait_time = RETRY_WAIT_SEC * (attempt + 1)
                print(f"  ⚠️ サーバーエラー({response.status_code})。{wait_time}秒待機後リトライ ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait_time)
            else:
                raise
        except requests.exceptions.Timeout:
            wait_time = RETRY_WAIT_SEC * (attempt + 1)
            print(f"  ⚠️ タイムアウト。{wait_time}秒待機後リトライ ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(wait_time)

    raise Exception(f"Overpass APIリクエスト失敗: {description}")


def fetch_forest_areas(bbox: BoundingBox) -> dict:
    """森林エリアを取得（広範なタグをカバー）"""
    query = f"""
    [out:json][timeout:180];
    (
      // landuse系
      way["landuse"="forest"]({bbox.to_overpass()});
      way["landuse"="meadow"]({bbox.to_overpass()});
      way["landuse"="grass"]({bbox.to_overpass()});
      relation["landuse"="forest"]({bbox.to_overpass()});
      relation["landuse"="meadow"]({bbox.to_overpass()});

      // natural系（森林・植生）
      way["natural"="wood"]({bbox.to_overpass()});
      way["natural"="scrub"]({bbox.to_overpass()});
      way["natural"="heath"]({bbox.to_overpass()});
      way["natural"="grassland"]({bbox.to_overpass()});
      way["natural"="fell"]({bbox.to_overpass()});
      way["natural"="moor"]({bbox.to_overpass()});
      way["natural"="wetland"]({bbox.to_overpass()});
      relation["natural"="wood"]({bbox.to_overpass()});
      relation["natural"="scrub"]({bbox.to_overpass()});
      relation["natural"="heath"]({bbox.to_overpass()});
      relation["natural"="grassland"]({bbox.to_overpass()});
      relation["natural"="wetland"]({bbox.to_overpass()});

      // landcover系
      way["landcover"="trees"]({bbox.to_overpass()});
      way["landcover"="grass"]({bbox.to_overpass()});
      relation["landcover"="trees"]({bbox.to_overpass()});

      // boundary系（国立公園・自然保護区など）
      way["boundary"="national_park"]({bbox.to_overpass()});
      way["boundary"="protected_area"]({bbox.to_overpass()});
      way["leisure"="nature_reserve"]({bbox.to_overpass()});
      relation["boundary"="national_park"]({bbox.to_overpass()});
      relation["boundary"="protected_area"]({bbox.to_overpass()});
      relation["leisure"="nature_reserve"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "森林エリア")


def fetch_residential_areas(bbox: BoundingBox) -> dict:
    """市街地・住宅地エリアを取得（広範なタグをカバー）"""
    query = f"""
    [out:json][timeout:180];
    (
      // landuse系（住宅・商業・工業）
      way["landuse"="residential"]({bbox.to_overpass()});
      way["landuse"="commercial"]({bbox.to_overpass()});
      way["landuse"="retail"]({bbox.to_overpass()});
      way["landuse"="industrial"]({bbox.to_overpass()});
      way["landuse"="education"]({bbox.to_overpass()});
      way["landuse"="institutional"]({bbox.to_overpass()});
      way["landuse"="religious"]({bbox.to_overpass()});
      way["landuse"="cemetery"]({bbox.to_overpass()});
      way["landuse"="construction"]({bbox.to_overpass()});
      way["landuse"="garages"]({bbox.to_overpass()});
      way["landuse"="railway"]({bbox.to_overpass()});
      relation["landuse"="residential"]({bbox.to_overpass()});
      relation["landuse"="commercial"]({bbox.to_overpass()});
      relation["landuse"="retail"]({bbox.to_overpass()});
      relation["landuse"="industrial"]({bbox.to_overpass()});
      relation["landuse"="education"]({bbox.to_overpass()});
      relation["landuse"="institutional"]({bbox.to_overpass()});

      // place系（集落・地区）
      way["place"~"village|town|city|hamlet|neighbourhood|suburb|quarter|isolated_dwelling"]({bbox.to_overpass()});
      relation["place"~"village|town|city|hamlet|neighbourhood|suburb|quarter"]({bbox.to_overpass()});

      // amenity系（学校・病院など大規模施設）
      way["amenity"~"school|university|college|hospital|clinic"]({bbox.to_overpass()});
      relation["amenity"~"school|university|hospital"]({bbox.to_overpass()});

      // 駅周辺
      way["railway"="station"]({bbox.to_overpass()});
      way["public_transport"="station"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "市街地エリア")


def fetch_buildings(bbox: BoundingBox) -> dict:
    """建物データを取得（市街地推定の補助用）"""
    query = f"""
    [out:json][timeout:120];
    (
      way["building"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "建物データ")


def fetch_roads_with_power(bbox: BoundingBox) -> dict:
    """道路と電力インフラを取得（センサー設置可能位置の判定用）"""
    query = f"""
    [out:json][timeout:120];
    (
      way["highway"~"primary|secondary|tertiary|residential"]({bbox.to_overpass()});
      node["power"="pole"]({bbox.to_overpass()});
      node["power"="tower"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "道路・電力インフラ")


def fetch_water_features(bbox: BoundingBox) -> dict:
    """河川（熊の移動経路になりやすい）を取得"""
    query = f"""
    [out:json][timeout:60];
    (
      way["waterway"~"river|stream"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "河川")


def fetch_attractants(bbox: BoundingBox) -> dict:
    """熊を引き寄せやすい地物（果樹園、養蜂場など）を取得"""
    query = f"""
    [out:json][timeout:60];
    (
      way["landuse"="orchard"]({bbox.to_overpass()});
      node["craft"="beekeeper"]({bbox.to_overpass()});
      node["amenity"="waste_disposal"]({bbox.to_overpass()});
      way["landuse"="farmland"]({bbox.to_overpass()});
    );
    out body;
    >;
    out skel qt;
    """
    return fetch_with_retry(query, "誘引地物")


def convert_to_geojson(overpass_data: dict) -> dict:
    """Overpass APIのレスポンスをGeoJSON形式に変換（relation対応）"""
    features = []
    nodes = {}
    ways = {}

    # まずノードを辞書に格納
    for element in overpass_data.get("elements", []):
        if element["type"] == "node":
            nodes[element["id"]] = {
                "lat": element["lat"],
                "lon": element["lon"]
            }

    # Wayを辞書に格納（relation処理用）
    for element in overpass_data.get("elements", []):
        if element["type"] == "way":
            coordinates = []
            for node_id in element.get("nodes", []):
                if node_id in nodes:
                    node = nodes[node_id]
                    coordinates.append([node["lon"], node["lat"]])
            if coordinates:
                ways[element["id"]] = {
                    "coordinates": coordinates,
                    "tags": element.get("tags", {})
                }

    # Wayをポリゴン/ラインに変換
    for element in overpass_data.get("elements", []):
        if element["type"] == "way":
            way_data = ways.get(element["id"])
            if not way_data:
                continue
            coordinates = way_data["coordinates"]

            if len(coordinates) >= 2:
                # 閉じたWayはポリゴン、そうでなければラインストリング
                if coordinates[0] == coordinates[-1] and len(coordinates) >= 4:
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [coordinates]
                    }
                else:
                    geometry = {
                        "type": "LineString",
                        "coordinates": coordinates
                    }

                features.append({
                    "type": "Feature",
                    "properties": element.get("tags", {}),
                    "geometry": geometry
                })

        elif element["type"] == "node" and "tags" in element:
            features.append({
                "type": "Feature",
                "properties": element.get("tags", {}),
                "geometry": {
                    "type": "Point",
                    "coordinates": [element["lon"], element["lat"]]
                }
            })

        # Relation（マルチポリゴン）を処理
        elif element["type"] == "relation":
            tags = element.get("tags", {})
            members = element.get("members", [])

            # outer ringを収集
            outer_rings = []
            inner_rings = []

            for member in members:
                if member["type"] == "way":
                    way_id = member["ref"]
                    way_data = ways.get(way_id)
                    if way_data and way_data["coordinates"]:
                        role = member.get("role", "outer")
                        if role == "outer":
                            outer_rings.append(way_data["coordinates"])
                        elif role == "inner":
                            inner_rings.append(way_data["coordinates"])

            # 各outer ringに対してポリゴンを作成
            for outer_coords in outer_rings:
                if len(outer_coords) >= 4:
                    # 閉じていない場合は閉じる
                    if outer_coords[0] != outer_coords[-1]:
                        outer_coords = outer_coords + [outer_coords[0]]

                    polygon_coords = [outer_coords]
                    # inner rings（穴）を追加
                    for inner_coords in inner_rings:
                        if len(inner_coords) >= 4:
                            if inner_coords[0] != inner_coords[-1]:
                                inner_coords = inner_coords + [inner_coords[0]]
                            polygon_coords.append(inner_coords)

                    features.append({
                        "type": "Feature",
                        "properties": tags,
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": polygon_coords
                        }
                    })

    return {
        "type": "FeatureCollection",
        "features": features
    }


def save_geojson(data: dict, filename: str, output_dir: Path):
    """GeoJSONファイルとして保存"""
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"保存完了: {output_path}")


def main():
    output_dir = Path(__file__).parent.parent / "data" / "osm"
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = MINAMI_UONUMA_BBOX
    print(f"対象地域: {bbox}")
    print(f"リクエスト間隔: {REQUEST_INTERVAL_SEC}秒")

    # 各種データを取得
    print("\n森林エリアを取得中...")
    forest_data = fetch_forest_areas(bbox)
    forest_geojson = convert_to_geojson(forest_data)
    save_geojson(forest_geojson, "forests.geojson", output_dir)
    print(f"  → {len(forest_geojson['features'])} 件の森林エリア")

    time.sleep(REQUEST_INTERVAL_SEC)

    print("\n市街地エリアを取得中...")
    residential_data = fetch_residential_areas(bbox)
    residential_geojson = convert_to_geojson(residential_data)
    save_geojson(residential_geojson, "residential.geojson", output_dir)
    print(f"  → {len(residential_geojson['features'])} 件の市街地エリア")

    time.sleep(REQUEST_INTERVAL_SEC)

    print("\n建物データを取得中...")
    buildings_data = fetch_buildings(bbox)
    buildings_geojson = convert_to_geojson(buildings_data)
    save_geojson(buildings_geojson, "buildings.geojson", output_dir)
    print(f"  → {len(buildings_geojson['features'])} 件の建物")

    time.sleep(REQUEST_INTERVAL_SEC)

    print("\n道路・電力インフラを取得中...")
    roads_data = fetch_roads_with_power(bbox)
    roads_geojson = convert_to_geojson(roads_data)
    save_geojson(roads_geojson, "infrastructure.geojson", output_dir)
    print(f"  → {len(roads_geojson['features'])} 件のインフラ")

    time.sleep(REQUEST_INTERVAL_SEC)

    print("\n河川を取得中...")
    water_data = fetch_water_features(bbox)
    water_geojson = convert_to_geojson(water_data)
    save_geojson(water_geojson, "waterways.geojson", output_dir)
    print(f"  → {len(water_geojson['features'])} 件の河川")

    time.sleep(REQUEST_INTERVAL_SEC)

    print("\n熊誘引地物を取得中...")
    attractants_data = fetch_attractants(bbox)
    attractants_geojson = convert_to_geojson(attractants_data)
    save_geojson(attractants_geojson, "attractants.geojson", output_dir)
    print(f"  → {len(attractants_geojson['features'])} 件の誘引地物")

    print("\n✅ 全データの取得が完了しました")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
