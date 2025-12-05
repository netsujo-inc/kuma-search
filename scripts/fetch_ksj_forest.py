#!/usr/bin/env python3
"""
国土数値情報の森林地域データ（A13）をダウンロードして処理するスクリプト
新潟県のデータから対象範囲を抽出してGeoJSONに変換
"""

import requests
import zipfile
import json
import io
import os
from pathlib import Path
from typing import List, Tuple, Optional
import xml.etree.ElementTree as ET

# 国土数値情報 森林地域データ（新潟県）
# 平成27年度データ
KSJ_FOREST_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/A13/A13-15/A13-15_15_GML.zip"

# 石打駅を中心とした10km四方の範囲
ISHIUCHI_STATION_LAT = 36.98945
ISHIUCHI_STATION_LON = 138.80444

TARGET_BBOX = {
    "south": ISHIUCHI_STATION_LAT - 0.045,
    "west": ISHIUCHI_STATION_LON - 0.055,
    "north": ISHIUCHI_STATION_LAT + 0.045,
    "east": ISHIUCHI_STATION_LON + 0.055
}


def download_and_extract(url: str, extract_dir: Path) -> Path:
    """ZIPファイルをダウンロードして展開"""
    print(f"ダウンロード中: {url}")
    print("  (ファイルサイズ: 約185MB、時間がかかります)")

    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    # メモリ上でZIPを処理
    zip_buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        zip_buffer.write(chunk)
        downloaded += len(chunk)
        if total_size > 0:
            progress = downloaded / total_size * 100
            print(f"\r  進捗: {progress:.1f}% ({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)", end="")

    print("\n展開中...")
    zip_buffer.seek(0)

    with zipfile.ZipFile(zip_buffer) as zf:
        zf.extractall(extract_dir)

    return extract_dir


def find_geojson_or_shp(extract_dir: Path) -> Tuple[Optional[Path], str]:
    """展開したディレクトリからGeoJSONまたはShapefileを探す"""
    # GeoJSONを優先
    for geojson_file in extract_dir.rglob("*.geojson"):
        return geojson_file, "geojson"

    # Shapefileを探す
    for shp_file in extract_dir.rglob("*.shp"):
        return shp_file, "shapefile"

    # GMLを探す
    for gml_file in extract_dir.rglob("*.gml"):
        return gml_file, "gml"

    return None, ""


def parse_gml_forest(gml_path: Path, bbox: dict) -> dict:
    """GMLファイルを解析して森林ポリゴンを抽出（BBOX内のみ）"""
    print(f"GMLファイルを解析中: {gml_path}")

    # 名前空間の定義
    namespaces = {
        'ksj': 'http://nlftp.mlit.go.jp/ksj/schemas/ksj-app',
        'gml': 'http://www.opengis.net/gml/3.2',
        'xlink': 'http://www.w3.org/1999/xlink'
    }

    features = []

    try:
        # 大きなファイルなのでイテレータで処理
        context = ET.iterparse(gml_path, events=('end',))

        count = 0
        for event, elem in context:
            # 森林地域要素を探す
            if elem.tag.endswith('ForestArea') or 'A13' in elem.tag:
                count += 1
                if count % 1000 == 0:
                    print(f"  処理中: {count} 件...")

                # 座標を抽出
                coords_elem = elem.find('.//gml:posList', namespaces)
                if coords_elem is None:
                    coords_elem = elem.find('.//{http://www.opengis.net/gml/3.2}posList')

                if coords_elem is not None and coords_elem.text:
                    coords_text = coords_elem.text.strip()
                    coords_list = coords_text.split()

                    # lat lon lat lon ... の形式
                    polygon_coords = []
                    for i in range(0, len(coords_list) - 1, 2):
                        try:
                            lat = float(coords_list[i])
                            lon = float(coords_list[i + 1])

                            # BBOX内かチェック（少なくとも1点が範囲内）
                            if (bbox["south"] <= lat <= bbox["north"] and
                                bbox["west"] <= lon <= bbox["east"]):
                                polygon_coords.append([lon, lat])
                            else:
                                polygon_coords.append([lon, lat])
                        except (ValueError, IndexError):
                            continue

                    # ポリゴンがBBOXと交差するか簡易チェック
                    if polygon_coords and len(polygon_coords) >= 3:
                        lats = [c[1] for c in polygon_coords]
                        lons = [c[0] for c in polygon_coords]

                        # バウンディングボックスが重なるかチェック
                        if (min(lats) <= bbox["north"] and max(lats) >= bbox["south"] and
                            min(lons) <= bbox["east"] and max(lons) >= bbox["west"]):

                            # 閉じたポリゴンにする
                            if polygon_coords[0] != polygon_coords[-1]:
                                polygon_coords.append(polygon_coords[0])

                            # 属性を抽出
                            properties = {}
                            for child in elem:
                                tag_name = child.tag.split('}')[-1]
                                if child.text and tag_name not in ['area', 'loc']:
                                    properties[tag_name] = child.text

                            features.append({
                                "type": "Feature",
                                "properties": properties,
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [polygon_coords]
                                }
                            })

                # メモリ解放
                elem.clear()

        print(f"  合計: {count} 件処理、{len(features)} 件が対象範囲内")

    except Exception as e:
        print(f"GML解析エラー: {e}")

    return {
        "type": "FeatureCollection",
        "features": features
    }


def load_shapefile(shp_path: Path, bbox: dict) -> dict:
    """Shapefileを読み込んでGeoJSONに変換（BBOX内のみ）"""
    try:
        import shapefile
    except ImportError:
        print("pyshpがインストールされていません。pip install pyshp を実行してください。")
        return {"type": "FeatureCollection", "features": []}

    print(f"Shapefile を読み込み中: {shp_path}")

    features = []
    sf = shapefile.Reader(str(shp_path), encoding='cp932')

    for i, (shape_rec, rec) in enumerate(zip(sf.shapes(), sf.records())):
        if i % 1000 == 0:
            print(f"  処理中: {i} 件...")

        if shape_rec.shapeType == shapefile.POLYGON:
            coords = shape_rec.points

            # バウンディングボックスチェック
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]

            if (min(lats) <= bbox["north"] and max(lats) >= bbox["south"] and
                min(lons) <= bbox["east"] and max(lons) >= bbox["west"]):

                # プロパティを構築
                properties = {}
                for j, field in enumerate(sf.fields[1:]):  # 最初のDeletionFlagをスキップ
                    field_name = field[0]
                    properties[field_name] = rec[j]

                # Polygon座標
                polygon_coords = [[p[0], p[1]] for p in coords]
                if polygon_coords[0] != polygon_coords[-1]:
                    polygon_coords.append(polygon_coords[0])

                features.append({
                    "type": "Feature",
                    "properties": properties,
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon_coords]
                    }
                })

    print(f"  {len(features)} 件が対象範囲内")

    return {
        "type": "FeatureCollection",
        "features": features
    }


def main():
    data_dir = Path(__file__).parent.parent / "data"
    ksj_dir = data_dir / "ksj"
    ksj_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "osm" / "ksj_forest.geojson"

    # 既にダウンロード済みか確認
    extract_dir = ksj_dir / "A13-15_15"

    if not extract_dir.exists():
        print("国土数値情報 森林地域データをダウンロードします...")
        download_and_extract(KSJ_FOREST_URL, ksj_dir)
    else:
        print(f"既存のデータを使用: {extract_dir}")

    # データファイルを探す
    data_file, file_type = find_geojson_or_shp(ksj_dir)

    if data_file is None:
        print("❌ データファイルが見つかりません")
        # ディレクトリの中身を表示
        print("展開されたファイル:")
        for f in ksj_dir.rglob("*"):
            print(f"  {f}")
        return

    print(f"データファイル: {data_file} ({file_type})")

    # データを読み込み・変換
    if file_type == "geojson":
        print("GeoJSONを読み込み中...")
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # BBOXでフィルタリング
        filtered_features = []
        for feature in data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") == "Polygon":
                coords = geom["coordinates"][0]
                lats = [c[1] for c in coords]
                lons = [c[0] for c in coords]
                if (min(lats) <= TARGET_BBOX["north"] and max(lats) >= TARGET_BBOX["south"] and
                    min(lons) <= TARGET_BBOX["east"] and max(lons) >= TARGET_BBOX["west"]):
                    filtered_features.append(feature)
        geojson_data = {"type": "FeatureCollection", "features": filtered_features}

    elif file_type == "shapefile":
        geojson_data = load_shapefile(data_file, TARGET_BBOX)

    elif file_type == "gml":
        geojson_data = parse_gml_forest(data_file, TARGET_BBOX)

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 森林地域データを保存: {output_path}")
    print(f"   対象範囲内の森林ポリゴン: {len(geojson_data['features'])} 件")


if __name__ == "__main__":
    main()
