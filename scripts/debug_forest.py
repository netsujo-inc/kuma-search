#!/usr/bin/env python3
"""森林データ読み込みのデバッグスクリプト"""

import json
from pathlib import Path
from shapely.geometry import shape, MultiPolygon, Polygon
from shapely.ops import unary_union

ksj_path = Path('/app/data/osm/ksj_forest.geojson')
print(f'Loading from: {ksj_path}')
print(f'File exists: {ksj_path.exists()}')

if not ksj_path.exists():
    print('File not found!')
    exit(1)

with open(ksj_path, 'r') as f:
    forest_data = json.load(f)

print(f'Total features: {len(forest_data.get("features", []))}')

polygons = []
for i, feature in enumerate(forest_data.get('features', [])):
    geom = feature.get('geometry', {})
    print(f'Feature {i}: type={geom.get("type")}')
    if geom.get('type') == 'Polygon':
        try:
            poly = shape(geom)
            was_valid = poly.is_valid
            if not poly.is_valid:
                # 自己交差などを修正
                poly = poly.buffer(0)
            print(f'  Original valid: {was_valid}, After fix: {poly.is_valid}, Area: {poly.area}')
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)
        except Exception as e:
            print(f'  Error: {e}')

print(f'\nTotal valid polygons: {len(polygons)}')

if polygons:
    print('Merging polygons...')
    merged = unary_union(polygons)
    print(f'Merged type: {type(merged).__name__}')
    if isinstance(merged, Polygon):
        result = MultiPolygon([merged])
    elif isinstance(merged, MultiPolygon):
        result = merged
    else:
        result = None
    print(f'Result: {type(result).__name__ if result else None}')
    if result:
        print(f'Result geoms: {len(result.geoms)}')
