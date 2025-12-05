#!/usr/bin/env python3
"""
センサー配置候補を地図上に可視化するスクリプト
Leaflet.jsを使用したHTMLファイルを生成
"""

import json
from pathlib import Path
from fetch_boundaries import MINAMI_UONUMA_BBOX


def load_geojson(filepath: Path) -> dict:
    """GeoJSONファイルを読み込み"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_html(sensors: dict, forests: dict, residential: dict, buildings: dict, waterways: dict, bbox: dict, candidate_buildings: dict) -> str:
    """Leaflet.jsを使用した地図HTMLを生成"""

    # センサー配置候補をJavaScript用に変換
    sensors_js = json.dumps(sensors, ensure_ascii=False)
    bbox_js = json.dumps(bbox, ensure_ascii=False)
    forests_js = json.dumps(forests, ensure_ascii=False)
    residential_js = json.dumps(residential, ensure_ascii=False)
    buildings_js = json.dumps(buildings, ensure_ascii=False)
    waterways_js = json.dumps(waterways, ensure_ascii=False)
    candidate_buildings_js = json.dumps(candidate_buildings, ensure_ascii=False)

    # 中心座標を計算（センサー候補の平均）
    if sensors["features"]:
        lats = [f["geometry"]["coordinates"][1] for f in sensors["features"]]
        lons = [f["geometry"]["coordinates"][0] for f in sensors["features"]]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        center_lat, center_lon = 37.05, 138.9  # デフォルト: 南魚沼市

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>熊検知センサー配置候補マップ</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ height: 100vh; width: 100%; }}
        .legend {{
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            line-height: 1.8;
        }}
        .legend h4 {{ margin-bottom: 8px; font-size: 14px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 12px; }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        }}
        .legend-line {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        .info-panel {{
            background: white;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            max-width: 300px;
        }}
        .info-panel h4 {{ margin-bottom: 8px; color: #333; }}
        .info-panel p {{ font-size: 12px; color: #666; margin: 4px 0; }}
        .score-badge {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 11px;
        }}
        .score-high {{ background: #22c55e; color: white; }}
        .score-medium {{ background: #f59e0b; color: white; }}
        .score-low {{ background: #ef4444; color: white; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <script>
        // データ
        const sensorsData = {sensors_js};
        const forestsData = {forests_js};
        const residentialData = {residential_js};
        const buildingsData = {buildings_js};
        const waterwaysData = {waterways_js};
        const bboxData = {bbox_js};
        const candidateBuildingsData = {candidate_buildings_js};

        // マップ初期化
        const map = L.map('map').setView([{center_lat}, {center_lon}], 12);

        // ベースマップ
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }}).addTo(map);

        // 森林エリア（緑）- デフォルト表示
        const forestLayer = L.geoJSON(forestsData, {{
            style: {{
                color: '#166534',
                fillColor: '#22c55e',
                fillOpacity: 0.3,
                weight: 1
            }}
        }}).addTo(map);

        // 市街地エリア（オレンジ）- デフォルト非表示
        const residentialLayer = L.geoJSON(residentialData, {{
            style: {{
                color: '#c2410c',
                fillColor: '#fb923c',
                fillOpacity: 0.3,
                weight: 1
            }}
        }});

        // 建物（紫）- デフォルト表示
        const buildingsLayer = L.geoJSON(buildingsData, {{
            style: {{
                color: '#7c3aed',
                fillColor: '#a78bfa',
                fillOpacity: 0.4,
                weight: 1
            }}
        }}).addTo(map);

        // 河川（青）- デフォルト表示
        const waterwaysLayer = L.geoJSON(waterwaysData, {{
            style: {{
                color: '#0ea5e9',
                weight: 2
            }}
        }}).addTo(map);

        // 候補建物（外周建物: オレンジ、川沿い建物: 水色、両方: 赤）- デフォルト表示
        const candidateBuildingsLayer = L.geoJSON(candidateBuildingsData, {{
            pointToLayer: function(feature, latlng) {{
                const category = feature.properties.category;
                let color = '#f97316';  // デフォルト: オレンジ（外周のみ）
                if (category === 'both') color = '#ef4444';  // 両方: 赤
                else if (category === 'riverside') color = '#06b6d4';  // 川沿いのみ: 水色

                return L.circleMarker(latlng, {{
                    radius: 4,
                    fillColor: color,
                    color: color,
                    weight: 1,
                    opacity: 0.7,
                    fillOpacity: 0.5
                }});
            }},
            onEachFeature: function(feature, layer) {{
                const props = feature.properties;
                const categoryName = props.category === 'both' ? '外周かつ川沿い' :
                                     props.category === 'peripheral' ? '外周建物' : '川沿い建物';
                layer.bindPopup(`
                    <div class="info-panel">
                        <h4>候補建物 #${{props.building_id}}</h4>
                        <p><strong>カテゴリ:</strong> ${{categoryName}}</p>
                        <p><strong>外周:</strong> ${{props.is_peripheral ? 'はい' : 'いいえ'}}</p>
                        <p><strong>川沿い:</strong> ${{props.is_riverside ? 'はい' : 'いいえ'}}</p>
                    </div>
                `);
            }}
        }}).addTo(map);

        // 検索対象範囲（赤い破線の矩形）
        const searchBounds = [
            [bboxData.south, bboxData.west],
            [bboxData.north, bboxData.east]
        ];
        const searchAreaLayer = L.rectangle(searchBounds, {{
            color: '#dc2626',
            weight: 3,
            dashArray: '10, 5',
            fill: false
        }}).addTo(map);
        searchAreaLayer.bindPopup(`
            <div class="info-panel">
                <h4>検索対象範囲</h4>
                <p><strong>南緯:</strong> ${{bboxData.south.toFixed(4)}}°</p>
                <p><strong>北緯:</strong> ${{bboxData.north.toFixed(4)}}°</p>
                <p><strong>西経:</strong> ${{bboxData.west.toFixed(4)}}°</p>
                <p><strong>東経:</strong> ${{bboxData.east.toFixed(4)}}°</p>
            </div>
        `);

        // センサー配置候補（赤いマーカー）- デフォルト非表示
        const sensorsLayer = L.geoJSON(sensorsData, {{
            pointToLayer: function(feature, latlng) {{
                const score = feature.properties.score;
                let color = '#ef4444';  // デフォルト: 赤
                if (score >= 70) color = '#22c55e';  // 高優先度: 緑
                else if (score >= 50) color = '#f59e0b';  // 中優先度: オレンジ

                return L.circleMarker(latlng, {{
                    radius: 10,
                    fillColor: color,
                    color: '#fff',
                    weight: 2,
                    opacity: 1,
                    fillOpacity: 0.9
                }});
            }},
            onEachFeature: function(feature, layer) {{
                const props = feature.properties;
                const scoreClass = props.score >= 70 ? 'score-high' : props.score >= 50 ? 'score-medium' : 'score-low';

                const popupContent = `
                    <div class="info-panel">
                        <h4>${{props.id}}</h4>
                        <p><span class="score-badge ${{scoreClass}}">スコア: ${{props.score}}</span></p>
                        <p><strong>道路距離:</strong> ${{props.road_distance_m}}m</p>
                        <p><strong>電柱距離:</strong> ${{props.power_distance_m}}m</p>
                        <p><strong>河川近接:</strong> ${{props.near_water ? 'はい' : 'いいえ'}}</p>
                        <p><strong>理由:</strong></p>
                        <ul style="font-size: 11px; margin-left: 15px;">
                            ${{props.reasons.map(r => '<li>' + r + '</li>').join('')}}
                        </ul>
                    </div>
                `;
                layer.bindPopup(popupContent);
            }}
        }});

        // レイヤーコントロール
        const overlays = {{
            "検索対象範囲": searchAreaLayer,
            "センサー候補": sensorsLayer,
            "候補建物（外周・川沿い）": candidateBuildingsLayer,
            "森林エリア": forestLayer,
            "市街地エリア": residentialLayer,
            "建物": buildingsLayer,
            "河川": waterwaysLayer
        }};

        L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

        // 凡例
        const legend = L.control({{ position: 'bottomright' }});
        legend.onAdd = function(map) {{
            const div = L.DomUtil.create('div', 'legend');
            div.innerHTML = `
                <h4>凡例</h4>
                <div class="legend-item">
                    <div class="legend-color" style="background: #22c55e;"></div>
                    <span>高優先度（スコア70+）</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f59e0b;"></div>
                    <span>中優先度（スコア50-69）</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444;"></div>
                    <span>低優先度（スコア50未満）</span>
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <div class="legend-item">
                    <div class="legend-color" style="background: #ef4444; width: 10px; height: 10px;"></div>
                    <span>候補: 外周かつ川沿い</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f97316; width: 10px; height: 10px;"></div>
                    <span>候補: 外周建物</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #06b6d4; width: 10px; height: 10px;"></div>
                    <span>候補: 川沿い建物</span>
                </div>
                <hr style="margin: 8px 0; border: none; border-top: 1px solid #ddd;">
                <div class="legend-item">
                    <div class="legend-line" style="background: #22c55e;"></div>
                    <span>森林エリア</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #fb923c;"></div>
                    <span>市街地エリア</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #a78bfa;"></div>
                    <span>建物</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #0ea5e9;"></div>
                    <span>河川</span>
                </div>
                <div class="legend-item">
                    <div class="legend-line" style="background: #dc2626; border-style: dashed;"></div>
                    <span>検索対象範囲</span>
                </div>
            `;
            return div;
        }};
        legend.addTo(map);

        // センサー候補にズーム
        if (sensorsData.features.length > 0) {{
            map.fitBounds(sensorsLayer.getBounds(), {{ padding: [50, 50] }});
        }}
    </script>
</body>
</html>
"""
    return html


def main():
    data_dir = Path(__file__).parent.parent / "data"
    osm_dir = data_dir / "osm"
    sensors_dir = data_dir / "sensors"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("データを読み込み中...")

    try:
        sensors = load_geojson(sensors_dir / "sensor_locations.geojson")
        forests = load_geojson(osm_dir / "forests.geojson")
        residential = load_geojson(osm_dir / "residential.geojson")
        waterways = load_geojson(osm_dir / "waterways.geojson")
        # 建物データはオプション（データ量が大きいため）
        try:
            buildings = load_geojson(osm_dir / "buildings.geojson")
        except FileNotFoundError:
            buildings = {"type": "FeatureCollection", "features": []}
            print("  ⚠️ 建物データなし（オプション）")
        # 候補建物データ
        try:
            candidate_buildings = load_geojson(sensors_dir / "candidate_buildings.geojson")
        except FileNotFoundError:
            candidate_buildings = {"type": "FeatureCollection", "features": []}
            print("  ⚠️ 候補建物データなし（オプション）")
    except FileNotFoundError as e:
        print(f"❌ データファイルが見つかりません: {e}")
        print("先に fetch_boundaries.py と analyze_placement.py を実行してください")
        return

    print(f"  センサー候補: {len(sensors['features'])} 箇所")
    print(f"  候補建物: {len(candidate_buildings['features'])} 箇所")
    print(f"  森林エリア: {len(forests['features'])} 箇所")
    print(f"  市街地エリア: {len(residential['features'])} 箇所")
    print(f"  建物: {len(buildings['features'])} 箇所")
    print(f"  河川: {len(waterways['features'])} 箇所")

    print("\nHTMLを生成中...")
    # 検索対象範囲（BBOX）
    bbox = {
        "south": MINAMI_UONUMA_BBOX.south,
        "west": MINAMI_UONUMA_BBOX.west,
        "north": MINAMI_UONUMA_BBOX.north,
        "east": MINAMI_UONUMA_BBOX.east
    }
    html = generate_html(sensors, forests, residential, buildings, waterways, bbox, candidate_buildings)

    output_path = output_dir / "sensor_map.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n✅ 地図を生成しました: {output_path}")
    print("ブラウザで開いて確認してください")


if __name__ == "__main__":
    main()
