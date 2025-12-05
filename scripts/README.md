# 熊検知センサー配置分析ツール

野生熊の出没リスクが高い場所にセンサーを設置するための候補地点を分析するツール群です。

## 概要

以下の条件を満たす建物をセンサー設置候補として抽出します：

1. **森林隣接建物**: 国土数値情報の森林地域データを使用し、森林境界に面している建物を検出
2. **河川近接建物**: 河川の両岸で最も近い建物を検出（熊の移動経路として重要）

これらの候補から、道路アクセス・電源確保の容易さを加味してスコアリングし、最適な配置を選定します。

## アルゴリズム

### 森林隣接建物の検出（境界起点アプローチ）

1. 国土数値情報（A13）から森林ポリゴンを取得
2. 森林境界線を20m間隔で補間してポイントを生成（約50万ポイント）
3. 各境界ポイントから最も近い建物（森林外）を検索
4. すべての境界ポイントで検出された建物を「森林隣接建物」として返す

このアプローチにより、森林から多少距離があっても「その境界線に対して最も近い建物」を確実に検出できます。

### 河川近接建物の検出（両岸アプローチ）

1. 河川LineStringを20m間隔で補間してポイントと進行方向を生成
2. 各ポイントで左岸・右岸それぞれの最寄り建物を検索
3. すべてのポイントで検出された建物を「河川近接建物」として返す

## ディレクトリ構成

```
scripts/
├── README.md              # このファイル
├── Dockerfile             # Python実行環境
├── requirements.txt       # Python依存関係
├── run_analysis.sh        # 一括実行スクリプト
├── fetch_boundaries.py    # OSMデータ取得（建物・道路・河川）
├── fetch_ksj_forest.py    # 国土数値情報 森林データ取得
├── analyze_placement.py   # センサー配置分析
├── visualize_map.py       # 結果可視化HTML生成
└── debug_forest.py        # デバッグ用

data/
├── osm/                   # OSMから取得したデータ
│   ├── buildings.geojson
│   ├── forests.geojson
│   ├── residential.geojson
│   ├── waterways.geojson
│   ├── infrastructure.geojson
│   └── ksj_forest.geojson # 国土数値情報の森林データ
├── ksj/                   # 国土数値情報の元データ
├── sensors/               # 分析結果
│   ├── sensor_locations.geojson
│   └── candidate_buildings.geojson
└── output/
    └── sensor_map.html    # 可視化マップ
```

## 使い方

### 1. 対象地域の設定

`fetch_boundaries.py` の以下の部分を編集して対象地域を変更します：

```python
# 中心座標（例: 石打駅）
ISHIUCHI_STATION_LAT = 36.98945
ISHIUCHI_STATION_LON = 138.80444

# 中心から約5km四方の範囲
MINAMI_UONUMA_BBOX = BoundingBox(
    south=ISHIUCHI_STATION_LAT - 0.045,  # 緯度5km ≈ 0.045度
    west=ISHIUCHI_STATION_LON - 0.055,   # 経度5km ≈ 0.055度（緯度37度付近）
    north=ISHIUCHI_STATION_LAT + 0.045,
    east=ISHIUCHI_STATION_LON + 0.055
)
```

また、`fetch_ksj_forest.py` の対象範囲も同様に更新してください：

```python
# 同じ座標設定
ISHIUCHI_STATION_LAT = 36.98945
ISHIUCHI_STATION_LON = 138.80444

TARGET_BBOX = {
    "south": ISHIUCHI_STATION_LAT - 0.045,
    "west": ISHIUCHI_STATION_LON - 0.055,
    "north": ISHIUCHI_STATION_LAT + 0.045,
    "east": ISHIUCHI_STATION_LON + 0.055
}

# 県別データURL（新潟県以外の場合は変更が必要）
KSJ_FOREST_URL = "https://nlftp.mlit.go.jp/ksj/gml/data/A13/A13-15/A13-15_15_GML.zip"
```

国土数値情報の森林データURLは以下から該当する都道府県のものを取得：
https://nlftp.mlit.go.jp/ksj/gml/datalist/KsjTmplt-A13.html

### 2. 分析の実行

```bash
cd scripts
./run_analysis.sh
```

これにより以下が自動的に実行されます：
1. Dockerイメージのビルド
2. OSMデータの取得
3. 国土数値情報 森林データの取得（初回のみ、約185MB）
4. センサー配置分析
5. 可視化マップの生成

### 3. 結果の確認

生成された `data/output/sensor_map.html` をブラウザで開きます。

デフォルト表示：
- 検索対象範囲（赤い破線）
- 候補建物（森林隣接: オレンジ、川沿い: 水色、両方: 赤）
- 森林エリア（緑）
- 建物（紫）
- 河川（青）

レイヤーコントロールから表示を切り替えられます：
- センサー候補（デフォルト非表示）
- 市街地エリア（デフォルト非表示）

## 出力データ

### sensor_locations.geojson

選定されたセンサー配置候補（最大50箇所）。各候補には以下の情報が含まれます：

- `score`: 配置スコア（0-100）
- `reasons`: スコアの理由リスト
- `road_distance_m`: 最寄り道路までの距離
- `power_distance_m`: 最寄り電柱までの距離
- `near_water`: 河川近接フラグ
- `priority`: 優先度（high/medium/low）

### candidate_buildings.geojson

すべての候補建物（森林隣接または河川近接）。各建物には以下の情報が含まれます：

- `category`: peripheral（森林隣接のみ）/ riverside（川沿いのみ）/ both（両方）
- `is_peripheral`: 森林隣接フラグ
- `is_riverside`: 河川近接フラグ

## 技術詳細

### 使用データソース

| データ | ソース | 用途 |
|--------|--------|------|
| 建物 | OpenStreetMap | センサー設置候補 |
| 道路 | OpenStreetMap | アクセス性評価 |
| 河川 | OpenStreetMap | 河川近接建物検出 |
| 森林地域 | 国土数値情報（A13） | 森林隣接建物検出 |

### 座標変換

緯度37度付近での近似変換係数：
- 緯度1度 ≈ 111km
- 経度1度 ≈ 91km（cos(37°) × 111km）

### パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 森林境界補間間隔 | 20m | 境界線上のポイント生成間隔 |
| 河川補間間隔 | 20m | 河川上のポイント生成間隔 |
| 最小センサー間隔 | 300m | 選定時のクラスタリング距離 |
| 最大センサー数 | 50 | 選定する最大候補数 |

## 依存関係

- Python 3.10+
- requests
- numpy
- scipy
- shapely
- pyshp

## ライセンス

OpenStreetMapデータは © OpenStreetMap contributors のライセンスに従います。
国土数値情報は国土交通省が提供するオープンデータです。
