# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

熊検知情報共有システム - OpenStreetMapエコシステムを活用したAIベースの熊検知・情報共有プラットフォーム。Raspberry Pi + カメラによるリアルタイム監視とAI推論を行い、検知情報をOSMベースの地図上で共有する。

## アーキテクチャ

```
Raspberry Pi (detector.py)  →  APIサーバー (FastAPI)  →  地図ビューアー (Leaflet)
     ↓ 検知データPOST           ↓ GeoJSON提供              ↓
  TFLite/ONNX推論         SQLiteストレージ           OSM連携・uMap埋込
     (Coral TPU対応)           市民報告統合
```

4つの独立したコンポーネント:
- **raspberry-pi/**: エッジデバイス上のAI推論・検知 (Python 3.9+, picamera2/OpenCV, TFLite/ONNX, Coral Edge TPU対応)
- **server/**: 検知データAPI + OSM Notes連携 (Python 3.10+, FastAPI, SQLite)
- **viewer/**: リアルタイム地図UI (Leaflet.js + MarkerCluster, 30秒自動更新)
- **scripts/**: OSMデータ取得・センサー配置分析ツール (Python 3.10+, Docker対応)

## 開発コマンド

### サーバー起動
```bash
cd server
pip install -r requirements.txt
python app.py  # http://localhost:8000 で起動
```

### Raspberry Piセットアップ
```bash
cd raspberry-pi
./setup.sh
cp config.yaml.example config.yaml
source venv/bin/activate
python detector.py
```

### scripts/ - OSMデータ分析ツール (Docker)
```bash
cd scripts

# 一括実行（推奨）
./run_analysis.sh

# または個別実行
docker build -t kuma-scripts .
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts fetch_boundaries.py
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts fetch_ksj_forest.py  # 国土数値情報
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts analyze_placement.py
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts visualize_map.py
```

**スクリプト一覧:**
- `fetch_boundaries.py`: Overpass APIで森林、市街地、建物、道路、電力インフラ、河川、果樹園等をGeoJSONで取得
- `fetch_ksj_forest.py`: 国土数値情報（A13）から森林地域データをダウンロード・変換（初回約185MB）
- `analyze_placement.py`: 森林境界・河川近接の建物を特定し、インフラアクセス性でスコアリング
- `visualize_map.py`: 分析結果をLeaflet.jsベースのHTMLマップとして出力
- `run_analysis.sh`: 上記を一括実行するシェルスクリプト

**出力ファイル:**
- `data/sensors/sensor_locations.geojson`: センサー配置候補（スコア・理由付き）
- `data/sensors/candidate_buildings.geojson`: 候補建物（外周/川沿い分類）
- `data/output/sensor_map.html`: 可視化マップ

## 主要API

- `POST /api/detections` - 検知情報登録（Raspberry Piからの報告）
- `GET /api/detections` - GeoJSON形式で検知情報取得（クエリ: hours, status, min_confidence, bbox）
- `GET /api/detections/summary` - ダッシュボード用サマリー
- `PATCH /api/detections/{id}` - ステータス更新（誤検知報告等）
- `GET /api/devices` - 登録デバイス一覧
- `GET /api/health` - ヘルスチェック
- `GET /docs` - OpenAPI仕様

## 技術スタック

| コンポーネント | 技術 |
|--------------|------|
| 推論エンジン | TFLite / ONNX Runtime (オプション: Coral Edge TPU) |
| カメラ | picamera2 / OpenCV |
| バックエンド | FastAPI + SQLite + Pydantic v2 |
| フロントエンド | Leaflet.js + MarkerCluster + OpenStreetMap |
| 地理空間処理 | Shapely, scipy (KD-Tree, ConvexHull) |
| OSM連携 | Overpass API, OSM Notes API |
| 外部データ | 国土数値情報（森林地域A13） |

## 対象地域

デフォルトでは石打駅（新潟県南魚沼市）を中心とした10km四方を対象としている。
`fetch_boundaries.py` の `MINAMI_UONUMA_BBOX` を変更することで他地域にも対応可能。

## センサー配置アルゴリズム

`analyze_placement.py` は以下のロジックで配置候補を算出:
1. 森林境界を20m間隔で補間し、各ポイントから最も近い建物を特定
2. 河川を20m間隔で補間し、左岸・右岸それぞれで最も近い建物を特定
3. 道路・電力インフラへのアクセス性でスコアリング
4. 300m以上の間隔でクラスタリングして最終候補を選定（最大50箇所）
