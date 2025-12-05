# 熊検知情報共有システム

OpenStreetMapエコシステムを活用した、AIベースの熊検知・情報共有プラットフォームです。

## 概要

山間部と市街地の境界に設置したRaspberry Pi + カメラによるリアルタイム監視と、AIによる熊の自動検知を行い、検知情報をOSMベースの地図上で共有します。

## システム構成

```
┌─────────────────────────────────────────────────────────────────┐
│                        熊検知システム                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐     ┌──────────────────┐     ┌────────────┐  │
│  │ Raspberry Pi  │────▶│   APIサーバー     │────▶│   uMap     │  │
│  │ + カメラ      │     │   (FastAPI)      │     │  (自治体)   │  │
│  │ + AI推論      │     └────────┬─────────┘     └────────────┘  │
│  └───────────────┘              │                                │
│         ×N台                    │                                │
│                                 ▼                                │
│                    ┌────────────────────────┐                   │
│                    │     地図ビューアー       │                   │
│                    │   (Leaflet + OSM)      │                   │
│                    └────────────────────────┘                   │
│                                 ▲                                │
│                                 │                                │
│                    ┌────────────────────────┐                   │
│                    │    市民報告 (OSM Notes) │                   │
│                    └────────────────────────┘                   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## ディレクトリ構成

```
kuma-search/
├── raspberry-pi/              # Raspberry Pi用コード
│   ├── detector.py            # メイン検知スクリプト（TFLite/ONNX対応）
│   ├── config.yaml.example    # 設定テンプレート
│   ├── setup.sh               # セットアップスクリプト（systemd対応）
│   └── requirements.txt
│
├── server/                    # バックエンドサーバー
│   ├── app.py                 # FastAPI サーバー（GeoJSON API）
│   ├── osm_utils.py           # OSM Notes連携・uMapエクスポート
│   └── requirements.txt
│
├── viewer/                    # フロントエンド
│   └── index.html             # Leaflet.js ビューアー（リアルタイム更新）
│
├── scripts/                   # OSMデータ取得・センサー配置分析（Docker対応）
│   ├── fetch_boundaries.py    # Overpass APIでOSMデータ取得
│   ├── fetch_ksj_forest.py    # 国土数値情報 森林地域データ取得
│   ├── analyze_placement.py   # センサー配置候補算出（森林境界・河川近接）
│   ├── visualize_map.py       # 分析結果の可視化マップ生成
│   ├── run_analysis.sh        # 一括実行スクリプト
│   ├── Dockerfile
│   └── requirements.txt
│
└── data/                      # データ保存ディレクトリ
    ├── osm/                   # OSM/国土数値情報から取得したGeoJSON
    ├── ksj/                   # 国土数値情報の生データ
    ├── sensors/               # センサー配置候補
    └── output/                # 可視化結果（HTML等）
```

## セットアップ手順

### 1. OSMデータの取得とセンサー配置分析

#### 一括実行（推奨）

```bash
cd scripts
chmod +x run_analysis.sh
./run_analysis.sh
```

これにより以下が自動実行されます:
1. Dockerイメージのビルド
2. Overpass APIからOSMデータ取得（森林、建物、河川等）
3. 国土数値情報から森林地域データ取得（初回約185MB）
4. センサー配置候補の算出
5. 可視化マップの生成

結果:
- `data/sensors/sensor_locations.geojson` - センサー配置候補
- `data/output/sensor_map.html` - 可視化マップ（ブラウザで開く）

#### 個別実行（Docker）

```bash
cd scripts
docker build -t kuma-scripts .

# OSMデータ取得
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts fetch_boundaries.py

# 国土数値情報 森林データ取得（オプション、より精度の高い森林境界）
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts fetch_ksj_forest.py

# センサー配置分析
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts analyze_placement.py

# 可視化マップ生成
docker run --rm -v $(pwd)/../data:/app/data kuma-scripts visualize_map.py
```

#### ローカル実行

```bash
cd scripts
pip install -r requirements.txt
python fetch_boundaries.py
python fetch_ksj_forest.py  # オプション
python analyze_placement.py
python visualize_map.py
```

### 2. サーバーのセットアップ

```bash
cd server
pip install -r requirements.txt

# サーバー起動
python app.py
```

サーバーは `http://localhost:8000` で起動します。

APIドキュメント: `http://localhost:8000/docs`

### 3. Raspberry Piのセットアップ

```bash
cd raspberry-pi
chmod +x setup.sh
./setup.sh

# 設定ファイルを編集
cp config.yaml.example config.yaml
nano config.yaml

# 動作確認
source venv/bin/activate
python detector.py
```

### 4. 地図ビューアーの確認

サーバー起動後、ブラウザで `http://localhost:8000` にアクセスすると地図ビューアーが表示されます。

## API仕様

### 検知情報の登録 (POST /api/detections)

```json
{
  "timestamp": "2025-12-03T14:30:00Z",
  "device_id": "cam-minami-uonuma-001",
  "latitude": 37.065,
  "longitude": 138.88,
  "confidence": 0.92,
  "class_name": "bear",
  "bbox": [100, 150, 300, 400],
  "image_path": "/captures/cam-001_20251203_143000.jpg"
}
```

### 検知情報の取得 (GET /api/detections)

GeoJSON形式で返却されます。緊急度（urgency）が自動計算されます。

クエリパラメータ:
- `hours`: 取得する時間範囲（デフォルト: 24、最大168）
- `status`: ステータスフィルター（active/cleared/false_positive）
- `min_confidence`: 最小信頼度（デフォルト: 0.5）
- `bbox`: バウンディングボックス（west,south,east,north）

### 検知サマリー (GET /api/detections/summary)

ダッシュボード表示用のサマリー情報を返却します。

### ステータス更新 (PATCH /api/detections/{id})

検知情報のステータスを更新（誤検知報告、解除等）。

### デバイス一覧 (GET /api/devices)

登録されているセンサーデバイスの一覧を返却。

### ヘルスチェック (GET /api/health)

システムの稼働状態を確認。

## OSMデータの活用

### 1. 境界データの取得

`fetch_boundaries.py` は以下のデータをOverpass APIから取得します:

- 森林エリア (`landuse=forest`, `natural=wood`)
- 市街地エリア (`landuse=residential`, `landuse=commercial`)
- 道路・電力インフラ
- 河川（熊の移動経路）
- 果樹園・養蜂場（熊の誘引要因）

### 2. センサー配置の最適化

`analyze_placement.py` は建物ベースのアプローチで最適な配置候補を算出します:

**アルゴリズム:**
1. 森林境界を20m間隔で補間し、各ポイントから最も近い建物を「森林隣接建物」として特定
2. 河川を20m間隔で補間し、左岸・右岸それぞれで最も近い建物を「河川近接建物」として特定
3. 森林隣接 OR 河川近接の建物をセンサー設置候補とする
4. 道路・電力インフラへのアクセス性でスコアリング
5. 300m以上の間隔でクラスタリングして最終候補を選定

**スコアリング基準:**

| 要素 | スコア |
|------|--------|
| 森林隣接建物（熊出没の最前線） | +30 |
| 河川近接建物（熊の移動経路） | +25 |
| 道路から100m以内（アクセス性） | +25 |
| 電源確保容易（50m以内） | +20 |
| 公共施設 | +10 |

**データソース:**
- OSM: 建物、道路、電柱、河川
- 国土数値情報（A13）: 森林地域（より精度の高い森林境界）

### 3. uMapとの連携

1. [uMap](https://umap.openstreetmap.fr/) でマップを作成
2. 「Remote data」設定でAPIエンドポイントを指定:
   ```
   https://your-server.example.com/api/detections
   ```
3. 自動更新を1分間隔に設定
4. 自治体のWebサイトにiframeで埋め込み

### 4. OSM Notesとの統合

市民からの熊目撃情報をOSM Notes経由で収集し、AI検知結果と統合することで信頼性を向上させます。

## 推奨ハードウェア

### センサーノード（1台あたり）

| 部品 | 推奨モデル | 概算価格 |
|------|-----------|---------|
| Raspberry Pi | Pi 5 (4GB) | ¥12,000 |
| カメラ | Camera Module v3 | ¥5,000 |
| 電源 | ソーラーパネル + バッテリー | ¥8,000 |
| ケース | 防水IP65以上 | ¥3,000 |
| 通信 | LTEモジュール (SORACOM等) | ¥5,000 |
| **合計** | | **約¥33,000** |

### オプション: 推論高速化

- Coral Edge TPU USB: 約¥10,000
  - 推論速度: ~10fps → ~30fps

## 熊検知モデル

### 推奨モデル

1. **YOLOv8n-bear** (カスタム学習)
   - サイズ: 約6MB
   - 速度: Raspberry Pi 5で約5fps
   
2. **EfficientDet-Lite (TFLite)**
   - サイズ: 約4MB
   - 速度: Raspberry Pi 5で約3fps

### モデル学習データ

- [Open Images Dataset V7](https://storage.googleapis.com/openimages/web/index.html) - "Bear"クラス
- 独自収集データ（ツキノワグマ、ヒグマ）
- データ拡張: 季節・天候・時間帯バリエーション

## ライセンス

MIT License

## 謝辞

- OpenStreetMap contributors
- Leaflet.js
- FastAPI
