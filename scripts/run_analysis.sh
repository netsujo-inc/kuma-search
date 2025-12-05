#!/bin/bash
#
# 熊検知センサー配置分析 - 一括実行スクリプト
#
# 使い方:
#   ./run_analysis.sh [--skip-ksj] [--rebuild]
#
# オプション:
#   --skip-ksj  国土数値情報のダウンロードをスキップ（既にデータがある場合）
#   --rebuild   Dockerイメージを強制的に再ビルド
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")/data"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# オプション解析
SKIP_KSJ=false
FORCE_REBUILD=false

for arg in "$@"; do
    case $arg in
        --skip-ksj)
            SKIP_KSJ=true
            shift
            ;;
        --rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        *)
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  熊検知センサー配置分析ツール${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# データディレクトリの作成
echo -e "${YELLOW}[1/6] データディレクトリを準備中...${NC}"
mkdir -p "$DATA_DIR/osm"
mkdir -p "$DATA_DIR/ksj"
mkdir -p "$DATA_DIR/sensors"
mkdir -p "$DATA_DIR/output"
echo -e "${GREEN}  ✓ 完了${NC}"

# Dockerイメージのビルド
echo ""
echo -e "${YELLOW}[2/6] Dockerイメージをビルド中...${NC}"
cd "$SCRIPT_DIR"

if [ "$FORCE_REBUILD" = true ]; then
    docker build --no-cache -t kuma-scripts . 2>&1 | tail -5
else
    docker build -t kuma-scripts . 2>&1 | tail -5
fi
echo -e "${GREEN}  ✓ 完了${NC}"

# OSMデータの取得
echo ""
echo -e "${YELLOW}[3/6] OSMデータを取得中...${NC}"
echo -e "  (建物・道路・河川・森林エリアをOverpass APIから取得)"
docker run --rm -v "$DATA_DIR:/app/data" kuma-scripts fetch_boundaries.py 2>&1 | grep -E "(✅|❌|件|箇所)"
echo -e "${GREEN}  ✓ 完了${NC}"

# 国土数値情報 森林データの取得
echo ""
echo -e "${YELLOW}[4/6] 国土数値情報 森林データを取得中...${NC}"

if [ "$SKIP_KSJ" = true ]; then
    echo -e "  (--skip-ksj オプションによりスキップ)"
elif [ -f "$DATA_DIR/osm/ksj_forest.geojson" ]; then
    echo -e "  (既存のデータを使用: ksj_forest.geojson)"
else
    echo -e "  (初回は約185MBのダウンロードが必要です)"
    docker run --rm -v "$DATA_DIR:/app/data" kuma-scripts fetch_ksj_forest.py 2>&1 | grep -E "(ダウンロード|進捗|展開|解析|保存|✅|❌)"
fi
echo -e "${GREEN}  ✓ 完了${NC}"

# センサー配置分析
echo ""
echo -e "${YELLOW}[5/6] センサー配置を分析中...${NC}"
docker run --rm -v "$DATA_DIR:/app/data" kuma-scripts analyze_placement.py 2>&1 | grep -E "(建物数|ポイント|件|箇所|✅|❌|スコア)"
echo -e "${GREEN}  ✓ 完了${NC}"

# 可視化マップの生成
echo ""
echo -e "${YELLOW}[6/6] 可視化マップを生成中...${NC}"
docker run --rm -v "$DATA_DIR:/app/data" kuma-scripts visualize_map.py 2>&1 | grep -E "(センサー|候補|森林|建物|河川|✅|❌)"
echo -e "${GREEN}  ✓ 完了${NC}"

# 完了メッセージ
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  分析が完了しました！${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "結果ファイル:"
echo -e "  地図: ${GREEN}$DATA_DIR/output/sensor_map.html${NC}"
echo -e "  センサー候補: $DATA_DIR/sensors/sensor_locations.geojson"
echo -e "  候補建物: $DATA_DIR/sensors/candidate_buildings.geojson"
echo ""
echo -e "ブラウザで地図を開くには:"
echo -e "  ${YELLOW}open $DATA_DIR/output/sensor_map.html${NC}  (macOS)"
echo -e "  ${YELLOW}xdg-open $DATA_DIR/output/sensor_map.html${NC}  (Linux)"
echo ""
