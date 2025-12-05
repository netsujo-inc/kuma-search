#!/bin/bash
#
# Raspberry Pi 熊検知システム セットアップスクリプト
# 対応: Raspberry Pi 4/5, Bookworm (64-bit)
#

set -e

echo "=========================================="
echo " 熊検知システム セットアップ"
echo "=========================================="

# 基本パッケージのインストール
echo ""
echo "[1/6] システムパッケージをインストール中..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-opencv \
    libatlas-base-dev \
    libopenblas-dev \
    libjpeg-dev \
    libpng-dev \
    git

# Python仮想環境の作成
echo ""
echo "[2/6] Python仮想環境を作成中..."
python3 -m venv venv
source venv/bin/activate

# Pythonパッケージのインストール
echo ""
echo "[3/6] Pythonパッケージをインストール中..."
pip install --upgrade pip
pip install -r requirements.txt

# カメラの有効化確認
echo ""
echo "[4/6] カメラ設定を確認中..."
if [ -e /dev/video0 ] || [ -e /dev/video1 ]; then
    echo "  ✓ カメラデバイスが検出されました"
else
    echo "  ⚠ カメラデバイスが見つかりません"
    echo "    Raspberry Pi Camera の場合: sudo raspi-config で有効化してください"
    echo "    USB カメラの場合: 接続を確認してください"
fi

# モデルディレクトリの作成
echo ""
echo "[5/6] ディレクトリ構造を作成中..."
mkdir -p models
mkdir -p captures
mkdir -p logs

# サンプルモデルのダウンロード（SSD MobileNet v2 COCO）
echo ""
echo "[6/6] サンプルモデルをダウンロード中..."
MODEL_URL="https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"

if [ ! -f models/detect.tflite ]; then
    echo "  COCOデータセットで学習済みのモデルをダウンロード..."
    wget -q $MODEL_URL -O /tmp/model.zip
    unzip -q /tmp/model.zip -d /tmp/model
    cp /tmp/model/detect.tflite models/
    rm -rf /tmp/model /tmp/model.zip
    echo "  ✓ サンプルモデルをダウンロードしました"
    echo ""
    echo "  ⚠ 注意: これは汎用の物体検知モデルです"
    echo "    熊専用モデルは別途用意してください"
else
    echo "  ✓ モデルは既に存在します"
fi

# ラベルファイルの作成
cat > models/labels.txt << 'EOF'
background
person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
EOF

echo "  ✓ ラベルファイルを作成しました"

# 設定ファイルのコピー
if [ ! -f config.yaml ]; then
    cp config.yaml.example config.yaml
    echo ""
    echo "  ✓ config.yaml を作成しました"
    echo "    位置情報とサーバーURLを編集してください"
fi

# systemd サービスファイルの作成
echo ""
echo "systemd サービスを設定中..."

WORKING_DIR=$(pwd)
USER=$(whoami)

sudo tee /etc/systemd/system/bear-detector.service > /dev/null << EOF
[Unit]
Description=Bear Detection System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$WORKING_DIR
Environment=PATH=$WORKING_DIR/venv/bin:/usr/bin
ExecStart=$WORKING_DIR/venv/bin/python detector.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
echo "  ✓ systemd サービスを登録しました"

# 完了メッセージ
echo ""
echo "=========================================="
echo " セットアップ完了!"
echo "=========================================="
echo ""
echo "次のステップ:"
echo ""
echo "1. 設定ファイルを編集:"
echo "   nano config.yaml"
echo ""
echo "2. テスト実行:"
echo "   source venv/bin/activate"
echo "   python detector.py"
echo ""
echo "3. サービスとして起動:"
echo "   sudo systemctl enable bear-detector"
echo "   sudo systemctl start bear-detector"
echo ""
echo "4. ログの確認:"
echo "   sudo journalctl -u bear-detector -f"
echo ""

# Coral Edge TPU を使用する場合
echo "----------------------------------------"
echo "オプション: Coral Edge TPU を使用する場合"
echo "----------------------------------------"
echo "echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
echo "curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
echo "sudo apt-get update"
echo "sudo apt-get install libedgetpu1-std"
echo "pip install pycoral"
echo ""
