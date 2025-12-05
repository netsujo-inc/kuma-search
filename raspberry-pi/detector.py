#!/usr/bin/env python3
"""
Raspberry Pi上で動作する熊検知システム
TensorFlow Lite または ONNX Runtime による推論

必要なハードウェア:
- Raspberry Pi 4/5 (4GB以上推奨)
- Raspberry Pi Camera Module v2/v3 または USB Webカメラ
- オプション: Coral Edge TPU (推論高速化)
"""

import os
import sys
import time
import json
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Callable

import yaml
import requests
import numpy as np

# カメラライブラリ（環境に応じて選択）
try:
    from picamera2 import Picamera2
    CAMERA_TYPE = "picamera2"
except ImportError:
    try:
        import cv2
        CAMERA_TYPE = "opencv"
    except ImportError:
        CAMERA_TYPE = None
        print("警告: カメラライブラリが見つかりません")

# 推論エンジン（環境に応じて選択）
try:
    import tflite_runtime.interpreter as tflite
    INFERENCE_ENGINE = "tflite"
except ImportError:
    try:
        import onnxruntime as ort
        INFERENCE_ENGINE = "onnx"
    except ImportError:
        INFERENCE_ENGINE = None
        print("警告: 推論エンジンが見つかりません")

# Coral Edge TPU（オプション）
try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common, detect
    HAS_CORAL = True
except ImportError:
    HAS_CORAL = False


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """検知結果"""
    timestamp: str
    device_id: str
    latitude: float
    longitude: float
    confidence: float
    class_name: str
    bbox: Optional[List[int]]  # [x1, y1, x2, y2]
    image_path: Optional[str]
    

@dataclass
class Config:
    """設定"""
    device_id: str
    latitude: float
    longitude: float
    model_path: str
    labels_path: str
    server_url: str
    detection_threshold: float
    target_classes: List[str]
    capture_interval: float
    image_save_dir: str
    use_coral: bool
    camera_resolution: Tuple[int, int]


def load_config(config_path: str = "config.yaml") -> Config:
    """設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return Config(
        device_id=data.get("device_id", "unknown"),
        latitude=data.get("latitude", 0.0),
        longitude=data.get("longitude", 0.0),
        model_path=data.get("model_path", "models/bear_detector.tflite"),
        labels_path=data.get("labels_path", "models/labels.txt"),
        server_url=data.get("server_url", "http://localhost:8000"),
        detection_threshold=data.get("detection_threshold", 0.6),
        target_classes=data.get("target_classes", ["bear", "クマ"]),
        capture_interval=data.get("capture_interval", 1.0),
        image_save_dir=data.get("image_save_dir", "captures"),
        use_coral=data.get("use_coral", False) and HAS_CORAL,
        camera_resolution=tuple(data.get("camera_resolution", [640, 480]))
    )


class CameraCapture:
    """カメラキャプチャクラス"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480)):
        self.resolution = resolution
        self.camera = None
        
        if CAMERA_TYPE == "picamera2":
            self._init_picamera2()
        elif CAMERA_TYPE == "opencv":
            self._init_opencv()
        else:
            raise RuntimeError("利用可能なカメラライブラリがありません")
    
    def _init_picamera2(self):
        """Picamera2の初期化"""
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": self.resolution, "format": "RGB888"}
        )
        self.camera.configure(config)
        self.camera.start()
        time.sleep(2)  # ウォームアップ
        logger.info("Picamera2を初期化しました")
    
    def _init_opencv(self):
        """OpenCVの初期化"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        logger.info("OpenCV VideoCapture を初期化しました")
    
    def capture(self) -> Optional[np.ndarray]:
        """フレームをキャプチャ"""
        if CAMERA_TYPE == "picamera2":
            return self.camera.capture_array()
        elif CAMERA_TYPE == "opencv":
            ret, frame = self.camera.read()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def close(self):
        """カメラを解放"""
        if CAMERA_TYPE == "picamera2" and self.camera:
            self.camera.stop()
        elif CAMERA_TYPE == "opencv" and self.camera:
            self.camera.release()


class BearDetector:
    """熊検知器"""
    
    def __init__(self, config: Config):
        self.config = config
        self.labels = self._load_labels()
        self.interpreter = None
        
        if config.use_coral:
            self._init_coral()
        elif INFERENCE_ENGINE == "tflite":
            self._init_tflite()
        elif INFERENCE_ENGINE == "onnx":
            self._init_onnx()
        else:
            raise RuntimeError("利用可能な推論エンジンがありません")
    
    def _load_labels(self) -> List[str]:
        """ラベルファイルを読み込み"""
        try:
            with open(self.config.labels_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            logger.warning("ラベルファイルが見つかりません。デフォルトを使用します")
            return ["background", "bear"]
    
    def _init_tflite(self):
        """TensorFlow Liteの初期化"""
        self.interpreter = tflite.Interpreter(model_path=self.config.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'][1:3]
        logger.info(f"TFLite モデルを読み込みました: {self.config.model_path}")
        logger.info(f"入力サイズ: {self.input_shape}")
    
    def _init_coral(self):
        """Coral Edge TPUの初期化"""
        self.interpreter = edgetpu.make_interpreter(self.config.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape'][1:3]
        logger.info(f"Coral Edge TPU モデルを読み込みました")
    
    def _init_onnx(self):
        """ONNX Runtimeの初期化"""
        self.session = ort.InferenceSession(
            self.config.model_path,
            providers=['CPUExecutionProvider']
        )
        
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape[2:4]  # [batch, channels, height, width]
        logger.info(f"ONNX モデルを読み込みました: {self.config.model_path}")
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """前処理（リサイズ、正規化）"""
        import cv2
        
        # リサイズ
        resized = cv2.resize(frame, tuple(self.input_shape[::-1]))
        
        # 正規化（モデルに応じて調整）
        normalized = resized.astype(np.float32) / 255.0
        
        # バッチ次元を追加
        if INFERENCE_ENGINE == "onnx":
            # ONNX: [batch, channels, height, width]
            return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        else:
            # TFLite: [batch, height, width, channels]
            return np.expand_dims(normalized, axis=0)
    
    def detect(self, frame: np.ndarray) -> List[Tuple[str, float, List[int]]]:
        """
        推論を実行
        Returns: [(class_name, confidence, [x1, y1, x2, y2]), ...]
        """
        input_data = self.preprocess(frame)
        
        if INFERENCE_ENGINE == "tflite" or self.config.use_coral:
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                input_data.astype(np.float32)
            )
            self.interpreter.invoke()
            
            # 出力の取得（モデル構造に依存）
            boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
            classes = self.interpreter.get_tensor(self.output_details[1]['index'])
            scores = self.interpreter.get_tensor(self.output_details[2]['index'])
            
        elif INFERENCE_ENGINE == "onnx":
            outputs = self.session.run(None, {self.input_name: input_data})
            boxes, scores, classes = outputs[0], outputs[1], outputs[2]
        
        # 結果をパース
        detections = []
        h, w = frame.shape[:2]
        
        for i in range(len(scores[0])):
            score = float(scores[0][i])
            if score < self.config.detection_threshold:
                continue
            
            class_id = int(classes[0][i])
            class_name = self.labels[class_id] if class_id < len(self.labels) else "unknown"
            
            # 対象クラスかチェック
            if class_name.lower() not in [c.lower() for c in self.config.target_classes]:
                continue
            
            # バウンディングボックス（正規化座標 → ピクセル座標）
            box = boxes[0][i]
            x1 = int(box[1] * w)
            y1 = int(box[0] * h)
            x2 = int(box[3] * w)
            y2 = int(box[2] * h)
            
            detections.append((class_name, score, [x1, y1, x2, y2]))
        
        return detections


class DetectionReporter:
    """検知結果をサーバーに報告"""
    
    def __init__(self, config: Config):
        self.config = config
        self.queue = queue.Queue()
        self.running = True
        
        # 送信スレッドを開始
        self.thread = threading.Thread(target=self._send_loop, daemon=True)
        self.thread.start()
    
    def report(self, detection: Detection):
        """検知結果をキューに追加"""
        self.queue.put(detection)
    
    def _send_loop(self):
        """送信ループ"""
        while self.running:
            try:
                detection = self.queue.get(timeout=1.0)
                self._send(detection)
            except queue.Empty:
                continue
    
    def _send(self, detection: Detection):
        """サーバーに送信"""
        try:
            url = f"{self.config.server_url}/api/detections"
            response = requests.post(
                url,
                json=asdict(detection),
                timeout=10
            )
            if response.status_code == 200:
                logger.info(f"検知結果を送信しました: {detection.device_id}")
            else:
                logger.warning(f"送信エラー: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"通信エラー: {e}")
            # リトライキューに追加（簡易実装）
            self.queue.put(detection)
            time.sleep(5)
    
    def stop(self):
        """送信スレッドを停止"""
        self.running = False
        self.thread.join(timeout=5)


def save_detection_image(
    frame: np.ndarray,
    detection: Tuple[str, float, List[int]],
    save_dir: str,
    device_id: str
) -> str:
    """検知画像を保存"""
    import cv2
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # バウンディングボックスを描画
    class_name, confidence, bbox = detection
    x1, y1, x2, y2 = bbox
    
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(
        frame_bgr,
        f"{class_name}: {confidence:.2f}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )
    
    # ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{device_id}_{timestamp}.jpg"
    filepath = str(Path(save_dir) / filename)
    
    cv2.imwrite(filepath, frame_bgr)
    logger.info(f"画像を保存: {filepath}")
    
    return filepath


def main():
    """メインループ"""
    # 設定読み込み
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        logger.error(f"設定ファイルが見つかりません: {config_path}")
        logger.info("config.yaml.example を参考に作成してください")
        sys.exit(1)
    
    config = load_config(str(config_path))
    logger.info(f"デバイスID: {config.device_id}")
    logger.info(f"位置: ({config.latitude}, {config.longitude})")
    
    # 各コンポーネントを初期化
    try:
        camera = CameraCapture(config.camera_resolution)
        detector = BearDetector(config)
        reporter = DetectionReporter(config)
    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        sys.exit(1)
    
    logger.info("熊検知システムを開始します...")
    
    try:
        while True:
            start_time = time.time()
            
            # フレームをキャプチャ
            frame = camera.capture()
            if frame is None:
                logger.warning("フレーム取得失敗")
                time.sleep(0.5)
                continue
            
            # 検知実行
            detections = detector.detect(frame)
            
            # 検知があれば報告
            for class_name, confidence, bbox in detections:
                logger.warning(f"🐻 熊を検知! 信頼度: {confidence:.2f}")
                
                # 画像を保存
                image_path = save_detection_image(
                    frame, (class_name, confidence, bbox),
                    config.image_save_dir, config.device_id
                )
                
                # 検知結果を作成
                detection = Detection(
                    timestamp=datetime.utcnow().isoformat() + "Z",
                    device_id=config.device_id,
                    latitude=config.latitude,
                    longitude=config.longitude,
                    confidence=confidence,
                    class_name=class_name,
                    bbox=bbox,
                    image_path=image_path
                )
                
                # サーバーに報告
                reporter.report(detection)
            
            # インターバル調整
            elapsed = time.time() - start_time
            sleep_time = max(0, config.capture_interval - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logger.info("終了シグナルを受信")
    finally:
        camera.close()
        reporter.stop()
        logger.info("システムを終了しました")


if __name__ == "__main__":
    main()
