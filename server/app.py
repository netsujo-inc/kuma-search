#!/usr/bin/env python3
"""
ステップ3: 検知データをGeoJSON形式で公開するAPIサーバー
FastAPI + SQLite による軽量実装
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from contextlib import contextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn


# =============================================================================
# 設定
# =============================================================================

DATABASE_PATH = "data/detections.db"
DETECTION_EXPIRY_HOURS = 24  # 検知情報の有効期間
MAX_DETECTIONS_RESPONSE = 500


# =============================================================================
# データモデル
# =============================================================================

class DetectionInput(BaseModel):
    """検知データ入力"""
    timestamp: str
    device_id: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    confidence: float = Field(..., ge=0, le=1)
    class_name: str
    bbox: Optional[List[int]] = None
    image_path: Optional[str] = None


class DetectionRecord(BaseModel):
    """検知データレコード"""
    id: int
    timestamp: str
    device_id: str
    latitude: float
    longitude: float
    confidence: float
    class_name: str
    status: str  # "active", "cleared", "false_positive"
    created_at: str


class GeoJSONFeature(BaseModel):
    """GeoJSON Feature"""
    type: str = "Feature"
    geometry: dict
    properties: dict


class GeoJSONCollection(BaseModel):
    """GeoJSON FeatureCollection"""
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


# =============================================================================
# データベース
# =============================================================================

def init_database():
    """データベースを初期化"""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            device_id TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            confidence REAL NOT NULL,
            class_name TEXT NOT NULL,
            bbox TEXT,
            image_path TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_status 
        ON detections(status)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_detections_timestamp 
        ON detections(timestamp)
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            name TEXT,
            latitude REAL,
            longitude REAL,
            last_seen TEXT,
            status TEXT DEFAULT 'online'
        )
    """)
    
    conn.commit()
    conn.close()


@contextmanager
def get_db():
    """データベース接続のコンテキストマネージャー"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# FastAPI アプリケーション
# =============================================================================

app = FastAPI(
    title="熊検知情報共有API",
    description="ラズパイセンサーからの熊検知情報をGeoJSON形式で提供",
    version="1.0.0"
)

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """起動時の初期化"""
    init_database()


# =============================================================================
# API エンドポイント
# =============================================================================

@app.post("/api/detections", tags=["Detections"])
async def create_detection(detection: DetectionInput, background_tasks: BackgroundTasks):
    """
    新しい検知情報を登録
    Raspberry Piからの検知報告を受け付ける
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 検知情報を保存
        cursor.execute("""
            INSERT INTO detections 
            (timestamp, device_id, latitude, longitude, confidence, class_name, bbox, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection.timestamp,
            detection.device_id,
            detection.latitude,
            detection.longitude,
            detection.confidence,
            detection.class_name,
            str(detection.bbox) if detection.bbox else None,
            detection.image_path
        ))
        
        detection_id = cursor.lastrowid
        
        # デバイス情報を更新
        cursor.execute("""
            INSERT OR REPLACE INTO devices (device_id, latitude, longitude, last_seen, status)
            VALUES (?, ?, ?, ?, 'online')
        """, (
            detection.device_id,
            detection.latitude,
            detection.longitude,
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
    
    # バックグラウンドで通知処理を実行（将来の拡張用）
    background_tasks.add_task(notify_detection, detection_id)
    
    return {"status": "success", "id": detection_id}


@app.get("/api/detections", response_model=GeoJSONCollection, tags=["Detections"])
async def get_detections(
    status: str = Query("active", description="ステータスフィルター"),
    hours: int = Query(24, ge=1, le=168, description="取得する時間範囲"),
    min_confidence: float = Query(0.5, ge=0, le=1, description="最小信頼度"),
    bbox: Optional[str] = Query(None, description="バウンディングボックス (west,south,east,north)")
):
    """
    検知情報をGeoJSON形式で取得
    地図表示用のエンドポイント
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 基本クエリ
        query = """
            SELECT id, timestamp, device_id, latitude, longitude, 
                   confidence, class_name, status, created_at
            FROM detections
            WHERE status = ?
            AND confidence >= ?
            AND datetime(timestamp) >= datetime('now', ?)
        """
        params = [status, min_confidence, f'-{hours} hours']
        
        # bbox フィルター
        if bbox:
            try:
                west, south, east, north = map(float, bbox.split(','))
                query += " AND longitude >= ? AND longitude <= ? AND latitude >= ? AND latitude <= ?"
                params.extend([west, east, south, north])
            except ValueError:
                raise HTTPException(400, "Invalid bbox format")
        
        query += f" ORDER BY timestamp DESC LIMIT {MAX_DETECTIONS_RESPONSE}"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
    
    # GeoJSON形式に変換
    features = []
    for row in rows:
        # 経過時間を計算
        detection_time = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
        age_minutes = (datetime.utcnow().replace(tzinfo=detection_time.tzinfo) - detection_time).total_seconds() / 60
        
        # 緊急度を判定
        if age_minutes < 30:
            urgency = "critical"
        elif age_minutes < 120:
            urgency = "warning"
        else:
            urgency = "info"
        
        feature = GeoJSONFeature(
            geometry={
                "type": "Point",
                "coordinates": [row['longitude'], row['latitude']]
            },
            properties={
                "id": row['id'],
                "timestamp": row['timestamp'],
                "device_id": row['device_id'],
                "confidence": row['confidence'],
                "class_name": row['class_name'],
                "status": row['status'],
                "urgency": urgency,
                "age_minutes": round(age_minutes)
            }
        )
        features.append(feature)
    
    return GeoJSONCollection(features=features)


@app.get("/api/detections/summary", tags=["Detections"])
async def get_detection_summary():
    """
    検知情報のサマリーを取得
    ダッシュボード表示用
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # アクティブな検知数
        cursor.execute("""
            SELECT COUNT(*) as count FROM detections 
            WHERE status = 'active' 
            AND datetime(timestamp) >= datetime('now', '-24 hours')
        """)
        active_24h = cursor.fetchone()['count']
        
        # 直近1時間
        cursor.execute("""
            SELECT COUNT(*) as count FROM detections 
            WHERE status = 'active' 
            AND datetime(timestamp) >= datetime('now', '-1 hour')
        """)
        active_1h = cursor.fetchone()['count']
        
        # デバイス状態
        cursor.execute("""
            SELECT COUNT(*) as count FROM devices WHERE status = 'online'
        """)
        online_devices = cursor.fetchone()['count']
        
        # 地域別集計
        cursor.execute("""
            SELECT 
                ROUND(latitude, 1) as lat_bucket,
                ROUND(longitude, 1) as lon_bucket,
                COUNT(*) as count
            FROM detections
            WHERE status = 'active'
            AND datetime(timestamp) >= datetime('now', '-24 hours')
            GROUP BY lat_bucket, lon_bucket
            ORDER BY count DESC
            LIMIT 10
        """)
        hotspots = [dict(row) for row in cursor.fetchall()]
    
    return {
        "active_detections_24h": active_24h,
        "active_detections_1h": active_1h,
        "online_devices": online_devices,
        "hotspots": hotspots,
        "updated_at": datetime.utcnow().isoformat()
    }


@app.patch("/api/detections/{detection_id}", tags=["Detections"])
async def update_detection_status(
    detection_id: int,
    status: str = Query(..., description="新しいステータス (cleared, false_positive)")
):
    """
    検知情報のステータスを更新
    誤検知の報告や解除に使用
    """
    if status not in ["active", "cleared", "false_positive"]:
        raise HTTPException(400, "Invalid status")
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE detections 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, detection_id))
        
        if cursor.rowcount == 0:
            raise HTTPException(404, "Detection not found")
        
        conn.commit()
    
    return {"status": "success"}


@app.get("/api/devices", tags=["Devices"])
async def get_devices():
    """
    登録されているデバイス一覧を取得
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT device_id, name, latitude, longitude, last_seen, status
            FROM devices
            ORDER BY last_seen DESC
        """)
        devices = [dict(row) for row in cursor.fetchall()]
    
    return {"devices": devices}


@app.get("/api/health", tags=["System"])
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# =============================================================================
# バックグラウンドタスク
# =============================================================================

async def notify_detection(detection_id: int):
    """
    検知通知処理（将来の拡張用）
    - LINE通知
    - Slack通知
    - メール通知
    - 自治体システム連携
    """
    # TODO: 実際の通知処理を実装
    pass


# =============================================================================
# 静的ファイル（ビューアー用）
# =============================================================================

viewer_path = Path(__file__).parent.parent / "viewer"
if viewer_path.exists():
    app.mount("/", StaticFiles(directory=str(viewer_path), html=True), name="viewer")


# =============================================================================
# メイン
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
