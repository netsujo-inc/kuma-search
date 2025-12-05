#!/usr/bin/env python3
"""
OSM Notes との連携ユーティリティ
市民からの熊目撃情報をOSM Notes経由で収集・統合
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import requests


OSM_API_BASE = "https://api.openstreetmap.org/api/0.6"
OSM_NOTES_SEARCH = "https://api.openstreetmap.org/api/0.6/notes/search"


@dataclass
class OSMNote:
    """OSM Note データ"""
    id: int
    lat: float
    lon: float
    status: str  # "open" or "closed"
    created_at: str
    comments: List[Dict[str, Any]]


class OSMNotesClient:
    """OSM Notes API クライアント"""
    
    def __init__(self, user_agent: str = "BearDetectionSystem/1.0"):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent
        })
    
    def search_bear_notes(
        self,
        bbox: tuple,  # (west, south, east, north)
        keywords: List[str] = None,
        days: int = 30
    ) -> List[OSMNote]:
        """
        熊関連のNoteを検索
        
        Args:
            bbox: 検索範囲のバウンディングボックス
            keywords: 検索キーワード（デフォルト: 熊関連ワード）
            days: 何日前までを検索対象とするか
        """
        if keywords is None:
            keywords = ["熊", "クマ", "くま", "bear", "ツキノワグマ", "ヒグマ"]
        
        west, south, east, north = bbox
        
        # 日付フィルタ
        from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        
        bear_notes = []
        
        for keyword in keywords:
            try:
                # OSM Notes API にはテキスト検索がないため、
                # bbox内のすべてのNotesを取得してフィルタリング
                params = {
                    "bbox": f"{west},{south},{east},{north}",
                    "closed": 7,  # 7日以内にクローズされたものも含む
                    "limit": 100,
                    "format": "json"
                }
                
                response = self.session.get(
                    f"{OSM_API_BASE}/notes",
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()
                
                # キーワードでフィルタリング
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    comments = props.get("comments", [])
                    
                    # コメント内にキーワードが含まれるかチェック
                    has_keyword = False
                    for comment in comments:
                        text = comment.get("text", "").lower()
                        if keyword.lower() in text:
                            has_keyword = True
                            break
                    
                    if has_keyword:
                        coords = feature.get("geometry", {}).get("coordinates", [])
                        note = OSMNote(
                            id=props.get("id"),
                            lat=coords[1] if len(coords) > 1 else 0,
                            lon=coords[0] if len(coords) > 0 else 0,
                            status=props.get("status", "unknown"),
                            created_at=props.get("date_created", ""),
                            comments=comments
                        )
                        bear_notes.append(note)
                
                # レート制限対策
                time.sleep(1)
                
            except requests.RequestException as e:
                print(f"OSM Notes検索エラー ({keyword}): {e}")
                continue
        
        # 重複を除去
        seen_ids = set()
        unique_notes = []
        for note in bear_notes:
            if note.id not in seen_ids:
                seen_ids.add(note.id)
                unique_notes.append(note)
        
        return unique_notes
    
    def create_note(
        self,
        lat: float,
        lon: float,
        text: str,
        oauth_token: Optional[str] = None
    ) -> Optional[int]:
        """
        新しいNoteを作成
        
        注意: OSM APIへの書き込みにはOAuth認証が必要
        市民向けの熊目撃報告をOSM Notesに自動投稿する場合に使用
        """
        if not oauth_token:
            print("警告: OAuth認証なしではNoteを作成できません")
            return None
        
        # OAuth認証ヘッダーを設定
        headers = {
            "Authorization": f"Bearer {oauth_token}"
        }
        
        params = {
            "lat": lat,
            "lon": lon,
            "text": text
        }
        
        try:
            response = self.session.post(
                f"{OSM_API_BASE}/notes",
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            # レスポンスからNote IDを取得
            # XMLレスポンスをパースする必要がある
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.text)
            note_elem = root.find("note")
            if note_elem is not None:
                return int(note_elem.get("id"))
            
        except requests.RequestException as e:
            print(f"Note作成エラー: {e}")
        
        return None


def integrate_citizen_reports(
    osm_notes: List[OSMNote],
    ai_detections: List[Dict],
    distance_threshold_m: float = 500
) -> List[Dict]:
    """
    市民からのOSM Notes報告とAI検知結果を統合
    
    Args:
        osm_notes: OSM Notesから取得した熊関連情報
        ai_detections: AI検知システムからの検知結果
        distance_threshold_m: 同一と見なす距離閾値（メートル）
    
    Returns:
        統合された検知情報リスト
    """
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """2点間の距離（メートル）"""
        R = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        
        a = sin(delta_phi/2)**2 + cos(phi1)*cos(phi2)*sin(delta_lambda/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c
    
    integrated = []
    matched_note_ids = set()
    
    # AI検知を基準に市民報告をマッチング
    for detection in ai_detections:
        det_lat = detection.get("latitude")
        det_lon = detection.get("longitude")
        det_time = datetime.fromisoformat(detection.get("timestamp", "").replace("Z", "+00:00"))
        
        # 近くの市民報告を探す
        nearby_reports = []
        for note in osm_notes:
            distance = haversine_distance(det_lat, det_lon, note.lat, note.lon)
            
            if distance <= distance_threshold_m:
                # 時間的にも近いかチェック（前後2時間以内）
                try:
                    note_time = datetime.fromisoformat(note.created_at.replace("Z", "+00:00"))
                    time_diff = abs((det_time - note_time).total_seconds() / 3600)
                    
                    if time_diff <= 2:
                        nearby_reports.append({
                            "note_id": note.id,
                            "distance_m": distance,
                            "time_diff_h": time_diff
                        })
                        matched_note_ids.add(note.id)
                except ValueError:
                    pass
        
        # 統合レコードを作成
        integrated_record = {
            **detection,
            "source": "ai" if not nearby_reports else "ai+citizen",
            "citizen_reports": nearby_reports,
            "confidence_boost": 0.1 * len(nearby_reports)  # 市民報告があれば信頼度アップ
        }
        
        # 信頼度を調整（最大1.0）
        base_confidence = detection.get("confidence", 0.5)
        integrated_record["adjusted_confidence"] = min(
            1.0,
            base_confidence + integrated_record["confidence_boost"]
        )
        
        integrated.append(integrated_record)
    
    # AI検知とマッチしなかった市民報告を追加
    for note in osm_notes:
        if note.id not in matched_note_ids:
            integrated.append({
                "timestamp": note.created_at,
                "latitude": note.lat,
                "longitude": note.lon,
                "source": "citizen",
                "osm_note_id": note.id,
                "confidence": 0.5,  # 市民報告のみの場合は中程度の信頼度
                "adjusted_confidence": 0.5,
                "comments": [c.get("text", "") for c in note.comments]
            })
    
    return integrated


def convert_to_geojson(integrated_data: List[Dict]) -> Dict:
    """統合データをGeoJSON形式に変換"""
    features = []
    
    for item in integrated_data:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [item["longitude"], item["latitude"]]
            },
            "properties": {
                "timestamp": item.get("timestamp"),
                "source": item.get("source"),
                "confidence": item.get("confidence"),
                "adjusted_confidence": item.get("adjusted_confidence"),
                "device_id": item.get("device_id"),
                "osm_note_id": item.get("osm_note_id"),
                "citizen_reports": item.get("citizen_reports", [])
            }
        }
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }


# =============================================================================
# uMap 連携
# =============================================================================

class UMapExporter:
    """
    uMap形式へのエクスポート
    uMapは直接のAPI更新をサポートしていないため、
    GeoJSONファイルを生成してホスティングする形式
    """
    
    @staticmethod
    def export_for_umap(
        detections: List[Dict],
        output_path: str,
        map_config: Dict = None
    ):
        """
        uMapインポート用のGeoJSONを生成
        
        uMapの設定:
        1. uMap (https://umap.openstreetmap.fr/) でマップを作成
        2. "Remote data" でこのGeoJSONのURLを設定
        3. 自動更新を有効化（1分〜）
        """
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        
        for det in detections:
            # 緊急度に応じたアイコン色
            age_minutes = det.get("age_minutes", 0)
            if age_minutes < 30:
                color = "red"
                icon = "alert"
            elif age_minutes < 120:
                color = "orange"
                icon = "caution"
            else:
                color = "yellow"
                icon = "information"
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [det["longitude"], det["latitude"]]
                },
                "properties": {
                    "name": f"🐻 熊検知 ({det.get('confidence', 0)*100:.0f}%)",
                    "description": f"""
                        <b>検知時刻:</b> {det.get('timestamp', '不明')}<br>
                        <b>信頼度:</b> {det.get('confidence', 0)*100:.0f}%<br>
                        <b>デバイス:</b> {det.get('device_id', '不明')}<br>
                        <b>情報源:</b> {det.get('source', 'AI')}<br>
                    """,
                    "_umap_options": {
                        "color": color,
                        "iconClass": icon,
                        "showLabel": True
                    }
                }
            }
            geojson["features"].append(feature)
        
        # ファイルに保存
        import json
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        
        print(f"uMap用GeoJSONを保存: {output_path}")
        return output_path


# =============================================================================
# メイン
# =============================================================================

if __name__ == "__main__":
    # 使用例
    client = OSMNotesClient()
    
    # 南魚沼市付近の熊関連Notesを検索
    bbox = (138.80, 36.95, 139.10, 37.15)
    notes = client.search_bear_notes(bbox, days=30)
    
    print(f"熊関連のOSM Notes: {len(notes)} 件")
    
    for note in notes[:5]:
        print(f"  - Note #{note.id}: ({note.lat:.4f}, {note.lon:.4f})")
        print(f"    ステータス: {note.status}")
        print(f"    作成日: {note.created_at}")
