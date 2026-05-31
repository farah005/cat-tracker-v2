"""
app/api/alerts.py
──────────────────
Endpoints WebSocket et REST pour le geofencing.

WebSocket
─────────
  GET /ws/{chat_id}
    → Le frontend se connecte ici et reçoit les alertes en temps réel.
    → Format JSON reçu :
        {
          "chat_id": 1,
          "zone_id": "zone_maison",
          "zone_name": "Zone maison",
          "event": "exit",           // "exit" ou "enter"
          "latitude": 48.859,
          "longitude": 2.356,
          "timestamp": "2025-05-01T19:10:00",
          "message": "🚨 Zone maison : votre chat a QUITTÉ la zone !"
        }

REST – Zones
────────────
  GET    /zones/{chat_id}          → liste des zones du chat
  POST   /zones/{chat_id}          → créer une zone
  DELETE /zones/{chat_id}/{zone_id} → supprimer une zone

REST – Simulation
─────────────────
  POST /alerts/simulate/{chat_id}  → simule une position pour tester les alertes
                                     (utile en démo / sans collier réel)
"""
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.orm import Chat, Position
from app.services.geofencing import (
    Zone, ZoneType,
    ws_manager, geofence_manager,
)

router = APIRouter(tags=["alertes & geofencing"])


# ══════════════════════════════════════════════════════════════════════════════
# Schémas Pydantic
# ══════════════════════════════════════════════════════════════════════════════

class ZoneCreateCircle(BaseModel):
    name:       str
    zone_type:  ZoneType = ZoneType.CIRCLE
    center_lat: float
    center_lon: float
    radius_m:   float = 200.0
    color:      str   = "#e94560"


class ZoneCreatePolygon(BaseModel):
    name:           str
    zone_type:      ZoneType = ZoneType.POLYGON
    polygon_points: List[List[float]]   # [[lat,lon], ...]
    color:          str = "#f39c12"


class SimulatePosition(BaseModel):
    latitude:  float
    longitude: float
    timestamp: Optional[str] = None


class ZoneOut(BaseModel):
    zone_id:        str
    chat_id:        int
    name:           str
    zone_type:      ZoneType
    center_lat:     Optional[float] = None
    center_lon:     Optional[float] = None
    radius_m:       Optional[float] = None
    polygon_points: Optional[List[List[float]]] = None
    color:          str
    active:         bool


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket endpoint
# ══════════════════════════════════════════════════════════════════════════════

@router.websocket("/ws/{chat_id}")
async def websocket_endpoint(chat_id: int, ws: WebSocket):
    """
    Connexion WebSocket pour recevoir les alertes en temps réel.

    Protocole :
      - Connexion → le serveur envoie {"type":"connected","chat_id":N}
      - Alerte    → le serveur envoie l'objet Alert en JSON
      - Le client peut envoyer {"type":"ping"} → réponse {"type":"pong"}
    """
    await ws_manager.connect(chat_id, ws)
    import json
    try:
        # Message de bienvenue
        await ws.send_text(json.dumps({
            "type":    "connected",
            "chat_id": chat_id,
            "message": f"✅ Connecté aux alertes pour le chat {chat_id}",
            "active_zones": len(geofence_manager.get_zones(chat_id)),
        }))

        # Boucle de lecture (le client peut envoyer des pings)
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except Exception:
                pass

    except WebSocketDisconnect:
        ws_manager.disconnect(chat_id, ws)


# ══════════════════════════════════════════════════════════════════════════════
# Gestion des zones
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/zones/{chat_id}", response_model=List[ZoneOut])
def list_zones(chat_id: int, db: Session = Depends(get_db)):
    """Liste toutes les zones de geofencing d'un chat."""
    _ensure_cat(chat_id, db)
    zones = geofence_manager.get_zones(chat_id)
    return [_zone_to_out(z) for z in zones]


@router.post("/zones/{chat_id}/circle", response_model=ZoneOut, status_code=201)
def create_circle_zone(
    chat_id: int,
    payload: ZoneCreateCircle,
    db:      Session = Depends(get_db),
):
    """Créer une zone circulaire (centre + rayon en mètres)."""
    _ensure_cat(chat_id, db)
    zone = Zone(
        zone_id    = str(uuid.uuid4())[:8],
        chat_id    = chat_id,
        name       = payload.name,
        zone_type  = ZoneType.CIRCLE,
        center_lat = payload.center_lat,
        center_lon = payload.center_lon,
        radius_m   = payload.radius_m,
        color      = payload.color,
    )
    geofence_manager.add_zone(zone)
    return _zone_to_out(zone)


@router.post("/zones/{chat_id}/polygon", response_model=ZoneOut, status_code=201)
def create_polygon_zone(
    chat_id: int,
    payload: ZoneCreatePolygon,
    db:      Session = Depends(get_db),
):
    """Créer une zone polygonale (liste de points [[lat,lon], ...])."""
    _ensure_cat(chat_id, db)
    if len(payload.polygon_points) < 3:
        raise HTTPException(400, "Un polygone nécessite au moins 3 points")
    zone = Zone(
        zone_id        = str(uuid.uuid4())[:8],
        chat_id        = chat_id,
        name           = payload.name,
        zone_type      = ZoneType.POLYGON,
        polygon_points = payload.polygon_points,
        color          = payload.color,
    )
    geofence_manager.add_zone(zone)
    return _zone_to_out(zone)


@router.delete("/zones/{chat_id}/{zone_id}", status_code=204)
def delete_zone(chat_id: int, zone_id: str, db: Session = Depends(get_db)):
    """Supprimer une zone de geofencing."""
    _ensure_cat(chat_id, db)
    if not geofence_manager.remove_zone(zone_id):
        raise HTTPException(404, f"Zone {zone_id} introuvable")


# ══════════════════════════════════════════════════════════════════════════════
# Simulation (test sans collier réel)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/alerts/simulate/{chat_id}")
async def simulate_position(
    chat_id: int,
    payload: SimulatePosition,
    db:      Session = Depends(get_db),
):
    """
    Simule une position GPS pour tester le geofencing sans collier réel.
    Tous les clients WebSocket connectés reçoivent les alertes.
    """
    _ensure_cat(chat_id, db)
    ts = payload.timestamp or datetime.now(timezone.utc).isoformat()
    alerts = await geofence_manager.check_position(
        chat_id   = chat_id,
        lat       = payload.latitude,
        lon       = payload.longitude,
        timestamp = ts,
    )
    return {
        "chat_id":       chat_id,
        "position":      {"lat": payload.latitude, "lon": payload.longitude},
        "alerts_fired":  len(alerts),
        "alerts":        [a.to_json() for a in alerts],
        "ws_clients":    ws_manager.active_count(chat_id),
    }


@router.get("/alerts/status")
def ws_status():
    """Retourne le nombre de clients WebSocket connectés par chat."""
    from app.services.geofencing import ws_manager as wm
    return {
        "connections": {
            str(cid): wm.active_count(cid)
            for cid in wm._connections
        },
        "total_zones": len(geofence_manager.get_all_zones()),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_cat(chat_id: int, db: Session):
    if not db.query(Chat).filter(Chat.id == chat_id).first():
        raise HTTPException(404, f"Chat {chat_id} introuvable")


def _zone_to_out(z: Zone) -> ZoneOut:
    return ZoneOut(
        zone_id        = z.zone_id,
        chat_id        = z.chat_id,
        name           = z.name,
        zone_type      = z.zone_type,
        center_lat     = z.center_lat,
        center_lon     = z.center_lon,
        radius_m       = z.radius_m,
        polygon_points = z.polygon_points,
        color          = z.color,
        active         = z.active,
    )
