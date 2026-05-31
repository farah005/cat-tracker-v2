"""
app/services/geofencing.py
───────────────────────────
Geofencing + gestionnaire WebSocket.

Fonctionnement
──────────────
1. Chaque chat peut avoir une ou plusieurs zones définies (cercle ou polygone).
2. Quand une nouvelle position est reçue, on vérifie si le chat est
   dans/hors de chaque zone.
3. Si le chat SORT d'une zone → alerte "exit"
   Si le chat ENTRE dans une zone → alerte "enter"
4. L'alerte est broadcastée à tous les clients WebSocket connectés
   pour ce chat (le frontend reçoit la notification en temps réel).

Types de zones supportés
─────────────────────────
  - CIRCLE  : centre (lat, lon) + rayon en mètres
  - POLYGON : liste de points [(lat1,lon1), (lat2,lon2), ...]

Stockage
────────
  Les zones sont stockées en mémoire (dict) pour la démo.
  En production → table PostgreSQL `zones`.
"""
import asyncio
import json
import logging
import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set

from fastapi import WebSocket
from shapely.geometry import Point, Polygon

from app.services.geo import haversine_m

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Modèles de zones
# ══════════════════════════════════════════════════════════════════════════════

class ZoneType(str, Enum):
    CIRCLE  = "circle"
    POLYGON = "polygon"


@dataclass
class Zone:
    zone_id:   str
    chat_id:   int
    name:      str
    zone_type: ZoneType

    # Pour CIRCLE
    center_lat: Optional[float] = None
    center_lon: Optional[float] = None
    radius_m:   Optional[float] = None

    # Pour POLYGON [[lat,lon], ...]
    polygon_points: Optional[List[List[float]]] = None

    color:  str = "#e94560"
    active: bool = True

    def contains(self, lat: float, lon: float) -> bool:
        """Retourne True si le point (lat, lon) est dans la zone."""
        if not self.active:
            return True   # zone inactive → on considère toujours "inside"

        if self.zone_type == ZoneType.CIRCLE:
            dist = haversine_m(lat, lon, self.center_lat, self.center_lon)
            return dist <= self.radius_m

        elif self.zone_type == ZoneType.POLYGON:
            point   = Point(lon, lat)   # Shapely: (x=lon, y=lat)
            polygon = Polygon([(p[1], p[0]) for p in self.polygon_points])
            return polygon.contains(point)

        return True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Alert:
    chat_id:   int
    zone_id:   str
    zone_name: str
    event:     str          # "exit" | "enter"
    latitude:  float
    longitude: float
    timestamp: str
    message:   str

    def to_json(self) -> str:
        return json.dumps(asdict(self))


# ══════════════════════════════════════════════════════════════════════════════
# Gestionnaire WebSocket
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """
    Gère les connexions WebSocket actives.
    Chaque client se connecte sur /ws/{chat_id} et reçoit les alertes
    pour ce chat en temps réel.
    """
    def __init__(self):
        # chat_id → set de WebSocket actifs
        self._connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, chat_id: int, ws: WebSocket):
        await ws.accept()
        self._connections.setdefault(chat_id, set()).add(ws)
        log.info("WS connecté pour chat %d (total: %d)",
                 chat_id, len(self._connections[chat_id]))

    def disconnect(self, chat_id: int, ws: WebSocket):
        if chat_id in self._connections:
            self._connections[chat_id].discard(ws)
        log.info("WS déconnecté pour chat %d", chat_id)

    async def broadcast(self, chat_id: int, message: str):
        """Envoie un message JSON à tous les clients connectés pour chat_id."""
        dead = set()
        for ws in self._connections.get(chat_id, set()):
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[chat_id].discard(ws)

    async def broadcast_all(self, message: str):
        """Envoie à tous les chats (ex: message système)."""
        for chat_id in list(self._connections.keys()):
            await self.broadcast(chat_id, message)

    def active_count(self, chat_id: int) -> int:
        return len(self._connections.get(chat_id, set()))


# ══════════════════════════════════════════════════════════════════════════════
# Gestionnaire de zones et détection
# ══════════════════════════════════════════════════════════════════════════════

class GeofenceManager:
    """
    Stocke les zones et détecte les transitions (enter/exit).
    Utilise un état interne pour suivre si le chat était déjà dans la zone.
    """
    def __init__(self, ws_manager: ConnectionManager):
        self._zones: Dict[str, Zone] = {}
        # (chat_id, zone_id) → True si le chat était dans la zone au dernier check
        self._last_state: Dict[tuple, bool] = {}
        self._ws = ws_manager

    # ── Gestion des zones ─────────────────────────────────────────────────────

    def add_zone(self, zone: Zone) -> None:
        self._zones[zone.zone_id] = zone
        log.info("Zone ajoutée : %s (%s) pour chat %d", zone.name, zone.zone_type, zone.chat_id)

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self._zones:
            del self._zones[zone_id]
            return True
        return False

    def get_zones(self, chat_id: int) -> List[Zone]:
        return [z for z in self._zones.values() if z.chat_id == chat_id]

    def get_all_zones(self) -> List[Zone]:
        return list(self._zones.values())

    # ── Vérification géographique ─────────────────────────────────────────────

    async def check_position(
        self, chat_id: int, lat: float, lon: float, timestamp: str
    ) -> List[Alert]:
        """
        Vérifie la position par rapport à toutes les zones du chat.
        Retourne la liste des alertes générées (enter/exit).
        Broadcaste automatiquement via WebSocket.
        """
        alerts = []
        zones  = self.get_zones(chat_id)

        for zone in zones:
            key        = (chat_id, zone.zone_id)
            inside_now = zone.contains(lat, lon)
            was_inside = self._last_state.get(key, True)  # défaut: on suppose "inside"

            self._last_state[key] = inside_now

            if was_inside and not inside_now:
                event = "exit"
                msg   = f"🚨 {zone.name} : votre chat a QUITTÉ la zone !"
            elif not was_inside and inside_now:
                event = "enter"
                msg   = f"✅ {zone.name} : votre chat est RENTRÉ dans la zone."
            else:
                continue   # pas de changement d'état

            alert = Alert(
                chat_id=chat_id,
                zone_id=zone.zone_id,
                zone_name=zone.name,
                event=event,
                latitude=lat,
                longitude=lon,
                timestamp=timestamp,
                message=msg,
            )
            alerts.append(alert)
            await self._ws.broadcast(chat_id, alert.to_json())
            log.info("ALERTE [%s] chat=%d zone=%s", event.upper(), chat_id, zone.name)

        return alerts


# ══════════════════════════════════════════════════════════════════════════════
# Singletons partagés (importés par les routers)
# ══════════════════════════════════════════════════════════════════════════════

ws_manager      = ConnectionManager()
geofence_manager = GeofenceManager(ws_manager)
