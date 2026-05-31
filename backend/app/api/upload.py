"""
app/api/upload.py  (v2 - avec LSTM+Attention et geofencing)
"""
import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from app.database import get_db
from app.models.orm import Chat, Position
from app.models.schemas import UploadResult, PredictionOut
from app.services.ingestion import ingest_csv
from app.services.geofencing import geofence_manager
from app.ml.lstm_attention import LSTMAttentionPredictor
from app.config import get_settings

log      = logging.getLogger(__name__)
settings = get_settings()
router   = APIRouter(tags=["upload & prediction"])


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload/{chat_id}", response_model=UploadResult)
async def upload_csv(
    chat_id:    int,
    background: BackgroundTasks,
    file:       UploadFile = File(...),
    db:         Session    = Depends(get_db),
):
    """
    Upload un CSV GPS → nettoie → insère → réentraîne LSTM+Attention.
    Le réentraînement se fait en arrière-plan (non bloquant).
    """
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Chat {chat_id} introuvable")
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, detail="Seuls les fichiers .csv sont acceptés")

    content = await file.read()
    try:
        inserted, skipped = ingest_csv(
            content, chat_id, db, cat.lat_home, cat.lon_home
        )
    except ValueError as e:
        raise HTTPException(400, detail=str(e))

    # Vérification geofencing pour la dernière position insérée
    background.add_task(_check_last_position_geofence, chat_id, db)
    # Réentraînement LSTM+Attention
    background.add_task(_retrain_attention, chat_id, db)

    return UploadResult(
        chat_id=chat_id,
        inserted=inserted,
        skipped=skipped,
        model_retrained=False,
    )


# ── Prédiction ────────────────────────────────────────────────────────────────

@router.get("/predict/{chat_id}", response_model=PredictionOut)
def predict_next(chat_id: int, db: Session = Depends(get_db)):
    """Prédit la prochaine position avec LSTM+Attention."""
    cat = db.query(Chat).filter(Chat.id == chat_id).first()
    if not cat:
        raise HTTPException(404, detail=f"Chat {chat_id} introuvable")

    predictor = LSTMAttentionPredictor(chat_id)
    if not predictor.is_trained():
        raise HTTPException(
            503,
            detail="Modèle non entraîné. Uploadez un CSV d'abord (≥ 56 points).",
        )

    rows = (
        db.query(Position)
        .filter(Position.chat_id == chat_id)
        .order_by(Position.ts.desc())
        .limit(settings.sequence_len)
        .all()
    )
    df = pd.DataFrame([{
        "ts":              r.ts,
        "latitude":        r.latitude,
        "longitude":       r.longitude,
        "distance_home_m": r.distance_home_m or 0.0,
        "vitesse_ms":      r.vitesse_ms or 0.0,
    } for r in reversed(rows)])

    result = predictor.predict_next(df)
    if result is None:
        raise HTTPException(503, detail="Prédiction impossible – données insuffisantes.")

    pred_lat, pred_lon = result
    return PredictionOut(
        chat_id=chat_id,
        predicted_latitude=pred_lat,
        predicted_longitude=pred_lon,
        model_version="lstm_attention_v2",
    )


# ── Tâches de fond ────────────────────────────────────────────────────────────

def _retrain_attention(chat_id: int, db: Session):
    """Charge toutes les positions et réentraîne LSTM+Attention."""
    try:
        rows = (
            db.query(Position)
            .filter(Position.chat_id == chat_id)
            .order_by(Position.ts)
            .all()
        )
        df = pd.DataFrame([{
            "ts":              r.ts,
            "latitude":        r.latitude,
            "longitude":       r.longitude,
            "distance_home_m": r.distance_home_m or 0.0,
            "vitesse_ms":      r.vitesse_ms or 0.0,
        } for r in rows])

        predictor = LSTMAttentionPredictor(chat_id)
        predictor.fit(df, fine_tune=True)   # utilise le modèle Kaggle si dispo
        log.info("LSTM+Attention réentraîné pour chat %d", chat_id)
    except Exception as exc:
        log.error("Réentraînement échoué pour chat %d : %s", chat_id, exc)


async def _check_last_position_geofence(chat_id: int, db: Session):
    """Vérifie la dernière position insérée contre les zones geofencing."""
    try:
        last = (
            db.query(Position)
            .filter(Position.chat_id == chat_id)
            .order_by(Position.ts.desc())
            .first()
        )
        if last:
            await geofence_manager.check_position(
                chat_id   = chat_id,
                lat       = last.latitude,
                lon       = last.longitude,
                timestamp = last.ts.isoformat(),
            )
    except Exception as exc:
        log.error("Geofence check échoué pour chat %d : %s", chat_id, exc)
