"""
app/ml/lstm_attention.py
─────────────────────────
LSTM amélioré avec mécanisme d'Attention (Bahdanau-style).

Pourquoi l'attention ?
──────────────────────
Sans attention : le LSTM résume toute la séquence dans un vecteur fixe.
Avec attention  : le modèle apprend à "regarder" les pas de temps les plus
                  pertinents. Ex : si le chat sort toujours à 19h, l'attention
                  donnera plus de poids aux observations de 19h précédentes.

Architecture
────────────
  Input [batch, SEQ_LEN, 6 features]
       ↓
  LSTM(64, return_sequences=True)   ← produit une sortie par pas de temps
       ↓
  AttentionLayer                    ← calcule un score d'importance par pas
       ↓                              et retourne la somme pondérée
  Dropout(0.2)
       ↓
  LSTM(32)
       ↓
  Dropout(0.2)
       ↓
  Dense(32, relu) → Dense(2)       ← [lat_norm, lon_norm]
       ↓
  MinMaxScaler.inverse_transform → [lat°, lon°]

Entraînement sur données réelles
──────────────────────────────────
  Le modèle est d'abord pré-entraîné sur les 228 chats néo-zélandais
  (transfer learning), puis fine-tuné sur les données du chat cible.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

from app.config import get_settings

log      = logging.getLogger(__name__)
settings = get_settings()

SEQ_LEN  = settings.sequence_len   # 6 steps
EPOCHS   = settings.lstm_epochs    # 50 max (early stopping)
BATCH    = settings.lstm_batch_size
MIN_ROWS = SEQ_LEN + 50


# ══════════════════════════════════════════════════════════════════════════════
# Couche d'Attention personnalisée (Keras Layer)
# ══════════════════════════════════════════════════════════════════════════════

class BahdanauAttention(layers.Layer):
    """
    Mécanisme d'attention additive (Bahdanau 2015).

    Pour chaque pas de temps h_t produit par le LSTM :
        score(t) = V · tanh(W · h_t)       [scalaire]
        alpha(t) = softmax(score(t))        [poids normalisé 0-1]
        context  = Σ alpha(t) × h_t        [vecteur de contexte]

    Le modèle apprend quels pas de temps sont les plus prédictifs.
    """
    def __init__(self, units: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.W = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1,     use_bias=False)

    def call(self, hidden_states):
        # hidden_states : [batch, seq_len, lstm_units]
        score  = self.V(tf.nn.tanh(self.W(hidden_states)))  # [batch, seq_len, 1]
        alpha  = tf.nn.softmax(score, axis=1)               # [batch, seq_len, 1]
        context = tf.reduce_sum(alpha * hidden_states, axis=1)  # [batch, lstm_units]
        return context, alpha

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.V.units})
        return config


# ══════════════════════════════════════════════════════════════════════════════
# Prédicteur principal
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAttentionPredictor:
    """
    LSTM + Attention. Compatible drop-in avec LSTMPredictor de base.
    Ajoute :
      • mécanisme d'attention
      • pré-entraînement sur dataset Kaggle (transfer learning)
      • fine-tuning sur données du chat cible
    """

    def __init__(self, chat_id: int):
        self.chat_id    = chat_id
        self.model_dir  = Path(settings.ml_model_dir) / str(chat_id)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pretrain_dir = Path(settings.ml_model_dir) / "pretrained"

        self.model:      Optional[keras.Model] = None
        self.scaler_in:  Optional[MinMaxScaler] = None
        self.scaler_out: Optional[MinMaxScaler] = None

    # ── API publique ──────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, fine_tune: bool = True) -> None:
        """
        Entraîne le modèle.
        Si un modèle pré-entraîné (Kaggle) existe → fine-tune dessus.
        Sinon → entraînement complet.
        """
        if len(df) < MIN_ROWS:
            log.warning("Pas assez de données (%d < %d) pour chat %d",
                        len(df), MIN_ROWS, self.chat_id)
            return

        X, y = self._build_sequences(df)
        if X is None:
            return

        # Essayer de charger le modèle pré-entraîné
        pretrained_path = self.pretrain_dir / "pretrained.keras"
        if fine_tune and pretrained_path.exists():
            log.info("Fine-tuning sur modèle pré-entraîné Kaggle pour chat %d", self.chat_id)
            self.model = keras.models.load_model(
                str(pretrained_path),
                custom_objects={"BahdanauAttention": BahdanauAttention}
            )
            # Dégeler toutes les couches pour le fine-tuning
            for layer in self.model.layers:
                layer.trainable = True
            self.model.compile(optimizer=keras.optimizers.Adam(1e-4), loss="mse", metrics=["mae"])
        else:
            log.info("Entraînement LSTM+Attention from scratch pour chat %d", self.chat_id)
            self.model = self._build_model(X.shape[1:])

        self.model.fit(
            X, y,
            epochs=EPOCHS,
            batch_size=BATCH,
            validation_split=0.1,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
            ],
        )
        self._save()
        log.info("LSTM+Attention entraîné pour chat %d (%d séquences)", self.chat_id, len(X))

    def predict_next(self, df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """Prédit la prochaine position (lat, lon)."""
        if self.model is None:
            self._load()
        if self.model is None:
            return None
        if len(df) < SEQ_LEN:
            return None

        features = self._extract_features(df.tail(SEQ_LEN))
        X        = self.scaler_in.transform(features).reshape(1, SEQ_LEN, -1)
        y_norm   = self.model.predict(X, verbose=0)[0]
        lat, lon = self.scaler_out.inverse_transform([y_norm])[0]
        return float(lat), float(lon)

    def get_attention_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Retourne les poids d'attention pour les SEQ_LEN derniers points.
        Utile pour visualiser quels pas de temps influencent la prédiction.
        """
        if self.model is None:
            self._load()
        if self.model is None:
            return None

        # Modèle intermédiaire qui expose les poids d'attention
        try:
            attn_layer = next(l for l in self.model.layers
                              if isinstance(l, BahdanauAttention))
        except StopIteration:
            return None

        features = self._extract_features(df.tail(SEQ_LEN))
        X        = self.scaler_in.transform(features).reshape(1, SEQ_LEN, -1)

        attn_model = keras.Model(
            inputs=self.model.input,
            outputs=attn_layer.output,
        )
        _, alphas = attn_model.predict(X, verbose=0)
        return alphas.squeeze()   # shape [SEQ_LEN]

    def is_trained(self) -> bool:
        return (self.model_dir / "model_attn.keras").exists()

    # ── Construction du modèle ─────────────────────────────────────────────────

    @staticmethod
    def _build_model(input_shape: Tuple) -> keras.Model:
        """
        Architecture LSTM + Attention.

        input_shape : (SEQ_LEN, n_features)
        """
        inputs = keras.Input(shape=input_shape)                          # [B, T, F]

        # Premier LSTM → produit une sortie par pas de temps
        x = layers.LSTM(64, return_sequences=True)(inputs)               # [B, T, 64]
        x = layers.Dropout(0.2)(x)

        # Couche d'attention : contexte + poids
        context, _ = BahdanauAttention(units=32)(x)                      # [B, 64]

        # Deuxième LSTM classique (séquence → vecteur)
        x2 = layers.LSTM(32)(x)                                          # [B, 32]
        x2 = layers.Dropout(0.2)(x2)

        # Fusionner contexte attention + sortie LSTM
        merged = layers.Concatenate()([context, x2])                     # [B, 96]

        x3  = layers.Dense(32, activation="relu")(merged)
        out = layers.Dense(2)(x3)                                        # [B, 2]

        model = keras.Model(inputs=inputs, outputs=out)
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ── Features ──────────────────────────────────────────────────────────────

    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        hours    = pd.to_datetime(df["ts"]).dt.hour + pd.to_datetime(df["ts"]).dt.minute / 60
        hour_sin = np.sin(2 * np.pi * hours / 24).values
        hour_cos = np.cos(2 * np.pi * hours / 24).values
        return np.column_stack([
            df["latitude"].values,
            df["longitude"].values,
            hour_sin,
            hour_cos,
            df["distance_home_m"].fillna(0).values,
            df["vitesse_ms"].fillna(0).values,
        ])

    def _build_sequences(self, df: pd.DataFrame):
        feats   = self._extract_features(df)
        targets = df[["latitude", "longitude"]].values

        self.scaler_in  = MinMaxScaler()
        self.scaler_out = MinMaxScaler()
        feats_s   = self.scaler_in.fit_transform(feats)
        targets_s = self.scaler_out.fit_transform(targets)

        X, y = [], []
        for i in range(len(feats_s) - SEQ_LEN):
            X.append(feats_s[i: i + SEQ_LEN])
            y.append(targets_s[i + SEQ_LEN])

        if not X:
            return None, None
        return np.array(X), np.array(y)

    # ── Persistance ────────────────────────────────────────────────────────────

    def _save(self):
        self.model.save(self.model_dir / "model_attn.keras")
        joblib.dump(self.scaler_in,  self.model_dir / "scaler_in.pkl")
        joblib.dump(self.scaler_out, self.model_dir / "scaler_out.pkl")

    def _load(self):
        model_path = self.model_dir / "model_attn.keras"
        if not model_path.exists():
            return
        self.model = keras.models.load_model(
            str(model_path),
            custom_objects={"BahdanauAttention": BahdanauAttention}
        )
        self.scaler_in  = joblib.load(self.model_dir / "scaler_in.pkl")
        self.scaler_out = joblib.load(self.model_dir / "scaler_out.pkl")
        log.info("LSTM+Attention chargé pour chat %d", self.chat_id)
