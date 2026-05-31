.PHONY: help up down build logs test prepare-data pretrain seed-db clean

help:
	@echo ""
	@echo "  🐱 CatTracker v2 — commandes disponibles"
	@echo "  ────────────────────────────────────────────────────────────"
	@echo "  make up            Démarrer tous les services Docker"
	@echo "  make down          Arrêter tous les services"
	@echo "  make build         Reconstruire les images"
	@echo "  make logs          Voir les logs en temps réel"
	@echo ""
	@echo "  ── Dataset Kaggle ──────────────────────────────────────────"
	@echo "  make prepare-data  Nettoyer le dataset Kaggle → CSV propres"
	@echo "  make pretrain      Pré-entraîner LSTM+Attention (Kaggle)"
	@echo "  make seed-db       Charger un chat Kaggle dans la DB"
	@echo ""
	@echo "  ── Tests ───────────────────────────────────────────────────"
	@echo "  make test          Lancer les 23 tests unitaires"
	@echo ""
	@echo "  ── Nettoyage ───────────────────────────────────────────────"
	@echo "  make clean         Supprimer volumes + images Docker"
	@echo ""

## ── Docker ───────────────────────────────────────────────────────────────────

up:
	cp -n .env.example .env 2>/dev/null || true
	docker compose up -d --build
	@echo ""
	@echo "  ✅  Stack démarrée !"
	@echo "  Dashboard  : http://localhost:8501"
	@echo "  API Swagger: http://localhost:8000/docs"
	@echo ""

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

## ── Dataset & ML ─────────────────────────────────────────────────────────────

prepare-data:
	@echo "📊 Préparation du dataset Kaggle..."
	python scripts/prepare_kaggle_dataset.py --min-points 200
	@echo "✅ Données prêtes dans data/kaggle_clean.csv"

pretrain:
	@echo "🧠 Pré-entraînement LSTM+Attention sur données Kaggle..."
	@echo "   (Peut prendre 5-15 min selon votre machine)"
	python scripts/pretrain_kaggle.py --max-cats 50 --epochs 30
	@echo "✅ Modèle pré-entraîné dans backend/ml_models/pretrained/"

pretrain-docker:
	@echo "🧠 Pré-entraînement dans le container Docker..."
	docker compose exec backend python /app/data/../scripts/pretrain_kaggle.py \
		--max-cats 50 --epochs 30

seed-db:
	@echo "🐱 Chargement de Luna (chat Kaggle avec 5151 points)..."
	@python3 -c "\
import json; \
stats = json.load(open('data/kaggle_stats.json')); \
cats = sorted(stats['cats'], key=lambda c: c['n_points'], reverse=True); \
top = cats[0]; \
print(f'Chat sélectionné: {top[\"name\"]} ({top[\"n_points\"]} pts)'); \
print(f'Position maison: {top[\"lat_home\"]}, {top[\"lon_home\"]}') \
"
	@# Uploader le CSV du chat avec le plus de points
	@BEST_CAT=$$(python3 -c "import json; s=json.load(open('data/kaggle_stats.json')); print(sorted(s['cats'],key=lambda c:c['n_points'],reverse=True)[0]['name'])") && \
	 FILE="data/kaggle_per_cat/$${BEST_CAT}.csv" && \
	 echo "Upload: $$FILE" && \
	 curl -s -X POST "http://localhost:8000/upload/1" \
		-F "file=@$$FILE" | python3 -m json.tool
	@echo ""
	@echo "✅ Données réelles chargées — ouvrez http://localhost:8501"

## ── Tests ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

## ── Nettoyage ─────────────────────────────────────────────────────────────────

clean:
	docker compose down -v --rmi local
