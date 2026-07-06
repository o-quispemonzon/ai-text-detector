.PHONY: setup data prepare lint format test mlflow clean

# En WSL2, si el repo vive en /mnt/c (NTFS), el venv se crea en el disco ext4
# de Linux: crear miles de archivos pequeños sobre NTFS vía 9P es lentísimo.
ifneq (,$(findstring /mnt/,$(CURDIR)))
export UV_PROJECT_ENVIRONMENT := $(HOME)/.venvs/ai-text-detector
endif

## Instala todas las dependencias (incluye extras) con uv
setup:
	uv sync --all-extras

## Descarga los datasets DAIGT y genera data/manifest.json
data:
	uv run python -m src.data.download

## Limpieza + splits train/val/test_iid/test_ood en data/processed/
prepare:
	uv run python -m src.data.prepare

## Entrena los modelos clásicos con tracking en MLflow
train-classic:
	uv run python -m src.models.train_classic

## Fine-tuning del transformer (requiere GPU; ~30-60 min en RTX 4070)
train-transformer:
	uv run python -m src.models.train_transformer

## Exporta el mejor modelo (por AUC OOD) desde MLflow a models/
export-model:
	uv run python -m src.models.export_best

## API de inferencia local en http://localhost:8000 (docs en /docs)
serve:
	uv run uvicorn src.serving.app:app --reload --port 8000

## Imagen Docker de serving y ejecución
docker-build:
	docker build -f docker/Dockerfile.serve -t ai-text-detector:latest .

docker-run:
	docker run --rm -p 8000:8000 ai-text-detector:latest

## Demo Streamlit (requiere la API corriendo: make serve)
streamlit:
	uv run streamlit run app_streamlit.py

## Simula tráfico contra la API y genera el reporte de drift
simulate-traffic:
	uv run python monitoring/simulate_traffic.py --n 300

drift-report:
	uv run python monitoring/drift_report.py

## Lint + verificación de formato
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

## Autoformatea y corrige lo corregible
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

## Tests unitarios
test:
	uv run pytest

## UI de MLflow en http://localhost:5000 (BD en ~/.ai-text-detector, ver src/utils/tracking.py)
mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///$(HOME)/.ai-text-detector/mlflow.db --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__
