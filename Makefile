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
