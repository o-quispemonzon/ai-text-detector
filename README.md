# AI Text Detector — ¿humano o LLM?

Clasificador de ensayos humanos vs. generados por LLM, basado en la competencia de Kaggle
[LLM - Detect AI Generated Text](https://www.kaggle.com/competitions/llm-detect-ai-generated-text)
y los datasets comunitarios DAIGT-v2/v3. El proyecto compara dos enfoques de modelado
(clásico: TF-IDF + LightGBM/LogReg vs. transformer: DeBERTa-v3) y lo envuelve en un stack
de MLOps completo que corre **100% en local**: MLflow, Docker, GitHub Actions, FastAPI,
Evidently y Streamlit.

> 🚧 Proyecto en construcción (sprint de 7 días). Este README se completa al final con
> resultados, comparación de enfoques y diagrama de arquitectura.

## Quickstart (WSL2 / Linux)

Requisitos: [uv](https://docs.astral.sh/uv/), cuenta de Kaggle con token de API.

```bash
# 1. Instalar dependencias (crea .venv con Python 3.11)
make setup

# 2. Credenciales de Kaggle (una sola vez)
#    kaggle.com -> Settings -> API -> "Create New Token" descarga kaggle.json
mkdir -p ~/.kaggle && mv /ruta/a/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# 3. Descargar datasets DAIGT-v2 y v3 (genera data/manifest.json con hashes)
make data

# 4. Verificar que todo está en orden
make lint test

# 5. UI de MLflow para explorar experimentos
make mlflow   # http://localhost:5000
```

## Estructura

```
configs/       Configuración YAML (datos, modelos)
data/          Datasets (gitignored; manifest.json versiona hashes y procedencia)
docker/        Dockerfiles de entrenamiento y serving
docs/          Decisiones de arquitectura y diagrama
monitoring/    Drift (Evidently) y simulación de tráfico
notebooks/     EDA y análisis de errores
src/           Código fuente: data / features / models / serving / utils
tests/         Tests unitarios (pytest)
```

## Decisiones de arquitectura

Ver [docs/decisions.md](docs/decisions.md): por qué MLflow y no W&B, por qué manifest en
lugar de DVC, por qué el CI no entrena modelos, etc.

## Licencia y datos

Los datasets DAIGT son de la comunidad de Kaggle (créditos a Darek Kłeczek y colaboradores).
Este repositorio no redistribuye datos; el script `make data` los descarga de la fuente.
