# Decisiones de arquitectura

Registro de decisiones técnicas del proyecto. Contexto: todo corre en una laptop
(i9-14900HX, RTX 4070 8GB, 16GB RAM, WSL2), sin nube, en un sprint de 7 días.

## 1. MLflow (local) en lugar de Weights & Biases

W&B en modo gratuito depende de su cloud; su modo self-hosted requiere levantar un servidor.
MLflow corre local con `mlflow ui`, sin cuenta, y es el estándar de facto en stacks
empresariales. Coherente con la restricción "solo local". Backend: SQLite, que es lo
recomendado desde MLflow 3.x (el file store quedó deprecado). La BD vive en
`~/.ai-text-detector/mlflow.db` (filesystem nativo de Linux) porque SQLite falla con
`disk I/O error` sobre NTFS/9p cuando el repo está en `/mnt/c` bajo WSL2; los
artefactos (modelos) sí son archivos planos y van junto al repo en `mlruns/`.

## 2. Manifest con SHA-256 en lugar de DVC

DVC sin remote storage aporta poca reproducibilidad extra y consume tiempo del sprint.
`data/manifest.json` registra hash, tamaño y procedencia de cada archivo, y el script de
descarga es idempotente. Cualquiera puede reproducir el dataset exacto y verificar
integridad. Trade-off aceptado: no hay time-travel de versiones de datos (no lo necesitamos:
los datasets DAIGT son estáticos).

## 3. uv como gestor de paquetes

Lockfile reproducible (`uv.lock`), resolución rápida, integración limpia con CI y Docker.
El índice de PyTorch cu124 se declara en `pyproject.toml`, de modo que `uv sync` instala
torch con CUDA sin pasos manuales.

## 4. DeBERTa-v3-small en lugar de base/large

Con 8GB de VRAM, small entrena con fp16 + gradient accumulation en ~1-2h, permitiendo
iterar dentro del sprint. En esta tarea la ganancia de modelos más grandes es marginal
frente al enfoque de char n-grams (ganador real de la competencia), lo que hace la
comparación clásico vs. transformer más interesante, no menos.

## 5. Split held-out por generador, además del split aleatorio

Un split aleatorio sobre DAIGT infla las métricas: el modelo memoriza estilos de
generadores vistos. Evaluamos también sobre generadores excluidos del entrenamiento para
medir generalización real. Es la decisión de evaluación más importante del proyecto.

## 6. CI valida, no entrena

Los runners gratuitos de GitHub Actions son CPU-only y con límites de tiempo. El CI corre
lint, tests y un smoke test del pipeline con fixtures sintéticas y modelo dummy. El
entrenamiento es un proceso aparte, reproducible vía Make + Docker. Es el patrón realista
en industria.

## 7. FastAPI como producto, Streamlit como demo

La API es el artefacto de producción: containerizada, testeada, con logging de
predicciones. Streamlit es un cliente visual que consume la API. Así el monitoreo vive en
el servicio y no se duplica en la UI.

## 8. Evidently para drift, sin Prometheus/Grafana

Evidently genera reportes de data drift (longitudes, vocabulario, distribución de scores)
en HTML, local y sin infraestructura. Un stack Prometheus+Grafana sería sobreingeniería
para una semana y no añadiría señal sobre el modelo.
