"""API de inferencia: ¿el texto fue escrito por un humano o por un LLM?

Ejecución local:
    uvicorn src.serving.app:app --reload --port 8000

El modelo es un Pipeline sklearn completo (features + clasificador) exportado
por src/models/export_best.py a models/model.joblib. Ruta configurable vía
MODEL_PATH. Cada predicción se registra (sin texto crudo) para monitoreo.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from src.serving.prediction_log import log_prediction
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    Prediction,
    PredictRequest,
    PredictResponse,
)


def _model_path() -> Path:
    return Path(os.environ.get("MODEL_PATH", "models/model.joblib"))


def _load_metadata(model_path: Path) -> dict:
    meta_path = model_path.with_name("model_meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {"model_name": model_path.stem}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = _model_path()
    if not path.exists():
        raise RuntimeError(
            f"No existe el modelo en {path}. Ejecuta primero "
            "`python -m src.models.export_best` o define MODEL_PATH."
        )
    app.state.model = joblib.load(path)
    app.state.meta = _load_metadata(path)
    yield


app = FastAPI(
    title="AI Text Detector",
    description="Clasifica ensayos como escritos por humano o generados por un LLM.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=app.state.meta.get("model_name", "unknown"),
        model_loaded=app.state.model is not None,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    t0 = time.perf_counter()
    proba = float(app.state.model.predict_proba([req.text])[0, 1])
    latency_ms = (time.perf_counter() - t0) * 1000
    model_name = app.state.meta.get("model_name", "unknown")
    log_prediction(req.text, proba, latency_ms, model_name)
    return PredictResponse(
        proba_ai=proba, label=int(proba >= 0.5), model_name=model_name, latency_ms=latency_ms
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    if any(not t.strip() for t in req.texts):
        raise HTTPException(status_code=422, detail="Hay textos vacíos en el batch")
    t0 = time.perf_counter()
    probas = app.state.model.predict_proba(req.texts)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000
    model_name = app.state.meta.get("model_name", "unknown")
    per_text = latency_ms / max(len(req.texts), 1)
    for text, proba in zip(req.texts, probas, strict=True):
        log_prediction(text, float(proba), per_text, model_name)
    return BatchPredictResponse(
        predictions=[Prediction(proba_ai=float(p), label=int(p >= 0.5)) for p in probas],
        model_name=model_name,
        latency_ms=latency_ms,
    )
