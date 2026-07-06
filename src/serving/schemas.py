"""Contratos de la API de inferencia (Pydantic)."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_TEXT_CHARS = 50_000


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class Prediction(BaseModel):
    proba_ai: float = Field(ge=0.0, le=1.0, description="P(texto generado por IA)")
    label: int = Field(description="1 = IA, 0 = humano (umbral 0.5)")


class PredictResponse(Prediction):
    model_name: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    predictions: list[Prediction]
    model_name: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool
