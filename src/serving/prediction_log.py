"""Logging de predicciones para monitoreo (producción simulada).

Se registra una línea JSON por predicción, SIN el texto crudo (privacidad):
solo hash, estadísticas ligeras y el score. Es el insumo del reporte de
drift de Evidently (monitoring/drift_report.py).

JSONL en lugar de SQLite: append atómico que funciona igual en NTFS/9p,
contenedores y CI, sin problemas de locks.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

_LOCK = threading.Lock()


def _log_path() -> Path:
    path = Path(os.environ.get("PREDICTION_LOG", "monitoring/logs/predictions.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def log_prediction(text: str, proba_ai: float, latency_ms: float, model_name: str) -> None:
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "n_chars": len(text),
        "n_words": len(text.split()),
        "proba_ai": round(float(proba_ai), 6),
        "label": int(proba_ai >= 0.5),
        "latency_ms": round(float(latency_ms), 2),
        "model_name": model_name,
    }
    line = json.dumps(record) + "\n"
    with _LOCK, _log_path().open("a", encoding="utf-8") as f:
        f.write(line)
