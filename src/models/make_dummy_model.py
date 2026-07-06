"""Genera un modelo mínimo para smoke tests (CI y build local sin entrenar).

Uso:
    python -m src.models.make_dummy_model

Entrena un pipeline real (misma arquitectura que producción) sobre datos
sintéticos y lo escribe en models/. NO es un modelo útil: existe para que el
CI pueda construir y probar la imagen Docker sin datasets ni GPU.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.vectorizers import build_features
from src.utils.config import resolve_path
from src.utils.log import get_logger

logger = get_logger(__name__)

CFG_FEATURES = {
    "tfidf_word": {"ngram_min": 1, "ngram_max": 1, "min_df": 1, "max_features": 500},
    "tfidf_char": {"ngram_min": 3, "ngram_max": 3, "min_df": 1, "max_features": 500},
}


def main() -> int:
    human = [f"my essay about school topic {i} was written with care and effort" for i in range(30)]
    ai = [f"as a language model i will now generate structured essay {i}" for i in range(30)]
    texts, labels = pd.Series(human + ai), [0] * 30 + [1] * 30

    pipe = Pipeline(
        [
            ("features", build_features(CFG_FEATURES, kind="full")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(texts, labels)

    out_dir = resolve_path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib", compress=3)
    meta = {"model_name": "dummy-smoke", "note": "solo para CI/smoke, no usar en serio"}
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Modelo dummy escrito en %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
