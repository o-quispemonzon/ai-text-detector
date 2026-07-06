"""Construcción de bloques de features a partir de la config YAML.

Todo se emite en float32: con ~175k features TF-IDF sobre 42k documentos,
float64 duplica la memoria sin ganancia de calidad (importante en WSL2,
que por defecto solo dispone de la mitad de la RAM del equipo).
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features.stylometric import StylometricFeatures


def _to_float32(x):
    return x.astype(np.float32)


def build_features(cfg_features: dict, kind: str = "full") -> FeatureUnion:
    """Crea el extractor de features.

    kind="full":        TF-IDF word + TF-IDF char + estilométricas.
    kind="stylometric": solo estilométricas (ablación: ¿cuánto rinde el estilo?).
    """
    style = Pipeline(
        [
            ("extract", StylometricFeatures()),
            ("scale", StandardScaler()),
            ("cast", FunctionTransformer(_to_float32)),
        ]
    )
    if kind == "stylometric":
        return FeatureUnion([("style", style)])
    if kind != "full":
        raise ValueError(f"kind desconocido: {kind!r}")

    w = cfg_features["tfidf_word"]
    c = cfg_features["tfidf_char"]
    return FeatureUnion(
        [
            (
                "tfidf_word",
                TfidfVectorizer(
                    ngram_range=(w["ngram_min"], w["ngram_max"]),
                    min_df=w["min_df"],
                    max_features=w["max_features"],
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            (
                "tfidf_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(c["ngram_min"], c["ngram_max"]),
                    min_df=c["min_df"],
                    max_features=c["max_features"],
                    sublinear_tf=True,
                    strip_accents="unicode",
                    dtype=np.float32,
                ),
            ),
            ("style", style),
        ]
    )
