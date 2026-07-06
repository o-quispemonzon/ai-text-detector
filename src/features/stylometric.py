"""Features estilométricas: señales de estilo independientes del vocabulario.

Transformer de sklearn (picklable, usable dentro de un Pipeline). Captura
regularidades que distinguen prosa humana de generada: variedad léxica,
longitud de oraciones, uso de puntuación, mayúsculas, etc.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_RE = r"[a-z']+"
_SENT_RE = r"[.!?]+"


class StylometricFeatures(BaseEstimator, TransformerMixin):
    """Extrae features numéricas de estilo a partir de texto crudo."""

    feature_names = [
        "n_chars",
        "n_words",
        "avg_word_len",
        "n_sentences",
        "avg_sentence_words",
        "type_token_ratio",
        "stopword_ratio",
        "comma_per_char",
        "period_per_char",
        "semicolon_per_char",
        "exclaim_per_char",
        "question_per_char",
        "quote_per_char",
        "newline_per_char",
        "uppercase_ratio",
        "digit_ratio",
    ]

    def fit(self, X, y=None):  # noqa: N803 (convención sklearn)
        return self

    def transform(self, X) -> np.ndarray:  # noqa: N803
        s = pd.Series(list(X), dtype="string").fillna("")
        n_chars = s.str.len().clip(lower=1)
        words = s.str.lower().str.findall(_WORD_RE)
        n_words = words.str.len().clip(lower=1)
        n_sents = s.str.count(_SENT_RE).clip(lower=1)

        stop = ENGLISH_STOP_WORDS

        def avg_len(w: list[str]) -> float:
            return float(np.mean([len(x) for x in w])) if w else 0.0

        out = pd.DataFrame(
            {
                "n_chars": n_chars,
                "n_words": n_words,
                "avg_word_len": words.map(avg_len),
                "n_sentences": n_sents,
                "avg_sentence_words": n_words / n_sents,
                "type_token_ratio": words.map(lambda w: len(set(w)) / max(len(w), 1)),
                "stopword_ratio": words.map(lambda w: sum(x in stop for x in w) / max(len(w), 1)),
                "comma_per_char": s.str.count(",") / n_chars,
                "period_per_char": s.str.count(r"\.") / n_chars,
                "semicolon_per_char": s.str.count(";") / n_chars,
                "exclaim_per_char": s.str.count("!") / n_chars,
                "question_per_char": s.str.count(r"\?") / n_chars,
                "quote_per_char": s.str.count(r"[\"“”]") / n_chars,
                "newline_per_char": s.str.count("\n") / n_chars,
                "uppercase_ratio": s.str.count(r"[A-Z]") / n_chars,
                "digit_ratio": s.str.count(r"[0-9]") / n_chars,
            }
        )
        return out[self.feature_names].to_numpy(dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.feature_names)
