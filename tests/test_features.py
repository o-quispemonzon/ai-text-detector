"""Tests de las features estilométricas y del ensamblado de pipelines."""

import io

import joblib
import numpy as np

from src.features.stylometric import StylometricFeatures
from src.features.vectorizers import build_features

TEXTS = [
    "This is a simple essay. It has two sentences!",
    "Driverless cars, however, are the future; many disagree.\nNew paragraph here?",
    "Short one.",
]

CFG_FEATURES = {
    "tfidf_word": {"ngram_min": 1, "ngram_max": 2, "min_df": 1, "max_features": 500},
    "tfidf_char": {"ngram_min": 3, "ngram_max": 4, "min_df": 1, "max_features": 500},
}


def test_stylometric_shape_and_no_nans():
    out = StylometricFeatures().fit_transform(TEXTS)
    assert out.shape == (3, len(StylometricFeatures.feature_names))
    assert np.isfinite(out).all()


def test_stylometric_is_deterministic():
    a = StylometricFeatures().fit_transform(TEXTS)
    b = StylometricFeatures().fit_transform(TEXTS)
    np.testing.assert_array_equal(a, b)


def test_stylometric_counts_are_sensible():
    out = StylometricFeatures().fit_transform(["One two three. Four five!"])
    row = dict(zip(StylometricFeatures.feature_names, out[0], strict=True))
    assert row["n_words"] == 5
    assert row["n_sentences"] == 2


def test_stylometric_is_picklable():
    buf = io.BytesIO()
    joblib.dump(StylometricFeatures(), buf)
    buf.seek(0)
    restored = joblib.load(buf)
    np.testing.assert_array_equal(
        restored.fit_transform(TEXTS), StylometricFeatures().fit_transform(TEXTS)
    )


def test_build_features_full_and_stylometric():
    full = build_features(CFG_FEATURES, kind="full")
    x = full.fit_transform(TEXTS)
    assert x.shape[0] == 3
    assert x.shape[1] > len(StylometricFeatures.feature_names)

    style = build_features(CFG_FEATURES, kind="stylometric")
    assert style.fit_transform(TEXTS).shape == (3, len(StylometricFeatures.feature_names))


def test_empty_text_does_not_crash():
    out = StylometricFeatures().fit_transform(["", "   ", None])
    assert np.isfinite(out).all()
