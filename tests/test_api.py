"""Tests de la API de inferencia con un modelo real (chico) en fixture."""

import json

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.vectorizers import build_features

CFG_FEATURES = {
    "tfidf_word": {"ngram_min": 1, "ngram_max": 1, "min_df": 1, "max_features": 300},
    "tfidf_char": {"ngram_min": 3, "ngram_max": 3, "min_df": 1, "max_features": 300},
}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Entrena un pipeline mínimo y lo exporta como lo haría export_best."""
    texts = [f"the quick brown fox essay number {i} runs happily today" for i in range(20)]
    texts += [f"as an ai language model i generate structured essay {i}" for i in range(20)]
    labels = [0] * 20 + [1] * 20
    pipe = Pipeline(
        [
            ("features", build_features(CFG_FEATURES, kind="full")),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipe.fit(pd.Series(texts), labels)

    out = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, out / "model.joblib")
    (out / "model_meta.json").write_text(json.dumps({"model_name": "logreg-test"}))
    return out


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "model.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    from src.serving.app import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "logreg-test"


def test_predict_returns_valid_probability(client):
    r = client.post("/predict", json={"text": "students should be allowed to use phones"})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["proba_ai"] <= 1.0
    assert body["label"] in (0, 1)
    assert body["latency_ms"] > 0


def test_predict_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_predict_batch(client):
    texts = ["a human wrote this short essay", "as an ai language model i generate text"]
    r = client.post("/predict/batch", json={"texts": texts})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 2
    assert all(0.0 <= p["proba_ai"] <= 1.0 for p in body["predictions"])


def test_predictions_are_logged_without_raw_text(client, tmp_path):
    client.post("/predict", json={"text": "some very private essay content here"})
    log_file = tmp_path / "predictions.jsonl"
    lines = [json.loads(x) for x in log_file.read_text().splitlines()]
    assert len(lines) >= 1
    record = lines[-1]
    assert {"ts", "text_sha256", "n_words", "proba_ai", "label", "latency_ms"} <= record.keys()
    assert "private" not in json.dumps(record), "el texto crudo no debe loggearse"
