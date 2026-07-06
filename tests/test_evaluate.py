"""Tests de las métricas y la evaluación OOD."""

import numpy as np
import pandas as pd

from src.models.evaluate import build_ood_frame, compute_metrics


def test_compute_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, proba)
    assert m["roc_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["brier"] < 0.05


def test_compute_metrics_prefix():
    m = compute_metrics([0, 1], [0.2, 0.8], prefix="val_")
    assert set(m) == {"val_roc_auc", "val_avg_precision", "val_f1", "val_accuracy", "val_brier"}


def test_build_ood_frame_combines_ood_ai_with_iid_humans():
    test_iid = pd.DataFrame({"text": ["h1", "h2", "a1"], "label": [0, 0, 1]})
    test_ood = pd.DataFrame({"text": ["o1", "o2"], "label": [1, 1]})
    ood = build_ood_frame(test_iid, test_ood)
    assert len(ood) == 4  # 2 humanos IID + 2 AI OOD
    assert ood["label"].sum() == 2
    assert "a1" not in set(ood["text"])  # el AI de IID no entra en la eval OOD
