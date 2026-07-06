"""Métricas de evaluación, incluida la evaluación OOD.

La evaluación OOD combina los textos AI de generadores nunca vistos
(test_ood) con los textos humanos de test_iid: mide si el detector
distingue humano vs AI cuando el generador no estuvo en el train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, prefix: str = "") -> dict[str, float]:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= 0.5).astype(int)
    return {
        f"{prefix}roc_auc": float(roc_auc_score(y_true, proba)),
        f"{prefix}avg_precision": float(average_precision_score(y_true, proba)),
        f"{prefix}f1": float(f1_score(y_true, pred)),
        f"{prefix}accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}brier": float(brier_score_loss(y_true, proba)),
    }


def build_ood_frame(test_iid: pd.DataFrame, test_ood: pd.DataFrame) -> pd.DataFrame:
    """AI de familias no vistas + humanos del test IID."""
    humans = test_iid[test_iid["label"] == 0]
    return pd.concat([test_ood, humans], ignore_index=True)


def evaluate_all(predict_proba, splits: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Evalúa un modelo sobre val, test_iid y test_ood.

    predict_proba: callable que recibe una lista/Series de textos y devuelve
    la probabilidad de la clase AI (1).
    """
    metrics: dict[str, float] = {}
    for name in ["val", "test_iid"]:
        df = splits[name]
        metrics |= compute_metrics(df["label"], predict_proba(df["text"]), prefix=f"{name}_")

    ood = build_ood_frame(splits["test_iid"], splits["test_ood"])
    metrics |= compute_metrics(ood["label"], predict_proba(ood["text"]), prefix="test_ood_")
    return metrics
