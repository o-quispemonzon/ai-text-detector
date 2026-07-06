"""Entrenamiento del enfoque clásico con tracking en MLflow.

Uso:
    python -m src.models.train_classic                      # los 3 modelos
    python -m src.models.train_classic --models logreg      # subconjunto
    python -m src.models.train_classic --subsample 3000     # smoke test / CI

Cada modelo es un Pipeline sklearn completo (features + clasificador), de modo
que el artefacto guardado se usa tal cual en inferencia: entra texto crudo,
sale probabilidad.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features.vectorizers import build_features
from src.models.evaluate import evaluate_all
from src.utils.config import load_config, resolve_path
from src.utils.log import get_logger
from src.utils.tracking import get_tracking_uri

logger = get_logger(__name__)

CLASSIFIERS = {
    "logreg": LogisticRegression,
    "lightgbm": LGBMClassifier,
    "random_forest": RandomForestClassifier,
}


def load_splits(processed_dir: Path, subsample: int | None = None, seed: int = 42) -> dict:
    splits = {
        name: pd.read_parquet(processed_dir / f"{name}.parquet")
        for name in ["train", "val", "test_iid", "test_ood"]
    }
    if subsample:
        splits = {
            name: df.sample(n=min(subsample, len(df)), random_state=seed).reset_index(drop=True)
            for name, df in splits.items()
        }
    return splits


def build_pipeline(model_name: str, model_cfg: dict, cfg_features: dict, seed: int) -> Pipeline:
    clf_cls = CLASSIFIERS[model_name]
    params = dict(model_cfg["params"])
    if "random_state" in clf_cls().get_params():
        params.setdefault("random_state", seed)
    return Pipeline(
        [
            ("features", build_features(cfg_features, kind=model_cfg["features"])),
            ("clf", clf_cls(**params)),
        ]
    )


def train_one(name: str, cfg: dict, splits: dict, tags: dict) -> dict[str, float]:
    pipe = build_pipeline(name, cfg["models"][name], cfg["features"], cfg["seed"])

    with mlflow.start_run(run_name=name):
        mlflow.set_tags(tags | {"model": name, "features": cfg["models"][name]["features"]})
        mlflow.log_params(cfg["models"][name]["params"])
        mlflow.log_param("n_train", len(splits["train"]))

        logger.info("Entrenando %s (features=%s)...", name, cfg["models"][name]["features"])
        pipe.fit(splits["train"]["text"], splits["train"]["label"])

        metrics = evaluate_all(lambda texts: pipe.predict_proba(texts)[:, 1], splits)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / f"{name}.joblib"
            joblib.dump(pipe, model_path, compress=3)
            mlflow.log_artifact(str(model_path), artifact_path="model")

        logger.info(
            "%s | AUC val=%.4f iid=%.4f ood=%.4f",
            name,
            metrics["val_roc_auc"],
            metrics["test_iid_roc_auc"],
            metrics["test_ood_roc_auc"],
        )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_classic.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="subconjunto de modelos")
    parser.add_argument("--subsample", type=int, default=None, help="filas por split (smoke)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(cfg["experiment"])

    splits = load_splits(
        resolve_path(data_cfg["paths"]["processed_dir"]), args.subsample, cfg["seed"]
    )
    model_names = args.models or list(cfg["models"])
    tags = {"subsample": str(args.subsample or "full")}

    results = {name: train_one(name, cfg, splits, tags) for name in model_names}

    summary = pd.DataFrame(results).T[
        ["val_roc_auc", "test_iid_roc_auc", "test_ood_roc_auc", "test_iid_f1", "test_ood_f1"]
    ]
    logger.info("Resumen:\n%s", summary.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
