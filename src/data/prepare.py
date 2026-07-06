"""Limpieza y generación de splits reproducibles.

Uso:
    python -m src.data.prepare

Produce en data/processed/:
    train.parquet      entrenamiento
    val.parquet        validación (tuning, early stopping)
    test_iid.parquet   test aleatorio estratificado (distribución conocida)
    test_ood.parquet   textos AI de familias NUNCA vistas en train/val
    split_stats.json   composición de cada split (auditoría y referencia de drift)

Diseño (ver notebooks/01_eda.ipynb): la métrica OOD se calcula combinando
test_ood (AI de generadores no vistos) con los humanos de test_iid.
"""

from __future__ import annotations

import hashlib
import json
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import load_config, resolve_path
from src.utils.log import get_logger

logger = get_logger(__name__)

KEEP_COLS = ["text", "label", "prompt_name", "source", "model", "text_hash"]


def _hash_text(s: str) -> str:
    return hashlib.md5(s.strip().encode("utf-8")).hexdigest()


def clean(df: pd.DataFrame, min_words: int, drop_duplicates: bool = True) -> pd.DataFrame:
    """Limpieza mínima y auditada: nulos, duplicados exactos, textos degenerados."""
    n0 = len(df)
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].str.strip()
    df["text_hash"] = df["text"].map(_hash_text)

    if drop_duplicates:
        df = df.drop_duplicates(subset="text_hash", keep="first")

    n_words = df["text"].str.split().str.len()
    df = df[n_words >= min_words]

    logger.info("Limpieza: %d -> %d filas (-%d)", n0, len(df), n0 - len(df))
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, cfg_split: dict) -> dict[str, pd.DataFrame]:
    """Split OOD por familia de generador + train/val/test_iid estratificado.

    Estratifica por (label, model) para que la mezcla de generadores sea
    estable entre train, val y test_iid.
    """
    seed = cfg_split["seed"]
    ood_families = set(cfg_split["ood_model_families"])

    is_ood = df["model"].isin(ood_families)
    test_ood = df[is_ood]
    rest = df[~is_ood]

    if not (test_ood["label"] == 1).all():
        raise ValueError("test_ood debe contener solo textos AI (label=1)")

    strata = rest["label"].astype(str) + "_" + rest["model"].astype(str)
    train_val, test_iid = train_test_split(
        rest, test_size=cfg_split["test_size"], stratify=strata, random_state=seed
    )
    val_frac = cfg_split["val_size"] / (1 - cfg_split["test_size"])
    strata_tv = train_val["label"].astype(str) + "_" + train_val["model"].astype(str)
    train, val = train_test_split(
        train_val, test_size=val_frac, stratify=strata_tv, random_state=seed
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test_iid": test_iid.reset_index(drop=True),
        "test_ood": test_ood.reset_index(drop=True),
    }


def split_stats(splits: dict[str, pd.DataFrame]) -> dict:
    return {
        name: {
            "n": len(df),
            "pct_ai": round(float(df["label"].mean()), 4),
            "models": df["model"].value_counts().to_dict(),
        }
        for name, df in splits.items()
    }


def main() -> int:
    cfg = load_config("configs/data.yaml")
    raw_file = resolve_path(cfg["paths"]["raw_dir"]).parent / cfg["canonical"]["file"]
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Leyendo %s", raw_file)
    df = pd.read_csv(raw_file)
    df = clean(df, cfg["cleaning"]["min_words"], cfg["cleaning"]["drop_exact_duplicates"])
    splits = make_splits(df, cfg["split"])

    stats = split_stats(splits)
    for name, split_df in splits.items():
        path = out_dir / f"{name}.parquet"
        split_df[KEEP_COLS].to_parquet(path, index=False)
        logger.info(
            "%s: %d filas (%.0f%% AI) -> %s",
            name,
            stats[name]["n"],
            100 * stats[name]["pct_ai"],
            path,
        )

    (out_dir / "split_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
