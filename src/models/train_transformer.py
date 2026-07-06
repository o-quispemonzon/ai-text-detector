"""Fine-tuning de DeBERTa-v3-small con tracking en MLflow.

Uso:
    python -m src.models.train_transformer                    # entrenamiento completo
    python -m src.models.train_transformer --subsample 2000   # smoke test
    python -m src.models.train_transformer --epochs 1

Evalúa con los MISMOS splits y métricas que el enfoque clásico
(src/models/evaluate.py), para una comparación limpia entre enfoques.
El modelo entrenado se guarda como artefacto del run (formato HF).

Requiere el extra `transformer`: uv sync --all-extras
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.models.evaluate import evaluate_all
from src.models.train_classic import load_splits
from src.utils.config import load_config, resolve_path
from src.utils.log import get_logger
from src.utils.tracking import get_tracking_uri

logger = get_logger(__name__)


def tokenize_frame(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["text", "label"]].rename(columns={"label": "labels"}))
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def make_predict_proba(trainer: Trainer, tokenizer, max_length: int):
    """Devuelve un callable texto -> P(AI), compatible con evaluate_all."""

    def predict_proba(texts) -> np.ndarray:
        df = pd.DataFrame({"text": list(texts), "label": 0})
        ds = tokenize_frame(df, tokenizer, max_length).remove_columns(["labels"])
        logits = trainer.predict(ds).predictions
        return torch.softmax(torch.from_numpy(logits).float(), dim=-1)[:, 1].numpy()

    return predict_proba


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_transformer.yaml")
    parser.add_argument("--subsample", type=int, default=None, help="filas por split (smoke)")
    parser.add_argument("--epochs", type=float, default=None, help="override de epochs")
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="auto = bf16 si la GPU lo soporta; fp32 como fallback ante NaNs",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = load_config("configs/data.yaml")
    tr = cfg["training"]
    epochs = args.epochs or tr["epochs"]
    use_gpu = torch.cuda.is_available()
    if not use_gpu:
        logger.warning("Sin GPU: entrenará en CPU (lento). Precisión mixta desactivada.")
    # bf16 (Ampere/Ada+) no usa grad scaler y es más estable que fp16.
    if args.precision == "auto":
        mixed = bool(tr["fp16"]) and use_gpu
        use_bf16 = mixed and torch.cuda.is_bf16_supported()
        use_fp16 = mixed and not use_bf16
    else:
        use_bf16 = args.precision == "bf16" and use_gpu
        use_fp16 = args.precision == "fp16" and use_gpu
    logger.info("Precisión: bf16=%s fp16=%s", use_bf16, use_fp16)

    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.set_experiment(cfg["experiment"])

    splits = load_splits(
        resolve_path(data_cfg["paths"]["processed_dir"]), args.subsample, cfg["seed"]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)

    ds_train = tokenize_frame(splits["train"], tokenizer, cfg["max_length"])
    ds_val = tokenize_frame(splits["val"], tokenizer, cfg["max_length"])

    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run(run_name=cfg["model_name"]):
        mlflow.set_tags({"model": cfg["model_name"], "subsample": str(args.subsample or "full")})
        mlflow.log_params(
            {
                "max_length": cfg["max_length"],
                "epochs": epochs,
                "batch_size": tr["batch_size"],
                "grad_accum": tr["grad_accum"],
                "learning_rate": tr["learning_rate"],
                "n_train": len(splits["train"]),
            }
        )

        training_args = TrainingArguments(
            output_dir=str(Path(tmp) / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=tr["batch_size"],
            per_device_eval_batch_size=tr["batch_size"] * 2,
            gradient_accumulation_steps=tr["grad_accum"],
            learning_rate=float(tr["learning_rate"]),
            warmup_ratio=tr["warmup_ratio"],
            weight_decay=tr["weight_decay"],
            fp16=use_fp16,
            bf16=use_bf16,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=50,
            report_to=[],
            seed=cfg["seed"],
            dataloader_num_workers=2,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds_train,
            eval_dataset=ds_val,
            data_collator=DataCollatorWithPadding(tokenizer),
        )

        logger.info("Entrenando %s (%s epochs, GPU=%s)...", cfg["model_name"], epochs, use_gpu)
        trainer.train()

        metrics = evaluate_all(make_predict_proba(trainer, tokenizer, cfg["max_length"]), splits)
        mlflow.log_metrics(metrics)

        model_dir = Path(tmp) / "model"
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        mlflow.log_artifacts(str(model_dir), artifact_path="model")

        logger.info(
            "%s | AUC val=%.4f iid=%.4f ood=%.4f",
            cfg["model_name"],
            metrics["val_roc_auc"],
            metrics["test_iid_roc_auc"],
            metrics["test_ood_roc_auc"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
