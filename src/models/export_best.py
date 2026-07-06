"""Exporta el mejor modelo sklearn desde MLflow hacia models/.

Uso:
    python -m src.models.export_best                       # mejor por test_ood_roc_auc
    python -m src.models.export_best --metric test_iid_roc_auc
    python -m src.models.export_best --run-id <id>         # un run específico

Busca en el experimento `classic` (los artefactos del transformer son un
directorio HF y requieren otro loader; la decisión de servir el modelo
clásico está documentada en docs/decisions.md). Copia el .joblib a
models/model.joblib y escribe models/model_meta.json con métricas y
procedencia, de modo que la API sepa qué versión sirve.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import mlflow

from src.utils.config import resolve_path
from src.utils.log import get_logger
from src.utils.tracking import get_tracking_uri

logger = get_logger(__name__)


def find_best_run(client: mlflow.MlflowClient, experiment: str, metric: str):
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        raise SystemExit(f"No existe el experimento '{experiment}'. ¿Entrenaste ya?")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.subsample = 'full'",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit("No hay runs completos (tags.subsample='full') en el experimento.")
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="classic")
    parser.add_argument("--metric", default="test_ood_roc_auc")
    parser.add_argument("--run-id", default=None, help="exporta este run en vez del mejor")
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(get_tracking_uri())
    client = mlflow.MlflowClient()

    run = (
        client.get_run(args.run_id)
        if args.run_id
        else find_best_run(client, args.experiment, args.metric)
    )
    run_name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
    logger.info("Run elegido: %s (%s=%.4f)", run_name, args.metric, run.data.metrics[args.metric])

    local_dir = client.download_artifacts(run.info.run_id, "model")
    joblibs = list(Path(local_dir).glob("*.joblib"))
    if not joblibs:
        raise SystemExit(
            f"El run '{run_name}' no tiene artefacto .joblib (¿es un transformer?). "
            "Este export sirve modelos sklearn."
        )

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "model.joblib"
    shutil.copy2(joblibs[0], dest)

    meta = {
        "model_name": run_name,
        "run_id": run.info.run_id,
        "experiment": args.experiment,
        "selected_by": args.metric,
        "metrics": {k: round(v, 6) for k, v in run.data.metrics.items()},
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    logger.info("Modelo exportado a %s (run %s)", dest, run.info.run_id[:8])
    return 0


if __name__ == "__main__":
    sys.exit(main())
