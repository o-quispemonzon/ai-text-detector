"""Reporte de data drift: producción simulada vs. referencia.

Uso:
    python monitoring/drift_report.py [--fail-on-drift]

Referencia:  muestra de test_iid.parquet, con scores del modelo servido.
Actual:      monitoring/logs/predictions.jsonl (lo que registró la API).
Compara n_words, n_chars y proba_ai con DataDriftPreset de Evidently y
genera monitoring/reports/drift_report.html.

Con --fail-on-drift retorna código 1 si drifteó ≥50% de las columnas
(útil para automatizar una alerta).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.log import get_logger  # noqa: E402

logger = get_logger("drift_report")

DRIFT_COLUMNS = ["n_words", "n_chars", "proba_ai"]


def build_reference(model_path: Path, test_path: Path, n: int = 2000) -> pd.DataFrame:
    """Score del modelo sobre una muestra del test IID: así se veía 'producción sana'."""
    model = joblib.load(model_path)
    df = pd.read_parquet(test_path).sample(n=n, random_state=42)
    return pd.DataFrame(
        {
            "n_words": df["text"].str.split().str.len(),
            "n_chars": df["text"].str.len(),
            "proba_ai": model.predict_proba(df["text"])[:, 1],
        }
    )


def load_current(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise SystemExit(
            f"No hay log de predicciones en {log_path}. "
            "Levanta la API y genera tráfico (monitoring/simulate_traffic.py)."
        )
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(records)[DRIFT_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=REPO_ROOT / "models/model.joblib", type=Path)
    parser.add_argument("--test", default=REPO_ROOT / "data/processed/test_iid.parquet", type=Path)
    parser.add_argument("--log", default=REPO_ROOT / "monitoring/logs/predictions.jsonl", type=Path)
    parser.add_argument("--out", default=REPO_ROOT / "monitoring/reports", type=Path)
    parser.add_argument("--fail-on-drift", action="store_true")
    args = parser.parse_args()

    # Import aquí: evidently es pesado y solo se necesita para este script
    from evidently import Report
    from evidently.presets import DataDriftPreset

    reference = build_reference(args.model, args.test)
    current = load_current(args.log)
    logger.info("Referencia: %d filas | Actual: %d filas", len(reference), len(current))

    result = Report([DataDriftPreset()]).run(reference_data=reference, current_data=current)

    args.out.mkdir(parents=True, exist_ok=True)
    html_path = args.out / "drift_report.html"
    result.save_html(str(html_path))

    summary = json.loads(result.json())
    drifted = next(
        m["value"] for m in summary["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")
    )
    logger.info(
        "Columnas con drift: %d/%d (%.0f%%) | Reporte: %s",
        drifted["count"],
        len(DRIFT_COLUMNS),
        100 * drifted["share"],
        html_path,
    )

    if args.fail_on_drift and drifted["share"] >= 0.5:
        logger.error("DRIFT DETECTADO: revisa el reporte y considera reentrenar.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
