"""Configuración del tracking de MLflow.

SQLite no es fiable sobre NTFS/9p (repo en /mnt/c bajo WSL2): el locking del
filesystem produce `disk I/O error`. Por eso la BD vive en el filesystem
nativo de Linux (~/.ai-text-detector/), mientras que los artefactos (modelos,
reportes) sí se guardan como archivos planos junto al repo, donde no hay
problema de locks.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_tracking_uri() -> str:
    """URI de tracking: env MLFLOW_TRACKING_URI > sqlite en el home."""
    if uri := os.environ.get("MLFLOW_TRACKING_URI"):
        return uri
    db = Path.home() / ".ai-text-detector" / "mlflow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"
