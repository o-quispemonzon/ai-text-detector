"""Carga de configuración YAML relativa a la raíz del proyecto."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict:
    """Carga un YAML de configuración.

    Rutas relativas se resuelven contra la raíz del proyecto, de modo que
    los scripts funcionen sin importar el cwd desde el que se ejecuten.
    """
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    with p.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path) -> Path:
    """Resuelve una ruta relativa contra la raíz del proyecto."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
