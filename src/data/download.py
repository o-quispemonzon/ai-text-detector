"""Descarga idempotente de los datasets DAIGT desde Kaggle.

Uso:
    python -m src.data.download [--force]

Requiere credenciales de Kaggle en ~/.kaggle/kaggle.json o en las
variables de entorno KAGGLE_USERNAME / KAGGLE_KEY.

Al terminar escribe data/manifest.json con SHA-256, tamaño y procedencia
de cada archivo: es nuestro mecanismo ligero de versionado de datos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.utils.config import load_config, resolve_path
from src.utils.log import get_logger

logger = get_logger(__name__)

_CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    """SHA-256 de un archivo leído por chunks (apto para archivos grandes)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def kaggle_credentials_available() -> bool:
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def download_dataset(slug: str, dest: Path) -> None:
    """Descarga y descomprime un dataset de Kaggle en `dest`."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    logger.info("Descargando %s -> %s", slug, dest)
    subprocess.run(cmd, check=True)


def build_manifest_entries(name: str, slug: str, dest: Path) -> list[dict]:
    """Una entrada por archivo: ruta, tamaño y hash, para reproducibilidad."""
    entries = []
    for file in sorted(p for p in dest.rglob("*") if p.is_file()):
        entries.append(
            {
                "dataset": name,
                "kaggle_slug": slug,
                "file": file.relative_to(dest.parent.parent).as_posix(),
                "size_bytes": file.stat().st_size,
                "sha256": sha256_of(file),
            }
        )
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descarga los datasets DAIGT desde Kaggle")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--force", action="store_true", help="re-descarga aunque ya exista")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    manifest_path = resolve_path(cfg["paths"]["manifest"])

    if not kaggle_credentials_available():
        logger.error(
            "No hay credenciales de Kaggle. Crea un token en "
            "kaggle.com -> Settings -> API -> 'Create New Token' y guárdalo en "
            "~/.kaggle/kaggle.json (chmod 600)."
        )
        return 1

    all_entries: list[dict] = []
    for ds in cfg["datasets"]:
        dest = raw_dir / ds["name"]
        if dest.exists() and any(dest.iterdir()) and not args.force:
            logger.info("%s ya existe, se omite (--force para re-descargar)", ds["name"])
        else:
            download_dataset(ds["slug"], dest)
        all_entries.extend(build_manifest_entries(ds["name"], ds["slug"], dest))

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": all_entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info("Manifest escrito en %s (%d archivos)", manifest_path, len(all_entries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
