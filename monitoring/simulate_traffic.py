"""Simula tráfico contra la API para poblar el log de predicciones.

Uso:
    python monitoring/simulate_traffic.py --n 300              # tráfico normal
    python monitoring/simulate_traffic.py --n 300 --drift      # tráfico con drift

--drift trunca los textos a su primer 30% de palabras: simula un cambio en
producción (textos mucho más cortos) que el reporte de Evidently debe detectar.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.log import get_logger  # noqa: E402

logger = get_logger("simulate_traffic")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--drift", action="store_true", help="simula textos truncados")
    parser.add_argument("--test", default=REPO_ROOT / "data/processed/test_iid.parquet", type=Path)
    args = parser.parse_args()

    texts = pd.read_parquet(args.test)["text"].sample(n=args.n, random_state=7)
    if args.drift:
        texts = texts.map(lambda t: " ".join(t.split()[: max(len(t.split()) * 3 // 10, 10)]))
        logger.info("Modo drift: textos truncados al 30%% de su longitud")

    # trust_env=False: API local, no heredar proxies del entorno
    with httpx.Client(base_url=args.api, timeout=30, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        logger.info("API sana: %s", health.json()["model_name"])

        ok = 0
        for i in range(0, len(texts), 32):
            batch = texts.iloc[i : i + 32].tolist()
            r = client.post("/predict/batch", json={"texts": batch})
            r.raise_for_status()
            ok += len(batch)
        logger.info("Enviadas %d predicciones a %s", ok, args.api)
    return 0


if __name__ == "__main__":
    sys.exit(main())
