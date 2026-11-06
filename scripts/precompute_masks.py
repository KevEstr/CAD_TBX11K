"""Calcula y guarda en caché las máscaras de segmentación de todo el dataset.

Uso:
    python scripts/precompute_masks.py
    python scripts/precompute_masks.py --manifest data/df_base.csv --limit 100

La primera ejecución descarga los pesos de PSPNet si no están en
``~/.torchxrayvision/models_data``. El cálculo completo en CPU tarda varias horas;
las máscaras se guardan en ``data/masks`` y luego los notebooks las reutilizan.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import mask_cache_exists, segment_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(ROOT / "data" / "df_model.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(
            f"No existe {manifest}. Ejecuta primero el notebook 01."
        )

    df = pd.read_csv(manifest)
    if args.limit:
        df = df.head(args.limit)

    pendientes = [
        p for p in df["image_path"]
        if args.force or not mask_cache_exists(str(p))
    ]
    print(f"Imágenes: {len(df)} | pendientes: {len(pendientes)}")
    if not pendientes:
        print("Todas las máscaras ya están en caché.")
        return

    t0 = time.time()
    errores = 0
    for i, path in enumerate(pendientes, start=1):
        mask, quality = segment_image(str(path))
        if mask is None:
            errores += 1
        if i % 25 == 0 or i == len(pendientes):
            transcurrido = time.time() - t0
            ritmo = i / max(transcurrido, 1e-9)
            eta = (len(pendientes) - i) / max(ritmo, 1e-9)
            print(
                f"{i}/{len(pendientes)} | {ritmo:.2f} img/s | "
                f"ETA {eta / 3600:.1f} h | errores {errores}"
            )

    print(f"Listo. Máscaras en {ROOT / 'data' / 'masks'}")


if __name__ == "__main__":
    main()
