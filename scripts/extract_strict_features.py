"""Extrae textura estricta (sin pared torácica) y armonizada.

Salida:
    data/features_strict.csv
    data/features_strict_manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import extract_strict_features  # noqa: E402


def _worker(args):
    path, label, class_name = args
    try:
        return extract_strict_features(path, label, class_name)
    except Exception as exc:  # noqa: BLE001
        return {"__error__": f"{path}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(ROOT / "data" / "df_model.csv")
    if args.sample:
        df = pd.concat([
            g.sample(min(args.sample, len(g)), random_state=42)
            for _, g in df.groupby("class_name")
        ]).reset_index(drop=True)

    args_list = list(df[["image_path", "label", "class_name"]].itertuples(index=False, name=None))
    print(f"Imágenes: {len(args_list)} | workers: {args.workers}")

    rows, errors = [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res is None:
                errors.append(futures[fut][0])
            elif "__error__" in res:
                errors.append(res["__error__"])
            else:
                rows.append(res)
            if i % 100 == 0 or i == len(futures):
                elapsed = time.time() - t0
                print(f"{i}/{len(futures)} | {i/elapsed:.2f} img/s | ETA {(len(futures)-i)/(i/elapsed)/60:.1f} min")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "features_strict.csv", index=False)
    manifest = {
        "n_images": int(len(out)),
        "n_features": int(out.shape[1] - 3),
        "errors": len(errors),
        "seconds": round(time.time() - t0, 2),
        "modes": ["strict", "strict_harmonized"],
    }
    (ROOT / "data" / "features_strict_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
