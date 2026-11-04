"""Reconstruye la matriz de características con la segmentación validada.

Uso:
    python scripts/rebuild_features.py --workers 6
    python scripts/rebuild_features.py --sample 50 --workers 2

Salida:
    data/features_tbx11k_v2.csv
    data/features_v2_errors.json
    data/features_v2_manifest.json
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

from src.features import extract_features  # noqa: E402


def _worker(args: tuple[str, int, str]) -> dict | None:
    path, label, class_name = args
    try:
        return extract_features(path, label, class_name)
    except Exception as exc:  # noqa: BLE001
        return {"__error__": f"{path}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--sample", type=int, default=0, help="Imágenes por clase (0 = todas).")
    parser.add_argument("--out", default="data/features_tbx11k_v2.csv")
    args = parser.parse_args()

    meta_path = ROOT / "data" / "full_meta.csv"
    if not meta_path.exists():
        meta_path = ROOT / "data" / "df_base.csv"
    df = pd.read_csv(meta_path)
    if args.sample:
        df = pd.concat([
            g.sample(min(args.sample, len(g)), random_state=42)
            for _, g in df.groupby("class_name")
        ]).reset_index(drop=True)

    print(f"Imágenes: {len(df)} | workers: {args.workers}")
    args_list = list(df[["image_path", "label", "class_name"]].itertuples(index=False, name=None))

    rows: list[dict] = []
    errors: list[str] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, a): a for a in args_list}
        for i, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            if result is None:
                errors.append(futures[fut][0])
            elif "__error__" in result:
                errors.append(result["__error__"])
            else:
                rows.append(result)
            if i % 100 == 0 or i == len(futures):
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-9)
                eta = (len(futures) - i) / max(rate, 1e-9)
                print(f"{i}/{len(futures)} | {rate:.2f} img/s | ETA {eta/60:.1f} min")

    out = pd.DataFrame(rows)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    err_path = ROOT / "data" / "features_v2_errors.json"
    err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "n_images": int(len(out)),
        "n_features": int(out.shape[1] - 3) if len(out) else 0,
        "classes": out["class_name"].value_counts().to_dict() if len(out) else {},
        "errors": len(errors),
        "workers": args.workers,
        "seconds": round(time.time() - t0, 2),
        "output": str(out_path),
        "segmentation": "src/lung_segmentation.py (campo pulmonar validado, margen 20 px)",
    }
    (ROOT / "data" / "features_v2_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
