"""Predice TB con el especialista de parches de lesión.

Uso:
    python scripts/predict_lesion.py --image ruta/a/radiografia.png
    python scripts/predict_lesion.py --image ruta.png --top-k 5 --out lesion.json

El modelo recorre el pulmón con parches de 96×96, los clasifica como lesión o
pulmón normal y reporta la probabilidad máxima y los parches más sospechosos.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lesion_patches import background_patches, patch_features, sliding_patches  # noqa: E402
from src.preprocessing import segment_image  # noqa: E402

MODELS = ROOT / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = json.loads(
        (MODELS / "lesion_patch_manifest.json").read_text(encoding="utf-8")
    )
    modelo = joblib.load(MODELS / "lesion_patch.joblib")
    bg_model = None
    bg_path = MODELS / "background_patch.joblib"
    if bg_path.exists():
        bg_model = joblib.load(bg_path)
    feat_cols = list(manifest["feature_columns"])
    size = int(manifest["patch_size"])
    stride = int(manifest["patch_stride"])

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {args.image}")
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    mask, quality = segment_image(args.image)
    if mask is None:
        raise ValueError(f"No se pudo segmentar: {args.image}")

    coords, filas = [], []
    for x0, y0, patch in sliding_patches(img, mask, size=size, stride=stride):
        f = patch_features(patch)
        filas.append([f[c] for c in feat_cols])
        coords.append((x0, y0))

    if not filas:
        raise ValueError("No se generaron parches pulmonares válidos.")

    X = np.asarray(filas, dtype=np.float32)
    proba = modelo.predict_proba(X)[:, 1]
    order = np.argsort(proba)[::-1][: args.top_k]
    top = [
        {
            "x": int(coords[i][0]),
            "y": int(coords[i][1]),
            "size": size,
            "prob_lesion": float(proba[i]),
        }
        for i in order
    ]
    score_fondo = None
    if bg_model is not None:
        bg_scores = []
        for patch in background_patches(img):
            f = patch_features(patch)
            bg_scores.append(
                bg_model.predict_proba(
                    np.asarray([[f[c] for c in feat_cols]], dtype=np.float32)
                )[0, 1]
            )
        score_fondo = float(np.max(bg_scores)) if bg_scores else None

    resultado = {
        "imagen": args.image,
        "prob_tb": float(proba.max()),
        "score_fondo": score_fondo,
        "prediccion": "tb" if proba.max() >= 0.5 else "no_tb",
        "n_parches": int(len(proba)),
        "top_parches": top,
        "mask_area_fraction": float(quality.area_fraction),
        "mask_fallback": bool(quality.fallback),
    }
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
