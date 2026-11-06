"""Genera imágenes de control visual de la segmentación.

Uso:
    python scripts/preview_segmentation.py --image ruta.png --out-dir reports/seg
    python scripts/preview_segmentation.py --image ruta.png --out-dir reports/seg --strict

Salida:
    <stem>_original.png
    <stem>_mask.png
    <stem>_overlay.png
    <stem>_thorax.png
    <stem>_lung.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lung_segmentation import SEG_SIZE  # noqa: E402
from src.preprocessing import preprocess_bundle, segment_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-dir", default="reports/seg")
    parser.add_argument("--strict", action="store_true", help="Incluir vista estricta.")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")
    img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
    mask, quality = segment_image(str(image_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    cv2.imwrite(str(out_dir / f"{stem}_original.png"), img512)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), (mask.astype(np.uint8) * 255))

    overlay = cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    bundle = preprocess_bundle(str(image_path), include_strict=args.strict)
    if bundle is not None:
        cv2.imwrite(str(out_dir / f"{stem}_thorax.png"), bundle["thorax"][0])
        cv2.imwrite(str(out_dir / f"{stem}_lung.png"), bundle["lung"][0])

    print({
        "image": str(image_path),
        "area_fraction": round(quality.area_fraction, 4),
        "lr_balance": round(quality.lr_balance, 4),
        "contrast": round(quality.contrast, 4),
        "fallback": quality.fallback,
        "reason": quality.reason,
        "out_dir": str(out_dir),
    })


if __name__ == "__main__":
    main()
