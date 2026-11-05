"""Valida la máscara pulmonar contra máscaras manuales.

Uso:
    python scripts/validate_masks.py --masks-dir ruta/a/mascaras
    python scripts/validate_masks.py --masks-dir ruta/a/mascaras --limit 200

Las máscaras manuales deben tener el mismo nombre base que la imagen y estar
en PNG/TIF (binarias) o NPY. Se calculan Dice, IoU, precisión y recall de la
máscara del pipeline contra la referencia manual.

Salida:
    data/seg_metrics.csv
    data/seg_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lung_segmentation import SEG_SIZE, segment_lungs  # noqa: E402

DATA = ROOT / "data"


def load_reference_mask(path: Path) -> np.ndarray | None:
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = (mask > 0).astype(np.uint8)
    if mask.shape != SEG_SIZE:
        mask = cv2.resize(mask, SEG_SIZE, interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


def find_mask(masks_dir: Path, image_path: str) -> Path | None:
    stem = Path(image_path).stem
    for ext in (".png", ".tif", ".tiff", ".npy"):
        candidate = masks_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--manifest", default=str(DATA / "df_model.csv"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    masks_dir = Path(args.masks_dir)
    if not masks_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de máscaras: {masks_dir}")

    meta = pd.read_csv(args.manifest)
    if args.limit:
        meta = meta.head(args.limit)

    rows = []
    for _, row in meta.iterrows():
        mask_path = find_mask(masks_dir, row["image_path"])
        if mask_path is None:
            continue
        ref = load_reference_mask(mask_path)
        if ref is None or not ref.any():
            continue

        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img512 = cv2.resize(img, SEG_SIZE, interpolation=cv2.INTER_AREA)
        pred, quality = segment_lungs(img512)

        inter = int((pred & ref).sum())
        union = int((pred | ref).sum())
        pred_area = int(pred.sum())
        ref_area = int(ref.sum())
        precision = inter / pred_area if pred_area else 0.0
        recall = inter / ref_area if ref_area else 0.0
        dice = 2 * inter / (pred_area + ref_area) if (pred_area + ref_area) else 0.0
        iou = inter / union if union else 0.0

        rows.append({
            "image_name": Path(row["image_path"]).name,
            "class_name": row["class_name"],
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "pred_area": pred_area,
            "ref_area": ref_area,
            "fallback": int(quality.fallback),
            "area_fraction": quality.area_fraction,
            "lr_balance": quality.lr_balance,
            "contrast": quality.contrast,
        })

    if not rows:
        raise ValueError(
            "No se encontraron máscaras manuales con nombres coincidentes."
        )

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "seg_metrics.csv", index=False)
    summary = {
        "n_images": int(len(df)),
        "dice_mean": float(df["dice"].mean()),
        "dice_median": float(df["dice"].median()),
        "iou_mean": float(df["iou"].mean()),
        "precision_mean": float(df["precision"].mean()),
        "recall_mean": float(df["recall"].mean()),
        "fallback_rate": float(df["fallback"].mean()),
        "by_class": df.groupby("class_name")[["dice", "iou", "precision", "recall"]]
        .mean().round(4).to_dict(orient="index"),
    }
    (DATA / "seg_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
