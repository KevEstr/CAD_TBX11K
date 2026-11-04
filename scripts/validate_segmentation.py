"""Valida la segmentación pulmonar contra las cajas de lesión de TBX11K.

Genera:
    data/seg_validation_v2.csv          (una fila por imagen TB con cajas)
    data/seg_quality_v2.csv             (muestra estratificada por clase)
    data/seg_summary_v2.json
    reports/03_segmentacion_ejemplos.png
    reports/03_segmentacion_lesiones.png
    reports/03_segmentacion_comparacion.png
    reports/03_segmentacion_distribucion.png

Uso:
    python scripts/validate_segmentation.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lung_segmentation import segment_lungs  # noqa: E402

COCO = ROOT / "data" / "TBX11K" / "annotations" / "json" / "TBX11K_trainval.json"
META = ROOT / "data" / "full_meta.csv"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

TB_CATEGORIES = {1: "ActiveTuberculosis", 2: "ObsoletePulmonaryTuberculosis",
                 3: "PulmonaryTuberculosis"}


def old_otsu_mask(img256: np.ndarray) -> np.ndarray:
    """Segmentación heurística anterior (Otsu invertido), para comparación."""
    blur = cv2.GaussianBlur(img256, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(th, 8)
    h, w = img256.shape
    keep = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 500:
            continue
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            continue
        cx, cy = cent[i]
        if cy > 200:
            continue
        keep.append((area, i, cx, cy))
    left = [c for c in keep if c[2] < w / 2]
    right = [c for c in keep if c[2] >= w / 2]
    chosen = []
    if left:
        chosen.append(max(left, key=lambda x: x[0])[1])
    if right:
        chosen.append(max(right, key=lambda x: x[0])[1])
    if not chosen:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1, -1)
        return mask.astype(bool)
    mask = np.isin(labels, chosen)
    mask = binary_fill_holes(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def load_coco_boxes() -> tuple[dict, dict]:
    coco = json.loads(COCO.read_text(encoding="utf-8"))
    id_to_name = {im["id"]: Path(im["file_name"]).name for im in coco["images"]}
    boxes: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    for ann in coco["annotations"]:
        boxes[id_to_name[ann["image_id"]]].append((ann["category_id"], ann["bbox"]))
    return coco, boxes


def box_coverage(mask: np.ndarray, boxes: list[tuple[int, list[float]]]) -> tuple[float, float, dict]:
    covs, cents = [], []
    per_cat: dict[int, list[float]] = defaultdict(list)
    h, w = mask.shape
    for cid, (x, y, bw, bh) in boxes:
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(w, int(round(x + bw)))
        y1 = min(h, int(round(y + bh)))
        if x1 <= x0 or y1 <= y0:
            continue
        cov = float(mask[y0:y1, x0:x1].mean())
        covs.append(cov)
        per_cat[cid].append(cov)
        cents.append(bool(mask[int((y0 + y1) / 2), int((x0 + x1) / 2)]))
    mean_cov = float(np.mean(covs)) if covs else float("nan")
    mean_cent = float(np.mean(cents)) if cents else float("nan")
    cat_mean = {cid: float(np.mean(v)) for cid, v in per_cat.items()}
    return mean_cov, mean_cent, cat_mean


def main() -> None:
    coco, boxes = load_coco_boxes()
    print("Categorías COCO:", coco["categories"])
    print("Anotaciones por categoría:",
          pd.Series([a["category_id"] for a in coco["annotations"]]).value_counts().to_dict())
    print("Imágenes con cajas:", len(boxes), "| cajas:", sum(len(v) for v in boxes.values()))

    meta = pd.read_csv(META)
    tb = meta[meta["class_name"] == "tb"].copy()

    rows = []
    for _, r in tb.iterrows():
        img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        name = Path(r["image_path"]).name
        mask_new, q = segment_lungs(img512)
        cov_new, cent_new, cat_new = box_coverage(mask_new, boxes.get(name, []))

        img256 = cv2.resize(img512, (256, 256), interpolation=cv2.INTER_AREA)
        mask_old = old_otsu_mask(img256)
        mask_old = cv2.resize(mask_old.astype(np.uint8), (512, 512),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        cov_old, cent_old, _ = box_coverage(mask_old, boxes.get(name, []))

        rows.append({
            "image_name": name,
            "n_boxes": len(boxes.get(name, [])),
            "cov_new": cov_new,
            "cov_old": cov_old,
            "cent_new": cent_new,
            "cent_old": cent_old,
            "area_new": q.area_fraction,
            "balance_new": q.lr_balance,
            "contrast_new": q.contrast,
            "fallback_new": int(q.fallback),
            "cov_active": cat_new.get(1, np.nan),
            "cov_obsolete": cat_new.get(2, np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "seg_validation_v2.csv", index=False)

    # Calidad en una muestra estratificada (300 por clase).
    qrows = []
    for cls, g in meta.groupby("class_name"):
        for _, r in g.sample(min(300, len(g)), random_state=42).iterrows():
            img = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img512 = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
            _, q = segment_lungs(img512)
            qrows.append({
                "class_name": cls,
                "area": q.area_fraction,
                "balance": q.lr_balance,
                "contrast": q.contrast,
                "fallback": int(q.fallback),
            })
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(ROOT / "data" / "seg_quality_v2.csv", index=False)

    summary = {
        "n_tb": int(len(df)),
        "n_boxes": int(df["n_boxes"].sum()),
        "categories": {str(k): int(v) for k, v in
                       pd.Series([a["category_id"] for a in coco["annotations"]]).value_counts().items()},
        "new": {
            "coverage_mean": float(df["cov_new"].mean()),
            "coverage_ge_0.8": float((df["cov_new"] >= 0.8).mean()),
            "coverage_ge_0.5": float((df["cov_new"] >= 0.5).mean()),
            "centroid_inside": float(df["cent_new"].mean()),
            "fallback_rate": float(df["fallback_new"].mean()),
            "area_mean": float(df["area_new"].mean()),
            "balance_mean": float(df["balance_new"].mean()),
        },
        "old": {
            "coverage_mean": float(df["cov_old"].mean()),
            "coverage_ge_0.8": float((df["cov_old"] >= 0.8).mean()),
            "coverage_ge_0.5": float((df["cov_old"] >= 0.5).mean()),
            "centroid_inside": float(df["cent_old"].mean()),
        },
        "quality_by_class": qdf.groupby("class_name").agg(
            area=("area", "mean"), balance=("balance", "mean"),
            contrast=("contrast", "mean"), fallback=("fallback", "mean"),
        ).round(3).to_dict(orient="index"),
    }
    (ROOT / "data" / "seg_summary_v2.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------
    # Figuras
    # ------------------------------------------------------------------
    plt.rcParams.update({"figure.dpi": 120, "font.size": 8})

    # Ejemplos por clase.
    fig, axes = plt.subplots(3, 4, figsize=(11, 8.5))
    for row_i, cls in enumerate(["health", "sick", "tb"]):
        sample = meta[meta["class_name"] == cls].sample(1, random_state=7).iloc[0]
        img = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        img512 = cv2.resize(img, (512, 512))
        mask, q = segment_lungs(img512)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        overlay = cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
        x0, y0, x1, y1 = cv2.boundingRect(mask.astype(np.uint8))
        x0, y0 = max(0, x0 - 20), max(0, y0 - 20)
        x1, y1 = min(512, x1 + 20), min(512, y1 + 20)
        crop = img512[y0:y1, x0:x1]
        axes[row_i, 0].imshow(img512, cmap="gray")
        axes[row_i, 0].set_title(f"{cls} | original")
        axes[row_i, 1].imshow(mask, cmap="gray")
        axes[row_i, 1].set_title(f"máscara | área={q.area_fraction:.2f}")
        axes[row_i, 2].imshow(overlay, cmap="gray")
        axes[row_i, 2].set_title(f"contorno | balance={q.lr_balance:.2f}")
        axes[row_i, 3].imshow(crop, cmap="gray")
        axes[row_i, 3].set_title("recorte torácico")
        for ax in axes[row_i]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(REPORTS / "03_segmentacion_ejemplos.png", bbox_inches="tight")
    plt.close()

    # Lesiones: peores y mejores casos.
    tb_meta = meta[meta["class_name"] == "tb"].set_index("image_path")
    name_to_path = {Path(p).name: p for p in tb_meta.index}
    chosen = pd.concat([df.nsmallest(3, "cov_new"), df.nlargest(3, "cov_new")])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (_, r) in zip(axes.ravel(), chosen.iterrows()):
        path = name_to_path[r["image_name"]]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img512 = cv2.resize(img, (512, 512))
        mask, _ = segment_lungs(img512)
        vis = cv2.cvtColor(img512, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 1)
        for cid, (x, y, bw, bh) in boxes.get(r["image_name"], []):
            cv2.rectangle(vis, (int(x), int(y)), (int(x + bw), int(y + bh)),
                          (255, 255, 0), 1)
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{r['image_name']} | cov={r['cov_new']:.2f}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(REPORTS / "03_segmentacion_lesiones.png", bbox_inches="tight")
    plt.close()

    # Comparación nueva vs anterior.
    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = ["coverage_mean", "coverage_ge_0.8", "coverage_ge_0.5", "centroid_inside"]
    labels = ["Cobertura media", "Cobertura ≥ 0.8", "Cobertura ≥ 0.5", "Centroide dentro"]
    x = np.arange(len(metrics))
    ax.bar(x - 0.2, [summary["old"][m] for m in metrics], 0.4, label="Otsu anterior")
    ax.bar(x + 0.2, [summary["new"][m] for m in metrics], 0.4, label="Segmentación validada")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proporción")
    ax.set_title("Cobertura de las cajas de lesión TB")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORTS / "03_segmentacion_comparacion.png", bbox_inches="tight")
    plt.close()

    # Distribuciones por clase.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col, title in zip(axes, ["area", "balance", "contrast"],
                              ["Área / cuerpo", "Balance izquierda-derecha", "Contraste"]):
        for cls in ["health", "sick", "tb"]:
            vals = qdf[qdf["class_name"] == cls][col]
            ax.hist(vals, bins=25, alpha=0.5, label=cls)
        ax.set_title(title)
        ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "03_segmentacion_distribucion.png", bbox_inches="tight")
    plt.close()

    print("Figuras guardadas en reports/.")


if __name__ == "__main__":
    main()
