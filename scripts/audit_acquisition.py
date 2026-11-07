"""Auditoría de atributos globales de adquisición.

Grupos:
- intensidad y contraste,
- nitidez y procesamiento,
- geometría y formato,
- combinación.

Cada grupo se evalúa con LR, SVM y Random Forest sobre los mismos folds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_views import _bootstrap, _grupos, _metricas, _tarea  # noqa: E402

DATA = ROOT / "data"


def _features(path: str) -> dict | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    vals = img.astype(np.float64).ravel()
    mean, std = float(vals.mean()), float(vals.std())
    p = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])
    hist = np.histogram(vals, bins=16, range=(0, 255), density=True)[0]
    lap = float(cv2.Laplacian(img, cv2.CV_64F).var())
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = float(np.mean(np.abs(gx) + np.abs(gy)))
    grad_energy = float(np.mean(gx * gx + gy * gy))
    canny = cv2.Canny(img, 50, 150)
    edge_frac = float(np.mean(canny > 0))
    row = {
        "int_mean": mean, "int_std": std, "int_p1": p[0], "int_p5": p[1],
        "int_p25": p[2], "int_p50": p[3], "int_p75": p[4], "int_p95": p[5],
        "int_p99": p[6], "int_iqr": p[4] - p[2], "int_entropy": float(scipy_entropy(hist + 1e-9)),
        "sharp_lap": lap, "sharp_sobel": sobel, "sharp_grad": grad_energy,
        "sharp_edge": edge_frac,
        "geom_h": h, "geom_w": w, "geom_aspect": h / max(w, 1),
        "geom_pixels": h * w,
        "geom_extreme": float(np.mean((img < 5) | (img > 250))),
        "geom_black": float(np.mean(img < 10)),
        "geom_white": float(np.mean(img > 245)),
    }
    for i, v in enumerate(hist):
        row[f"int_hist_{i}"] = float(v)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    meta = pd.read_csv(DATA / "full_meta.csv")
    if args.limit:
        meta = meta.groupby("class_name").head(args.limit).reset_index(drop=True)

    filas, labels, paths = [], [], []
    for _, row in meta.iterrows():
        f = _features(row["image_path"])
        if f is None:
            continue
        filas.append(f)
        labels.append(int(row["label"]))
        paths.append(row["image_path"])
    df = pd.DataFrame(filas)
    y = np.asarray(labels)
    df.insert(0, "image_path", paths)
    df.insert(1, "label", y)
    df.to_csv(DATA / "auditoria_adquisicion.csv", index=False)

    grupos = {
        "intensidad": [c for c in df.columns if c.startswith("int_")],
        "nitidez": [c for c in df.columns if c.startswith("sharp_")],
        "geometria": [c for c in df.columns if c.startswith("geom_")],
    }
    grupos["combinado"] = sum(grupos.values(), [])

    modelos = {
        "LR": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42))]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("clf", SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=42))]),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced_subsample",
            n_jobs=-1, random_state=42),
    }

    dhash_groups = _grupos(paths)
    cv = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            df, y, groups=dhash_groups
        )
    )
    filas_metricas = []
    oof_rows = []
    rng = np.random.RandomState(123)

    for grupo, cols in grupos.items():
        X = df[cols].to_numpy(dtype=np.float32)
        for nombre, base in modelos.items():
            oof = np.zeros((len(y), 3), dtype=float)
            for tr, va in cv:
                modelo = base
                modelo.fit(X[tr], y[tr])
                oof[va] = modelo.predict_proba(X[va])
            for tarea in ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy", "tres_clases"]:
                if tarea == "tres_clases":
                    pred = oof.argmax(axis=1)
                    filas_metricas.append({
                        "grupo": grupo, "modelo": nombre, "tarea": tarea, "n": len(y),
                        "AUC": np.nan, "AP": np.nan,
                        "Balanced Acc": float(balanced_accuracy_score(y, pred)),
                        "F1": float(f1_score(y, pred, average="macro")),
                        "Recall": np.nan, "Specificity": np.nan,
                    })
                else:
                    yt, pred, score, mask = _tarea(y, oof, tarea)
                    g_t = dhash_groups if mask is None else dhash_groups[mask]
                    base_m, ci, _ = _bootstrap(yt, score, pred, rng, groups=g_t)
                    filas_metricas.append({
                        "grupo": grupo, "modelo": nombre, "tarea": tarea, "n": len(yt),
                        **base_m,
                        **{f"{k}_low": v[0] for k, v in ci.items()},
                        **{f"{k}_high": v[1] for k, v in ci.items()},
                    })
            if grupo == "combinado":
                for i, path in enumerate(paths):
                    oof_rows.append({
                        "image_path": path, "modelo": nombre, "y_true": int(y[i]),
                        "prob_health": float(oof[i, 0]),
                        "prob_sick": float(oof[i, 1]),
                        "prob_tb": float(oof[i, 2]),
                    })

    pd.DataFrame(filas_metricas).to_csv(
        DATA / "auditoria_adquisicion_metricas.csv", index=False
    )
    pd.DataFrame(oof_rows).to_csv(
        DATA / "auditoria_adquisicion_oof.csv", index=False
    )
    print(pd.DataFrame(filas_metricas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
