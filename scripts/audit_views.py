"""Comparación espacial controlada A/B/C/D.

A: imagen completa.
B: región pulmonar.
C: contexto exterior.
D: pulmones con exterior neutralizado.

Mismas imágenes, folds, familias de textura, selector y búsqueda de
hiperparámetros para las cuatro variantes. Sin HOG para evitar que el borde de
la máscara introduzca una señal artificial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import _normalize_mask, _texture_features  # noqa: E402
from src.preprocessing import preprocess, preprocess_bundle  # noqa: E402

DATA = ROOT / "data"
VISTAS = ["full", "thorax", "context", "lung"]
TAREAS = ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy", "tres_clases"]



def _dhash(path: str) -> int:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return hash(path)
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = img[:, 1:] > img[:, :-1]
    return int("".join("1" if x else "0" for x in diff.flatten()), 2)


def _grupos(paths):
    p = DATA / "09_dhash.csv"
    if p.exists():
        d = pd.read_csv(p)
        mapping = dict(zip(d["image_path"], d["dhash"]))
        return np.asarray([mapping.get(str(x), _dhash(str(x))) for x in paths])
    return np.asarray([_dhash(str(x)) for x in paths])


def _indices_grupo(rng, groups):
    groups = np.asarray(groups)
    unicos = np.unique(groups)
    por_grupo = [np.where(groups == g)[0] for g in unicos]
    elegidos = rng.choice(len(unicos), size=len(unicos), replace=True)
    return np.concatenate([por_grupo[i] for i in elegidos])


def _metricas(y, score, pred=None) -> dict:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    if pred is None:
        pred = (score >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return {"AUC": np.nan, "AP": np.nan, "Balanced Acc": np.nan,
                "F1": np.nan, "Recall": np.nan, "Specificity": np.nan}
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "AUC": float(roc_auc_score(y, score)),
        "AP": float(average_precision_score(y, score)),
        "Balanced Acc": float(balanced_accuracy_score(y, pred)),
        "F1": float(f1_score(y, pred, average="binary")),
        "Recall": tp / max(tp + fn, 1),
        "Specificity": tn / max(tn + fp, 1),
    }


def _bootstrap(y, score, pred, rng, groups=None, n_boot=500) -> tuple[dict, dict, dict]:
    base = _metricas(y, score, pred)
    vals = {k: [] for k in base}
    for _ in range(n_boot):
        idx = _indices_grupo(rng, groups if groups is not None else np.arange(len(y)))
        if len(np.unique(y[idx])) < 2:
            continue
        m = _metricas(y[idx], score[idx], pred[idx])
        for k in vals:
            vals[k].append(m[k])
    ci = {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in vals.items() if v
    }
    return base, ci, vals


def _tarea(y, oof, tarea):
    if tarea == "tres_clases":
        pred = oof.argmax(axis=1)
        return y, pred, pred.astype(float), None
    if tarea == "tb_vs_resto":
        mask = np.ones(len(y), dtype=bool)
        yb = (y == 2).astype(int)
    elif tarea == "tb_vs_sick":
        mask = np.isin(y, [1, 2])
        yb = (y[mask] == 2).astype(int)
    elif tarea == "tb_vs_healthy":
        mask = np.isin(y, [0, 2])
        yb = (y[mask] == 2).astype(int)
    else:
        raise ValueError(tarea)
    return yb, (oof[mask, 2] >= 0.5).astype(int), oof[mask, 2], mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1000)
    args = parser.parse_args()

    meta = pd.read_csv(DATA / "full_meta.csv")
    rng = np.random.RandomState(42)
    idx = []
    for clase in [0, 1, 2]:
        sub = np.where(meta["label"].to_numpy() == clase)[0]
        n = min(args.per_class, len(sub))
        idx.extend(rng.choice(sub, size=n, replace=False))
    idx = np.array(idx)
    y = meta.loc[idx, "label"].to_numpy()
    paths = meta.loc[idx, "image_path"].to_numpy()

    filas = {v: [] for v in VISTAS}
    for k, path in enumerate(paths, start=1):
        bundle = preprocess_bundle(path, intensity_norm="none")
        if bundle is None:
            for v in VISTAS:
                filas[v].append(None)
            continue
        thorax, mask = bundle["thorax"]
        context, context_mask = bundle["context"]
        full = preprocess(path, modo="full")
        if full is None:
            for v in VISTAS:
                filas[v].append(None)
            continue

        full_mask = np.ones_like(mask, dtype=bool)
        full_norm = _normalize_mask(full, full_mask)
        thorax_norm = _normalize_mask(thorax, mask)
        context_norm = _normalize_mask(context, context_mask)
        med = int(np.median(thorax_norm[mask])) if mask.any() else 127
        lung_norm = np.where(mask, thorax_norm, med).astype(np.uint8)

        filas["full"].append(_texture_features(full_norm, full_mask, ""))
        filas["thorax"].append(_texture_features(thorax_norm, mask, ""))
        filas["context"].append(_texture_features(context_norm, context_mask, ""))
        filas["lung"].append(_texture_features(lung_norm, np.ones_like(mask), ""))
        if k % 250 == 0:
            print(f"Vistas {k}/{len(paths)}")

    valid = np.array([r is not None for r in filas["full"]])
    idx = idx[valid]
    y = y[valid]
    paths = paths[valid]
    for v in VISTAS:
        filas[v] = [f for f in filas[v] if f is not None]

    feature_cols = list(filas["full"][0].keys())
    grupos = _grupos(paths)
    cv = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(y)), y, groups=grupos
        )
    )
    oofs = {}
    for vista in VISTAS:
        X = np.asarray([[f[c] for c in feature_cols] for f in filas[vista]], dtype=np.float32)
        oof = np.zeros((len(y), 3), dtype=float)
        for tr, va in cv:
            k = min(40, X.shape[1])
            sel = SelectKBest(f_classif, k=k).fit(X[tr], y[tr])
            scaler = StandardScaler().fit(sel.transform(X[tr]))
            clf = GridSearchCV(
                LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
                {"C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]},
                scoring="balanced_accuracy",
                cv=StratifiedKFold(3, shuffle=True, random_state=42),
                n_jobs=-1,
            )
            clf.fit(scaler.transform(sel.transform(X[tr])), y[tr])
            oof[va] = clf.predict_proba(scaler.transform(sel.transform(X[va])))
        oofs[vista] = oof

    filas_metricas = []
    filas_diff = []
    boot_rng = np.random.RandomState(123)
    for tarea in TAREAS:
        for vista in VISTAS:
            if tarea == "tres_clases":
                pred = oofs[vista].argmax(axis=1)
                base = {
                    "AUC": np.nan, "AP": np.nan,
                    "Balanced Acc": float(balanced_accuracy_score(y, pred)),
                    "F1": float(f1_score(y, pred, average="macro")),
                    "Recall": np.nan, "Specificity": np.nan,
                }
                ci = {}
            else:
                yt, pred, score, mask_t = _tarea(y, oofs[vista], tarea)
                g_t = grupos if mask_t is None else grupos[mask_t]
                base, ci, _ = _bootstrap(yt, score, pred, boot_rng, groups=g_t)
            filas_metricas.append({
                "vista": vista, "tarea": tarea, "n": int(len(y)),
                **base,
                **{f"{k}_low": v[0] for k, v in ci.items()},
                **{f"{k}_high": v[1] for k, v in ci.items()},
            })

        if tarea == "tres_clases":
            continue
        for a in VISTAS:
            for b in VISTAS:
                if a == b:
                    continue
                ya, pa, sa, mask_a = _tarea(y, oofs[a], tarea)
                yb, pb, sb, _ = _tarea(y, oofs[b], tarea)
                g_t = grupos if mask_a is None else grupos[mask_a]
                deltas = []
                for _ in range(500):
                    ii = _indices_grupo(boot_rng, g_t)
                    if len(np.unique(ya[ii])) < 2:
                        continue
                    deltas.append(
                        roc_auc_score(ya[ii], sa[ii]) - roc_auc_score(yb[ii], sb[ii])
                    )
                if deltas:
                    filas_diff.append({
                        "tarea": tarea, "vista_a": a, "vista_b": b,
                        "delta_auc": float(np.mean(deltas)),
                        "delta_low": float(np.percentile(deltas, 2.5)),
                        "delta_high": float(np.percentile(deltas, 97.5)),
                    })

    oof_rows = []
    for i, path in enumerate(paths):
        for vista in VISTAS:
            oof_rows.append({
                "image_path": path, "vista": vista, "y_true": int(y[i]),
                "prob_health": float(oofs[vista][i, 0]),
                "prob_sick": float(oofs[vista][i, 1]),
                "prob_tb": float(oofs[vista][i, 2]),
            })
    pd.DataFrame(oof_rows).to_csv(DATA / "auditoria_vistas_oof.csv", index=False)
    pd.DataFrame(filas_metricas).to_csv(
        DATA / "auditoria_vistas_metricas.csv", index=False
    )
    pd.DataFrame(filas_diff).to_csv(
        DATA / "auditoria_vistas_diferencias.csv", index=False
    )
    print(pd.DataFrame(filas_metricas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
