"""Mitigación de atajos dentro de TBX11K.

M0: pipeline pulmonar actual.
M1: reweighting por grupos de adquisición.
M3: selección de características estable entre grupos de adquisición.

M2 (augmentation de adquisición) queda para una fase posterior porque exige
volver a extraer características de imágenes transformadas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_views import (  # noqa: E402
    _bootstrap, _grupos, _metricas, _tarea,
)

DATA = ROOT / "data"


def _pesos_adquisicion(clusters, y, cap=10.0):
    df = pd.DataFrame({"cluster": clusters, "y": y})
    counts = df.groupby(["cluster", "y"]).size()
    w = np.array([
        1.0 / counts.loc[(c, yi)] for c, yi in zip(clusters, y)
    ], dtype=float)
    w = np.clip(w, None, cap)
    return w / w.mean()


def _seleccion_robusta(X, y, clusters, k, seed=42):
    overall = f_classif(X, y)[0]
    cluster_f = f_classif(X, clusters)[0]
    folds = []
    for c in np.unique(clusters):
        m = clusters == c
        if len(np.unique(y[m])) < 2:
            continue
        folds.append(f_classif(X[m], y[m])[0])
    if folds:
        arr = np.vstack(folds)
        mean_f = arr.mean(axis=0)
        min_f = arr.min(axis=0)
        estabilidad = min_f / (mean_f + 1e-9)
    else:
        mean_f = overall
        estabilidad = np.ones_like(overall)
    score = overall * estabilidad / (1.0 + cluster_f)
    return np.argsort(score)[::-1][:k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=0)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--k", type=int, default=300)
    args = parser.parse_args()

    X = np.load(DATA / "X_full.npy")
    y = np.load(DATA / "y_full.npy")
    meta = pd.read_csv(DATA / "full_meta.csv")
    audit = pd.read_csv(DATA / "auditoria_adquisicion.csv")
    audit = audit.set_index("image_path").reindex(meta["image_path"]).reset_index()
    acq_cols = [
        c for c in audit.columns if c not in ("image_path", "label")
    ]
    acq = audit[acq_cols].fillna(audit[acq_cols].median()).to_numpy(dtype=np.float32)

    if args.per_class:
        rng = np.random.RandomState(42)
        idx = []
        for clase in [0, 1, 2]:
            sub = np.where(y == clase)[0]
            idx.extend(rng.choice(sub, min(args.per_class, len(sub)), replace=False))
        idx = np.array(idx)
        X, y, meta = X[idx], y[idx], meta.iloc[idx].reset_index(drop=True)
        acq = acq[idx]

    acq_scaled = StandardScaler().fit_transform(acq)
    clusters = KMeans(
        n_clusters=args.clusters, n_init=10, random_state=42
    ).fit_predict(acq_scaled)
    pd.DataFrame({
        "image_path": meta["image_path"], "label": y, "cluster": clusters,
    }).to_csv(DATA / "mitigacion_clusters.csv", index=False)

    grupos = _grupos(meta["image_path"].tolist())
    cv = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            X, y, groups=grupos
        )
    )

    modelos = {
        "LR": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "SVM": SVC(
            kernel="linear", class_weight="balanced", probability=True,
            random_state=42,
        ),
    }
    variantes = ["M0", "M1", "M3"]
    oofs = {}
    filas_oof = []

    for nombre, base in modelos.items():
        for variante in variantes:
            oof = np.zeros((len(y), 3))
            for tr, va in cv:
                if variante == "M3":
                    idx = _seleccion_robusta(
                        X[tr], y[tr], clusters[tr], min(args.k, X.shape[1])
                    )
                    Xtr, Xva = X[tr][:, idx], X[va][:, idx]
                else:
                    sel = SelectKBest(
                        f_classif, k=min(args.k, X.shape[1])
                    ).fit(X[tr], y[tr])
                    Xtr, Xva = sel.transform(X[tr]), sel.transform(X[va])
                scaler = StandardScaler().fit(Xtr)
                Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)
                modelo = clone(base)
                if variante == "M1":
                    modelo.set_params(class_weight=None)
                    w = _pesos_adquisicion(clusters[tr], y[tr])
                    modelo.fit(Xtr_s, y[tr], sample_weight=w)
                else:
                    modelo.fit(Xtr_s, y[tr])
                oof[va] = modelo.predict_proba(Xva_s)
            oofs[(nombre, variante)] = oof
            for i, path in enumerate(meta["image_path"]):
                filas_oof.append({
                    "image_path": path, "modelo": nombre, "variante": variante,
                    "y_true": int(y[i]),
                    "prob_health": float(oof[i, 0]),
                    "prob_sick": float(oof[i, 1]),
                    "prob_tb": float(oof[i, 2]),
                })

    pd.DataFrame(filas_oof).to_csv(DATA / "mitigacion_oof.csv", index=False)

    filas_metricas = []
    filas_diff = []
    rng = np.random.RandomState(123)
    for (nombre, variante), oof in oofs.items():
        for tarea in ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy", "tres_clases"]:
            if tarea == "tres_clases":
                pred = oof.argmax(axis=1)
                filas_metricas.append({
                    "modelo": nombre, "variante": variante, "tarea": tarea,
                    "n": len(y), "AUC": np.nan, "AP": np.nan,
                    "Balanced Acc": float(balanced_accuracy_score(y, pred)),
                    "F1": float(f1_score(y, pred, average="macro")),
                    "Recall": np.nan, "Specificity": np.nan,
                })
                continue
            yt, pred, score, mask = _tarea(y, oof, tarea)
            g_t = grupos if mask is None else grupos[mask]
            base_m, ci, _ = _bootstrap(yt, score, pred, rng, groups=g_t)
            filas_metricas.append({
                "modelo": nombre, "variante": variante, "tarea": tarea,
                "n": len(yt), **base_m,
                **{f"{k}_low": v[0] for k, v in ci.items()},
                **{f"{k}_high": v[1] for k, v in ci.items()},
            })

        for tarea in ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy"]:
            yt, _, s0, mask = _tarea(y, oofs[(nombre, "M0")], tarea)
            g_t = grupos if mask is None else grupos[mask]
            for variante in ["M1", "M3"]:
                yv, _, sv, _ = _tarea(y, oofs[(nombre, variante)], tarea)
                deltas = []
                for _ in range(500):
                    ii = rng.choice(len(np.unique(g_t)), len(np.unique(g_t)), replace=True)
                    unicos = np.unique(g_t)
                    por_grupo = [np.where(g_t == g)[0] for g in unicos]
                    idx = np.concatenate([por_grupo[i] for i in ii])
                    if len(np.unique(yt[idx])) < 2:
                        continue
                    deltas.append(
                        roc_auc_score(yt[idx], sv[idx])
                        - roc_auc_score(yt[idx], s0[idx])
                    )
                if deltas:
                    filas_diff.append({
                        "modelo": nombre, "tarea": tarea,
                        "variante": variante,
                        "delta_auc": float(np.mean(deltas)),
                        "delta_low": float(np.percentile(deltas, 2.5)),
                        "delta_high": float(np.percentile(deltas, 97.5)),
                    })

    pd.DataFrame(filas_metricas).to_csv(
        DATA / "mitigacion_metricas.csv", index=False
    )
    pd.DataFrame(filas_diff).to_csv(
        DATA / "mitigacion_diferencias.csv", index=False
    )
    print(pd.DataFrame(filas_metricas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
