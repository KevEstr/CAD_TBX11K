"""M2: entrenamiento con variaciones moderadas de adquisición.

Entrena con imagen original + variantes de contraste, gamma y nitidez.
Evalúa en originales y en las mismas variantes, con folds agrupados por dHash.
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_views import _bootstrap, _grupos, _metricas, _tarea  # noqa: E402
from src.features import extract_features, model_feature_columns  # noqa: E402

DATA = ROOT / "data"
VARIANTES = {
    "contraste_bajo": lambda img: np.clip(
        img.mean() + (img.astype(np.float32) - img.mean()) * 0.85, 0, 255
    ).astype(np.uint8),
    "gamma_alto": lambda img: np.clip(
        (img.astype(np.float32) / 255.0) ** 1.1 * 255.0, 0, 255
    ).astype(np.uint8),
    "blur": lambda img: cv2.GaussianBlur(img, (3, 3), 0.6),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=1000)
    args = parser.parse_args()

    X = np.load(DATA / "X_full.npy")
    y = np.load(DATA / "y_full.npy")
    meta = pd.read_csv(DATA / "full_meta.csv")
    feat_cols = model_feature_columns(
        pd.read_csv(DATA / "features_tbx11k.csv", nrows=0)
    )

    rng = np.random.RandomState(42)
    if args.per_class:
        idx = []
        for clase in [0, 1, 2]:
            sub = np.where(y == clase)[0]
            idx.extend(rng.choice(sub, min(args.per_class, len(sub)), replace=False))
        idx = np.array(idx)
    else:
        idx = np.arange(len(y))
    Xb, yb, metab = X[idx], y[idx], meta.iloc[idx].reset_index(drop=True)

    aumentadas = {v: [] for v in VARIANTES}
    for k, (_, row) in enumerate(metab.iterrows(), start=1):
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            for v in VARIANTES:
                aumentadas[v].append(None)
            continue
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        for nombre, fn in VARIANTES.items():
            arr = fn(img)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, arr)
            try:
                f = extract_features(
                    tmp_path, int(row["label"]), row["class_name"],
                    include_context=False, include_shape=False,
                )
                aumentadas[nombre].append(
                    None if f is None else np.array(
                        [f[c] for c in feat_cols], dtype=np.float32
                    )
                )
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        if k % 200 == 0:
            print(f"Augment {k}/{len(metab)}")

    valid = np.array([
        all(aumentadas[v][i] is not None for v in VARIANTES)
        for i in range(len(metab))
    ])
    Xb, yb, metab = Xb[valid], yb[valid], metab[valid].reset_index(drop=True)
    metab.to_csv(DATA / 'mitigacion_augmentation_ids.csv', index=False)
    for v in VARIANTES:
        aumentadas[v] = [a for a in aumentadas[v] if a is not None]

    grupos = _grupos(metab["image_path"].tolist())
    cv = list(
        StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            Xb, yb, groups=grupos
        )
    )

    oof = {"M0_base": np.zeros((len(yb), 3)), "M2_base": np.zeros((len(yb), 3))}
    for v in VARIANTES:
        oof[f"M2_{v}"] = np.zeros((len(yb), 3))

    for tr, va in cv:
        sel = SelectKBest(f_classif, k=min(300, Xb.shape[1])).fit(Xb[tr], yb[tr])
        scaler = StandardScaler().fit(sel.transform(Xb[tr]))
        m0 = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ).fit(scaler.transform(sel.transform(Xb[tr])), yb[tr])
        oof["M0_base"][va] = m0.predict_proba(
            scaler.transform(sel.transform(Xb[va]))
        )

        Xtr = [Xb[tr]] + [np.asarray(aumentadas[v])[tr] for v in VARIANTES]
        Xtr = np.vstack(Xtr)
        ytr = np.concatenate([yb[tr]] * (1 + len(VARIANTES)))
        sel2 = SelectKBest(f_classif, k=min(300, Xtr.shape[1])).fit(Xtr, ytr)
        scaler2 = StandardScaler().fit(sel2.transform(Xtr))
        m2 = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ).fit(scaler2.transform(sel2.transform(Xtr)), ytr)
        oof["M2_base"][va] = m2.predict_proba(
            scaler2.transform(sel2.transform(Xb[va]))
        )
        for v in VARIANTES:
            Xv = np.asarray(aumentadas[v])[va]
            oof[f"M2_{v}"][va] = m2.predict_proba(
                scaler2.transform(sel2.transform(Xv))
            )

    filas_metricas, filas_diff = [], []
    rng_boot = np.random.RandomState(123)
    for nombre, pred in oof.items():
        for tarea in ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy", "tres_clases"]:
            if tarea == "tres_clases":
                p = pred.argmax(axis=1)
                filas_metricas.append({
                    "variante": nombre, "tarea": tarea, "n": len(yb),
                    "AUC": np.nan, "AP": np.nan,
                    "Balanced Acc": float(balanced_accuracy_score(yb, p)),
                    "F1": float(f1_score(yb, p, average="macro")),
                    "Recall": np.nan, "Specificity": np.nan,
                })
                continue
            yt, pr, score, mask = _tarea(yb, pred, tarea)
            g_t = grupos if mask is None else grupos[mask]
            base, ci, _ = _bootstrap(yt, score, pr, rng_boot, groups=g_t)
            filas_metricas.append({
                "variante": nombre, "tarea": tarea, "n": len(yt), **base,
                **{f"{k}_low": v[0] for k, v in ci.items()},
                **{f"{k}_high": v[1] for k, v in ci.items()},
            })

        for tarea in ["tb_vs_resto", "tb_vs_sick", "tb_vs_healthy"]:
            yt, _, s0, mask = _tarea(yb, oof["M0_base"], tarea)
            if nombre == "M0_base":
                continue
            yv, _, sv, _ = _tarea(yb, pred, tarea)
            g_t = grupos if mask is None else grupos[mask]
            deltas = []
            for _ in range(500):
                unicos = np.unique(g_t)
                por_grupo = [np.where(g_t == g)[0] for g in unicos]
                ii = rng_boot.choice(len(unicos), len(unicos), replace=True)
                idx_b = np.concatenate([por_grupo[i] for i in ii])
                if len(np.unique(yt[idx_b])) < 2:
                    continue
                deltas.append(
                    roc_auc_score(yt[idx_b], sv[idx_b])
                    - roc_auc_score(yt[idx_b], s0[idx_b])
                )
            if deltas:
                filas_diff.append({
                    "variante": nombre, "tarea": tarea,
                    "delta_auc": float(np.mean(deltas)),
                    "delta_low": float(np.percentile(deltas, 2.5)),
                    "delta_high": float(np.percentile(deltas, 97.5)),
                })

    pd.DataFrame(filas_metricas).to_csv(
        DATA / "mitigacion_augmentation_metricas.csv", index=False
    )
    pd.DataFrame(filas_diff).to_csv(
        DATA / "mitigacion_augmentation_diferencias.csv", index=False
    )
    print(pd.DataFrame(filas_metricas).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
