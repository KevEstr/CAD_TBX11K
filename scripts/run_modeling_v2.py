"""Ejecuta el protocolo de modelado y evaluación v2 (TB vs resto).

Uso:
    python scripts/run_modeling_v2.py --quick
    python scripts/run_modeling_v2.py

Salidas:
    data/v2/*.csv, *.json
    models/v2_lr_texture.joblib
    reports/v2/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import get_feature_groups  # noqa: E402
from src.modeling import (  # noqa: E402
    FEATURE_SETS,
    K_BY_SET,
    binary_metrics,
    bootstrap_ci,
    columns_for_set,
    make_pipeline,
    mcnemar_holm,
    nested_cv,
    prior_correction,
    threshold_for_sensitivity,
    youden_threshold,
    PARAM_GRIDS,
)

DATA = ROOT / "data"
OUT = DATA / "v2"
REPORTS = ROOT / "reports" / "v2"
MODELS = ROOT / "models"
for p in (OUT, REPORTS, MODELS):
    p.mkdir(parents=True, exist_ok=True)


def load_data(quick: bool = False):
    df = pd.read_csv(DATA / "features_tbx11k_v2.csv")
    if quick:
        df = pd.concat([
            g.sample(min(500, len(g)), random_state=42)
            for _, g in df.groupby("class_name")
        ]).reset_index(drop=True)
    y_bin = (df["class_name"] == "tb").astype(int).to_numpy()
    y_multi = df["label"].to_numpy()
    return df, y_bin, y_multi


def run_one(df, y, feature_set, model_name, outer, inner, nested_threshold=False):
    cols = columns_for_set(df, feature_set)
    X = df[cols].to_numpy(dtype=np.float32)
    print(f"\n>>> {feature_set} | {model_name} | X={X.shape} | k={K_BY_SET[feature_set]}")
    t0 = time.time()
    res = nested_cv(
        X, y, model_name, feature_set, outer, inner,
        nested_threshold=nested_threshold,
    )
    elapsed = time.time() - t0
    print(f"    {elapsed/60:.1f} min")

    thr_j = youden_threshold(y, res.oof)
    thr_s90 = threshold_for_sensitivity(y, res.oof, 0.90)
    m05 = binary_metrics(y, res.oof, 0.5)
    mj = binary_metrics(y, res.oof, thr_j)
    m90 = binary_metrics(y, res.oof, thr_s90)
    ci = bootstrap_ci(y, res.oof, 0.5)

    stem = f"{feature_set}__{model_name}"
    pd.DataFrame({
        "y_true": y, "proba": res.oof,
        "threshold_youden": thr_j,
        "threshold_sens90": thr_s90,
        "yhat_youden": (res.oof >= thr_j).astype(int),
        "yhat_sens90": (res.oof >= thr_s90).astype(int),
    }).to_csv(OUT / f"oof__{stem}.csv", index=False)
    res.folds.to_csv(OUT / f"folds__{stem}.csv", index=False)

    payload = {
        "feature_set": feature_set,
        "model": model_name,
        "n_features": int(len(cols)),
        "k": int(K_BY_SET[feature_set]),
        "seconds": round(elapsed, 2),
        "metrics_at_0.5": m05,
        "metrics_youden": mj,
        "metrics_sens90": m90,
        "threshold_youden": thr_j,
        "threshold_sens90": thr_s90,
        "bootstrap_ci_at_0.5": ci,
        "best_params": res.best_params,
    }
    (OUT / f"metrics__{stem}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if nested_threshold and res.nested_thresholds is not None:
        # Métricas fold a fold con el umbral elegido dentro del fold.
        rows = []
        for fold, (tr, te) in enumerate(outer, 1):
            thr = res.nested_thresholds[te][0]
            mm = binary_metrics(y[te], res.oof[te], thr)
            mm.update({"fold": fold, "threshold": float(thr)})
            rows.append(mm)
        fd = pd.DataFrame(rows)
        fd.to_csv(OUT / f"nested_threshold__{stem}.csv", index=False)
        payload["nested_threshold_folds"] = fd.to_dict(orient="records")
        payload["nested_threshold_mean"] = {
            "balanced_accuracy": float(fd["balanced_accuracy"].mean()),
            "sensitivity_tb": float(fd["sensitivity_tb"].mean()),
            "specificity": float(fd["specificity"].mean()),
            "ap": float(fd["ap"].mean()),
        }
        (OUT / f"metrics__{stem}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload, res


def fit_final(df, y, feature_set, model_name):
    cols = columns_for_set(df, feature_set)
    X = df[cols].to_numpy(dtype=np.float32)
    k = K_BY_SET[feature_set]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(
        make_pipeline(model_name, X.shape[1], k),
        PARAM_GRIDS[model_name], scoring="balanced_accuracy",
        cv=inner, n_jobs=1, refit=True,
    )
    gs.fit(X, y)
    return gs.best_estimator_, cols, gs.best_params_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-heavy", action="store_true")
    args = parser.parse_args()

    df, y, y_multi = load_data(quick=args.quick)
    print(f"Filas: {len(df)} | TB={int(y.sum())} | resto={int((1-y).sum())}")
    prevalence = float(y.mean())

    outer = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, y))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    runs = [
        ("texture", "dummy"),
        ("texture", "lr"),
        ("texture", "svm"),
        ("texture", "rf"),
        ("texture", "xgb"),
        ("hog", "lr"),
        ("texture_hog", "lr"),
        ("shape", "lr"),
        ("shape_quality", "lr"),
        ("context", "lr"),
        ("texture_shape_context", "lr"),
        ("all", "lr"),
    ]
    if args.skip_heavy:
        runs = [r for r in runs if r[1] in ("dummy", "lr")]

    summary = []
    oofs = {}
    for fs, model in runs:
        payload, res = run_one(df, y, fs, model, outer, inner,
                               nested_threshold=(fs == "texture" and model == "lr"))
        oofs[f"{fs}__{model}"] = res.oof
        summary.append({
            "feature_set": fs,
            "model": model,
            "n_features": payload["n_features"],
            "balacc_0.5": payload["metrics_at_0.5"]["balanced_accuracy"],
            "sens_0.5": payload["metrics_at_0.5"]["sensitivity_tb"],
            "spec_0.5": payload["metrics_at_0.5"]["specificity"],
            "ap_0.5": payload["metrics_at_0.5"]["ap"],
            "auc_0.5": payload["metrics_at_0.5"]["auc"],
            "balacc_youden": payload["metrics_youden"]["balanced_accuracy"],
            "sens_youden": payload["metrics_youden"]["sensitivity_tb"],
            "spec_youden": payload["metrics_youden"]["specificity"],
            "ap_youden": payload["metrics_youden"]["ap"],
        })
        pd.DataFrame(summary).to_csv(OUT / "summary_models.csv", index=False)

    # Corrección de prevalencia para LR textura.
    key = "texture__lr"
    if key in oofs:
        p_raw = oofs[key]
        p_cal = prior_correction(p_raw, prevalence)
        thr_j = youden_threshold(y, p_cal)
        cal = {
            "prevalence": prevalence,
            "raw_0.5": binary_metrics(y, p_raw, 0.5),
            "calibrated_0.5": binary_metrics(y, p_cal, 0.5),
            "calibrated_youden": binary_metrics(y, p_cal, thr_j),
            "brier_raw": float(np.mean((p_raw - y) ** 2)),
            "brier_calibrated": float(np.mean((p_cal - y) ** 2)),
            "threshold_youden": thr_j,
        }
        (OUT / "calibration_lr_texture.json").write_text(
            json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Curva de calibración.
        fig, ax = plt.subplots(figsize=(5, 5))
        for label, p, color in [("Sin calibrar", p_raw, "#2166AC"),
                                ("Calibrada a prevalencia", p_cal, "#B2182B")]:
            frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, "o-", label=label, color=color)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("Probabilidad predicha")
        ax.set_ylabel("Fracción de positivos")
        ax.set_title("Calibración LR textura (TB vs resto)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS / "calibracion_lr_texture.png", bbox_inches="tight")
        plt.close()

    # McNemar entre las corridas principales (umbral de Youden por modelo).
    preds = {}
    for name, p in oofs.items():
        preds[name] = (p >= youden_threshold(y, p)).astype(int)
    mc = mcnemar_holm(y, preds)
    mc.to_csv(OUT / "mcnemar_holm.csv", index=False)

    # Figuras comparativas.
    s = pd.DataFrame(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    order = s.sort_values("balacc_youden", ascending=True)
    axes[0].barh(order["model"] + " | " + order["feature_set"],
                 order["balacc_youden"], color="#2166AC")
    axes[0].axvline(0.5, color="k", ls="--", lw=0.8)
    axes[0].set_xlabel("Balanced accuracy (umbral Youden)")
    axes[0].set_title("Comparación de familias y modelos")
    order2 = s.sort_values("ap_youden", ascending=True)
    axes[1].barh(order2["model"] + " | " + order2["feature_set"],
                 order2["ap_youden"], color="#B2182B")
    axes[1].set_xlabel("Average Precision TB")
    axes[1].set_title("AP TB (umbral Youden)")
    plt.tight_layout()
    plt.savefig(REPORTS / "comparacion_familias.png", bbox_inches="tight")
    plt.close()

    # ROC/PR de las corridas LR.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, p in oofs.items():
        if not name.endswith("__lr"):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, lw=1.2, label=name.replace("__lr", ""))
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, lw=1.2, label=name.replace("__lr", ""))
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_xlabel("1 - Especificidad")
    axes[0].set_ylabel("Sensibilidad")
    axes[0].set_title("ROC")
    axes[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("Recall TB")
    axes[1].set_ylabel("Precisión TB")
    axes[1].set_title("Precision-Recall")
    for ax in axes:
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORTS / "roc_pr_lr.png", bbox_inches="tight")
    plt.close()

    # Matriz de confusión del modelo recomendado.
    if key in oofs:
        thr = youden_threshold(y, oofs[key])
        m = binary_metrics(y, oofs[key], thr)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=["resto", "TB"]).plot(ax=ax, colorbar=False)
        ax.set_title(f"LR textura | umbral {thr:.3f}")
        plt.tight_layout()
        plt.savefig(REPORTS / "confusion_lr_texture.png", bbox_inches="tight")
        plt.close()

    # Curva umbral-sensibilidad/especificidad.
    if key in oofs:
        p = oofs[key]
        ths = np.linspace(0.01, 0.99, 99)
        sens, spec = [], []
        for t in ths:
            m = binary_metrics(y, p, t)
            sens.append(m["sensitivity_tb"])
            spec.append(m["specificity"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ths, sens, label="Sensibilidad TB")
        ax.plot(ths, spec, label="Especificidad")
        ax.axvline(youden_threshold(y, p), color="k", ls="--", lw=0.8, label="Youden")
        ax.axhline(0.90, color="r", ls=":", lw=0.8, label="Sensibilidad 0.90")
        ax.set_xlabel("Umbral")
        ax.set_ylabel("Proporción")
        ax.set_title("LR textura: umbral vs métricas")
        ax.legend()
        plt.tight_layout()
        plt.savefig(REPORTS / "umbral_lr_texture.png", bbox_inches="tight")
        plt.close()

    # Modelo final recomendado.
    best, cols, best_params = fit_final(df, y, "texture", "lr")
    joblib.dump(best, MODELS / "v2_lr_texture.joblib")
    selector = best.named_steps["select"]
    support = selector.get_support()
    selected = [c for c, s in zip(cols, support) if s]
    coefs = best.named_steps["clf"].coef_.ravel()
    coef_df = pd.DataFrame({"feature": selected, "coefficient": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df.sort_values("coefficient", ascending=False).to_csv(
        OUT / "lr_texture_coeficientes.csv", index=False
    )
    manifest = {
        "model": "LogisticRegression",
        "feature_set": "texture",
        "n_features_input": len(cols),
        "n_features_selected": int(support.sum()),
        "selected_features": selected,
        "best_params": best_params,
        "prevalence": prevalence,
        "threshold_youden": float(youden_threshold(y, prior_correction(oofs[key], prevalence))) if key in oofs else None,
        "threshold_sens90": float(threshold_for_sensitivity(y, prior_correction(oofs[key], prevalence), 0.90)) if key in oofs else None,
        "note": "La selección y los hiperparámetros se ajustaron dentro de cada fold; el modelo final se reentrena con todos los datos.",
    }
    (MODELS / "v2_lr_texture_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Mapa espacial HOG (coeficientes del modelo texture_hog).
    key_hog = "texture_hog__lr"
    if key_hog in oofs:
        best_hog, cols_hog, _ = fit_final(df, y, "texture_hog", "lr")
        sel_hog = best_hog.named_steps["select"].get_support()
        hog_cols = [c for c, s in zip(cols_hog, sel_hog) if s and c.startswith("hog_")]
        coefs_hog = best_hog.named_steps["clf"].coef_.ravel()
        coef_map = {c: coefs_hog[i] for i, c in enumerate([c for c, s in zip(cols_hog, sel_hog) if s])}
        grid = np.zeros((7, 7), dtype=float)
        for c in hog_cols:
            idx = int(c.split("_")[1])
            block = idx // 36
            br, bc = divmod(block, 7)
            if 0 <= br < 7 and 0 <= bc < 7:
                grid[br, bc] += abs(coef_map.get(c, 0.0))
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        im = ax.imshow(grid, cmap="magma")
        ax.set_title("Importancia HOG por bloque (|coef| LR)")
        ax.set_xlabel("Bloque horizontal")
        ax.set_ylabel("Bloque vertical")
        plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        plt.savefig(REPORTS / "hog_mapa_espacial.png", bbox_inches="tight")
        plt.close()

    print("\nListo. Salidas en data/v2, models y reports/v2.")


if __name__ == "__main__":
    main()
