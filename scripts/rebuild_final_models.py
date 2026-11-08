"""Reconstruye los modelos finales del notebook 08 desde el manifiesto.

Uso:
    python scripts/rebuild_final_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.selection import build_selector  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
CLASS_NAMES = ["health", "sick", "tb"]
SLUGS = {
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
}


def main() -> None:
    manifest = json.loads((MODELS / "08_manifest.json").read_text(encoding="utf-8"))
    estrategia = manifest["estrategia"]
    metodo = manifest["selector"]["metodo"]
    n_sel = int(manifest["selector"]["n_features"])
    params = manifest["mejores_params"]
    X = np.load(DATA / "X_full.npy")
    y = np.load(DATA / "y_full.npy")

    cw = "balanced" if estrategia == "class_weight" else None
    rfcw = "balanced_subsample" if estrategia == "class_weight" else None

    modelos = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight=cw, max_iter=1000, random_state=42
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="linear", class_weight=cw, cache_size=500,
                random_state=42,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("clf", DecisionTreeClassifier(
                class_weight=cw, random_state=42
            )),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight=rfcw, max_features="sqrt",
                n_jobs=-1, random_state=42,
            )),
        ]),
        "XGBoost": Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                objective="multi:softprob", num_class=len(CLASS_NAMES),
                eval_metric="mlogloss", tree_method="hist",
                n_jobs=-1, random_state=42,
            )),
        ]),
    }

    for nombre, base in modelos.items():
        base = base.set_params(**params.get(nombre, {}))
        selector = build_selector(metodo, n_sel, random_state=42, n_jobs=-1)
        final = Pipeline([("selector", selector)] + list(base.steps))
        if nombre == "XGBoost" and estrategia == "class_weight":
            final.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
        else:
            final.fit(X, y)
        ruta = MODELS / f"08_{SLUGS[nombre]}.joblib"
        joblib.dump(final, ruta)
        print(nombre, "->", ruta.name, f"{ruta.stat().st_size/1024/1024:.1f} MB", "features", final.n_features_in_)


if __name__ == "__main__":
    main()
