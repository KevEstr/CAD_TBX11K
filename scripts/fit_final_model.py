"""Ajusta y guarda el modelo final descontaminado.

Modelo: SVM RBF sobre textura estricta residualizada contra contexto/forma.
La residualización se ajusta con todos los datos (el modelo es un artefacto
final, no una estimación de CV). Las métricas que se guardan son las OOF de
`data/v2/strict_deconfound_summary.csv`.

Salida:
    models/v2_svm_strict_resid_context.joblib
    models/v2_svm_strict_resid_context_manifest.json

Uso:
    python scripts/fit_final_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import ContextResidualizer  # noqa: E402

DATA = ROOT / "data"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def main() -> None:
    features_path = DATA / "features_tbx11k_v2.csv"
    strict_path = DATA / "features_strict.csv"
    if not features_path.exists() or not strict_path.exists():
        raise FileNotFoundError(
            "Faltan features. Ejecuta antes:\n"
            "  python scripts/rebuild_features.py --workers 8\n"
            "  python scripts/extract_strict_features.py --workers 8"
        )

    df = pd.read_csv(features_path)
    strict = pd.read_csv(strict_path)
    strict = strict.drop(columns=[c for c in ("class_name", "label") if c in strict.columns])
    df = df.merge(strict, on="image_path", how="left")

    y = (df["class_name"] == "tb").astype(int).to_numpy()
    context = [c for c in df.columns if c.startswith("ctx_")]
    texture = [
        c for c in df.columns
        if c.startswith("strict_") and not c.startswith(("strict_shape_", "strict_thumb_"))
    ]
    cols = texture + context
    X = df[cols].to_numpy(dtype=np.float32)

    model = Pipeline([
        ("resid", ContextResidualizer(n_texture=len(texture), alpha=10.0)),
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=48)),
        ("clf", SVC(probability=True, C=1.0, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    joblib.dump(model, MODELS / "v2_svm_strict_resid_context.joblib")

    summary_path = DATA / "v2" / "strict_deconfound_summary.csv"
    metrics = {}
    if summary_path.exists():
        s = pd.read_csv(summary_path)
        row = s[(s["set"] == "strict_resid_context") & (s["model"] == "svm")]
        if len(row):
            metrics = row.iloc[0].to_dict()

    manifest = {
        "model": "SVC RBF",
        "feature_set": "strict texture + residualized context",
        "n_texture": len(texture),
        "n_context": len(context),
        "n_selected": 48,
        "metrics_oof": metrics,
        "note": (
            "Modelo descontaminado: la textura se residualiza contra contexto "
            "dentro de cada fold en la evaluación. Aquí se ajusta con todos los "
            "datos para uso como artefacto final. Es una cota inferior "
            "conservadora, no el modelo de mayor accuracy."
        ),
    }
    (MODELS / "v2_svm_strict_resid_context_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
