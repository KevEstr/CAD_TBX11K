"""Utilidades de validación, umbrales y métricas para el modelado."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# Especificaciones

FEATURE_SETS: dict[str, list[str]] = {
    "texture": ["texture"],
    "hog": ["hog"],
    "texture_hog": ["texture", "hog"],
    "shape": ["shape"],
    "shape_quality": ["shape", "quality"],
    "context": ["context"],
    "texture_shape_context": ["texture", "shape", "context"],
    "all": ["texture", "hog", "shape", "context"],
}

# k fijo por familia: la selección se hace dentro de cada fold.
K_BY_SET: dict[str, int] = {
    "texture": 48,
    "hog": 120,
    "texture_hog": 150,
    "shape": 32,
    "shape_quality": 36,
    "context": 48,
    "texture_shape_context": 120,
    "all": 180,
}

PARAM_GRIDS: dict[str, dict] = {
    "dummy": {},
    "lr": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__class_weight": ["balanced", None],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [2000],
    },
    "svm": {
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale"],
        "clf__class_weight": ["balanced"],
    },
    "rf": {
        "clf__n_estimators": [300],
        "clf__max_depth": [None, 10],
        "clf__min_samples_leaf": [1, 5],
        "clf__class_weight": ["balanced_subsample"],
    },
    "xgb": {
        "clf__n_estimators": [300],
        "clf__max_depth": [2, 4],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8],
        "clf__colsample_bytree": [0.8],
    },
}


# Utilidades



def columns_for_set(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Devuelve las columnas de una familia lógica."""
    from .features import get_feature_groups

    groups = get_feature_groups(df)
    cols: list[str] = []
    for g in FEATURE_SETS[feature_set]:
        cols.extend(groups.get(g, []))
    # Sin duplicados, conservando el orden.
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _selector(n_features: int, k: int):
    """Selector univariante ANOVA F."""
    return SelectKBest(f_classif, k=min(k, n_features))


def make_pipeline(model_name: str, n_features: int, k: int) -> Pipeline:
    """Pipeline escalado + selección + clasificador."""
    if model_name == "dummy":
        clf = DummyClassifier(strategy="stratified", random_state=42)
    elif model_name == "lr":
        clf = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "svm":
        clf = SVC(probability=True, random_state=42)
    elif model_name == "rf":
        clf = RandomForestClassifier(random_state=42, n_jobs=1)
    elif model_name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost no está instalado.")
        clf = XGBClassifier(
            random_state=42, n_jobs=1, eval_metric="logloss",
            scale_pos_weight=9.32, tree_method="hist",
        )
    else:
        raise ValueError(f"Modelo desconocido: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", _selector(n_features, k)),
        ("clf", clf),
    ])


def get_param_grid(model_name: str) -> dict:
    return PARAM_GRIDS[model_name]


class ContextResidualizer(BaseEstimator, TransformerMixin):
    """Elimina de la textura la parte explicada por contexto/forma."""

    def __init__(self, n_texture: int, alpha: float = 10.0,
                 keep_nuisance: bool = False):
        self.n_texture = n_texture
        self.alpha = alpha
        self.keep_nuisance = keep_nuisance

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        Xt = X[:, : self.n_texture]
        Xn = X[:, self.n_texture:]
        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(Xn, Xt)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Xt = X[:, : self.n_texture]
        Xn = X[:, self.n_texture:]
        resid = Xt - self.model_.predict(Xn)
        if self.keep_nuisance:
            return np.hstack([resid, Xn])
        return resid


# Métricas

def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Métricas binarias para TB vs resto."""
    y_true = np.asarray(y_true).astype(int)
    yhat = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float((sens + spec) / 2),
        "sensitivity_tb": float(sens),
        "specificity": float(spec),
        "precision_tb": float(prec),
        "f1_tb": float(f1),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, proba)) if y_true.sum() else float("nan"),
        "brier": float(brier_score_loss(y_true, proba)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def threshold_for_sensitivity(y_true: np.ndarray, proba: np.ndarray, target: float = 0.90) -> float:
    """Mayor umbral que mantiene la sensibilidad objetivo."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    thresholds = np.unique(np.round(proba, 6))[::-1]
    for t in thresholds:
        yhat = (proba >= t).astype(int)
        tp = int(((yhat == 1) & (y_true == 1)).sum())
        fn = int(((yhat == 0) & (y_true == 1)).sum())
        sens = tp / max(tp + fn, 1)
        if sens >= target:
            return float(t)
    return 0.5


def youden_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Umbral de Youden (maximiza sensibilidad + especificidad - 1)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    thresholds = np.unique(np.round(proba, 6))
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        yhat = (proba >= t).astype(int)
        tp = int(((yhat == 1) & (y_true == 1)).sum())
        fn = int(((yhat == 0) & (y_true == 1)).sum())
        tn = int(((yhat == 0) & (y_true == 0)).sum())
        fp = int(((yhat == 1) & (y_true == 0)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


def prior_correction(proba: np.ndarray, prevalence: float) -> np.ndarray:
    """Corrige probabilidades balanceadas a la prevalencia real."""
    p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    logit_cal = logit + np.log(prevalence / (1 - prevalence))
    return 1.0 / (1.0 + np.exp(-logit_cal))


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    n_boot: int = 2000,
    random_state: int = 42,
) -> dict:
    """IC 95 % estratificado por bootstrap para métricas clave."""
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    rows = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        m = binary_metrics(y_true[idx], proba[idx], threshold)
        rows.append([m["balanced_accuracy"], m["sensitivity_tb"], m["specificity"],
                     m["ap"], m["auc"]])
    arr = np.array(rows)
    names = ["balanced_accuracy", "sensitivity_tb", "specificity", "ap", "auc"]
    return {
        f"{n}_ci95": [float(np.percentile(arr[:, i], 2.5)), float(np.percentile(arr[:, i], 97.5))]
        for i, n in enumerate(names)
    }


def mcnemar_holm(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """McNemar por pares con corrección de Holm."""
    names = list(predictions)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            correct_a = predictions[a] == y_true
            correct_b = predictions[b] == y_true
            n01 = int((~correct_a & correct_b).sum())
            n10 = int((correct_a & ~correct_b).sum())
            if n01 + n10 == 0:
                p = 1.0
            else:
                p = float(stats.binomtest(n01, n01 + n10, 0.5).pvalue)
            rows.append({"model_a": a, "model_b": b, "b_only_correct": n01,
                         "a_only_correct": n10, "p_value": p})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = np.argsort(out["p_value"].values)
    m = len(out)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * out.iloc[idx]["p_value"])
        running = max(running, val)
        adjusted[idx] = running
    out["p_holm"] = adjusted
    return out


# Validación cruzada anidada

@dataclass
class CVResult:
    oof: np.ndarray
    folds: pd.DataFrame
    best_params: list[dict]
    nested_thresholds: Optional[np.ndarray] = None


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    feature_set: str,
    outer_splits: Iterable,
    inner_splits: Iterable = None,
    nested_threshold: bool = False,
    target_sensitivity: float = 0.90,
    random_state: int = 42,
) -> CVResult:
    """CV anidada con selector e hiperparámetros dentro de cada fold."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    k = K_BY_SET[feature_set]
    oof = np.zeros(len(y), dtype=float)
    fold_rows = []
    best_params = []
    nested_thr = np.full(len(y), np.nan) if nested_threshold else None

    for fold, (tr, te) in enumerate(outer_splits, 1):
        pipe = make_pipeline(model_name, X.shape[1], k)
        grid = get_param_grid(model_name)
        gs = GridSearchCV(
            pipe, grid, scoring="balanced_accuracy",
            cv=inner_splits, n_jobs=1, refit=True,
        )
        gs.fit(X[tr], y[tr])
        best = gs.best_estimator_
        proba = best.predict_proba(X[te])[:, 1]
        oof[te] = proba
        best_params.append(gs.best_params_)

        if nested_threshold:
            inner_proba = cross_val_predict(
                clone(best), X[tr], y[tr], cv=inner_splits,
                method="predict_proba", n_jobs=1,
            )[:, 1]
            thr = threshold_for_sensitivity(y[tr], inner_proba, target_sensitivity)
            nested_thr[te] = thr
        else:
            thr = 0.5

        m = binary_metrics(y[te], proba, thr)
        m.update({"fold": fold, "best_params": str(gs.best_params_)})
        fold_rows.append(m)

    return CVResult(
        oof=oof,
        folds=pd.DataFrame(fold_rows),
        best_params=best_params,
        nested_thresholds=nested_thr,
    )

