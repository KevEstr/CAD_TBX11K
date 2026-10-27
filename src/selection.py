"""
cad_tbx11k.selection
====================
Selección de características supervisada para el pipeline CAD-TBX11K.

El selector se ajusta SIEMPRE dentro del fold de entrenamiento. Nunca debe
ajustarse con el fold de validación ni con el conjunto completo antes de la
validación cruzada.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class RFTopNSelector(BaseEstimator, TransformerMixin):
    """Selecciona exactamente las top-N features por importancia de Random Forest.

    Se ajusta en ``fit`` con el fold de entrenamiento y selecciona en
    ``transform``. Es un transformador compatible con ``Pipeline`` y
    ``GridSearchCV``.
    """

    def __init__(self, n_features=1000, n_estimators=100,
                 random_state=42, n_jobs=1):
        self.n_features = n_features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf.fit(X, y)
        self.importances_ = rf.feature_importances_
        self.top_idx_ = np.argsort(self.importances_)[::-1][:self.n_features]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return X[:, self.top_idx_]

    def get_support(self, indices=False):
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.top_idx_] = True
        return self.top_idx_ if indices else mask


def build_selector(method, n_features, random_state=42, n_estimators=100, n_jobs=1):
    """Construye el selector correspondiente al método elegido en el notebook 06."""
    if method == 'RF':
        return RFTopNSelector(
            n_features=n_features,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if method == 'Chi2':
        return Pipeline([
            ('mm', MinMaxScaler()),
            ('sel', SelectKBest(chi2, k=n_features)),
        ])
    if method == 'MI':
        return SelectKBest(mutual_info_classif, k=n_features)
    raise ValueError(f'Método de selección desconocido: {method}')


def feature_scores(method, X, y, random_state=42, n_estimators=100):
    """Devuelve el score de cada feature para el método indicado."""
    if method == 'RF':
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)
        return rf.feature_importances_
    if method == 'Chi2':
        Xs = MinMaxScaler().fit_transform(X)
        scores, _ = chi2(Xs, y)
        return scores
    if method == 'MI':
        return mutual_info_classif(X, y, random_state=random_state)
    raise ValueError(f'Método de selección desconocido: {method}')
