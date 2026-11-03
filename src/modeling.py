"""Particion, entrenamiento y evaluacion. Sin lectura de archivos."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from .config import ALGORITHMS, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, ZONE_COLUMNS
from .errors import AplicacionError
from .preprocessing import build_preprocessor


def build_estimator(algorithm: str, seed: int, n_estimators: int = 100, max_depth: int | None = None):
    if algorithm == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
        )
    if algorithm == "decision_tree":
        return DecisionTreeRegressor(max_depth=max_depth or 8, random_state=seed)
    if algorithm == "linear_regression":
        return LinearRegression()
    raise AplicacionError(f"Algoritmo no soportado: '{algorithm}'. Opciones: {list(ALGORITHMS)}")


def estimator_params(estimator) -> dict:
    """Parametros del estimador que se pueden escribir en JSON."""
    return {
        k: v
        for k, v in estimator.get_params().items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }


def zone_groups(X: pd.DataFrame) -> pd.Series:
    """Zona = celda de 1 grado de latitud/longitud (~100 km)."""
    lat, lon = ZONE_COLUMNS
    return np.floor(X[lat]).astype(int).astype(str) + "_" + np.floor(X[lon]).astype(int).astype(str)


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, seed: int, spatial: bool):
    """Separa entrenamiento y prueba ANTES de ajustar cualquier transformacion."""
    if not 0 < test_size < 1:
        raise AplicacionError(f"--test-size debe estar entre 0 y 1 (recibido {test_size}).")
    if spatial:
        if zone_groups(X).nunique() < 5:
            raise AplicacionError("Muy pocas zonas distintas para una particion espacial.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr, te = next(splitter.split(X, y, groups=zone_groups(X)))
        return X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def cross_validate_pipeline(pipeline, X_train, y_train, folds: int, seed: int, spatial: bool):
    """Validacion cruzada SOLO sobre entrenamiento; el pipeline se reajusta en cada fold."""
    if folds < 2:
        return None
    if spatial:
        grupos = zone_groups(X_train)
        n_splits = min(folds, grupos.nunique())
        if n_splits < 2:
            raise AplicacionError("Muy pocas zonas en entrenamiento para validacion cruzada espacial.")
        cv, groups = GroupKFold(n_splits=n_splits), grupos
    else:
        cv, groups = KFold(n_splits=folds, shuffle=True, random_state=seed), None
        n_splits = folds
    r = cross_validate(
        clone(pipeline),
        X_train,
        y_train,
        cv=cv,
        groups=groups,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=1,
    )
    return {
        "folds": int(n_splits),
        "esquema": "por zonas" if spatial else "aleatoria",
        "MAE_media": float(-r["test_mae"].mean()),
        "MAE_desv": float(r["test_mae"].std()),
        "RMSE_media": float(-r["test_rmse"].mean()),
        "R2_media": float(r["test_r2"].mean()),
    }


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    algorithm: str,
    seed: int,
    test_size: float,
    spatial: bool,
    cv_folds: int,
    n_estimators: int = 100,
    max_depth: int | None = None,
) -> dict:
    """Entrena el pipeline completo y lo evalua UNA vez sobre datos de prueba.

    El modelo base (media del entrenamiento) usa la misma particion y las mismas metricas.
    """
    X, y = df[FEATURES], df[target]
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, seed, spatial)
    if len(X_train) < 2 or len(X_test) < 1:
        raise AplicacionError("La particion dejo muy pocas filas en entrenamiento o prueba.")

    estimator = build_estimator(algorithm, seed, n_estimators, max_depth)
    pipeline = Pipeline(
        [
            ("prep", build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)),
            ("model", estimator),
        ]
    )

    cv = cross_validate_pipeline(pipeline, X_train, y_train, cv_folds, seed, spatial)
    pipeline.fit(X_train, y_train)
    metricas_modelo = compute_metrics(y_test, pipeline.predict(X_test))

    # El modelo base ignora X: predice la media de y_train.
    base = DummyRegressor(strategy="mean").fit(np.zeros((len(X_train), 1)), y_train)
    metricas_base = compute_metrics(y_test, base.predict(np.zeros((len(X_test), 1))))

    return {
        "pipeline": pipeline,
        "estimator": estimator,
        "metricas_modelo": metricas_modelo,
        "metricas_base": metricas_base,
        "validacion_cruzada": cv,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "X_train": X_train,
    }
