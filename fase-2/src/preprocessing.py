"""Preprocesamiento: un unico ColumnTransformer usado en entrenamiento y prediccion.

Todo se aprende SOLO con los datos con los que se haga fit. Si se usa validacion
cruzada, el pipeline completo se reajusta dentro de cada particion.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Numericas: imputacion por mediana. Categoricas: imputacion por moda + One-Hot.

    handle_unknown='ignore' conserva las categorias aprendidas en el entrenamiento y
    codifica una categoria nueva con ceros; no depende del lote que se transforme
    (a diferencia de pd.get_dummies con drop_first).
    """
    transformadores = []
    if numeric:
        transformadores.append(("num", SimpleImputer(strategy="median"), list(numeric)))
    if categorical:
        cat = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformadores.append(("cat", cat, list(categorical)))
    return ColumnTransformer(
        transformadores, remainder="drop", verbose_feature_names_out=False
    )
