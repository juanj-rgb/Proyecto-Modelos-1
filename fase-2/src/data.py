"""Lectura y validacion de datos, compartida por train.py y predict.py."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from .errors import AplicacionError

MIN_TRAINING_ROWS = 10


def read_csv(path) -> pd.DataFrame:
    """Lee un CSV y falla con un mensaje claro si no existe o no se puede leer."""
    p = Path(path)
    if not p.exists():
        raise AplicacionError(f"No existe el archivo: {p}")
    if not p.is_file():
        raise AplicacionError(f"La ruta no es un archivo: {p}")
    try:
        df = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise AplicacionError(f"El archivo esta vacio: {p}") from exc
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as exc:
        raise AplicacionError(f"No se pudo leer {p} como CSV: {exc}") from exc
    if df.empty:
        raise AplicacionError(f"El archivo no tiene filas de datos: {p}")
    return df


def file_sha256(path) -> str:
    """Huella del archivo de datos, para identificar con que datos se entreno."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def _to_text(series: pd.Series) -> pd.Series:
    """Convierte a texto los valores no nulos; los nulos quedan como NaN."""
    return series.map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)


def coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convierte columnas a numerico. Un valor presente pero no convertible es un
    error (no se convierte en nulo en silencio); una celda vacia si es nulo valido."""
    out = df.copy()
    for col in columns:
        original = out[col]
        convertida = pd.to_numeric(original, errors="coerce")
        invalidas = original.notna() & convertida.isna()
        if invalidas.any():
            # +2: el indice empieza en 0 y la linea 1 del CSV es el encabezado
            lineas = [int(i) + 2 for i in out.index[invalidas][:5]]
            raise AplicacionError(
                f"La columna '{col}' debe ser numerica pero tiene "
                f"{int(invalidas.sum())} valor(es) no numericos "
                f"(lineas del CSV: {lineas})."
            )
        out[col] = convertida.astype("float64")
    return out


def prepare_training_frame(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Valida el CSV de entrenamiento. Devuelve (datos listos, filas sin objetivo descartadas)."""
    if target in FEATURES:
        raise AplicacionError(
            f"La variable objetivo '{target}' no puede ser tambien una variable predictora."
        )
    if target not in df.columns:
        raise AplicacionError(
            f"La variable objetivo '{target}' no existe en el CSV. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    faltan = [c for c in FEATURES if c not in df.columns]
    if faltan:
        raise AplicacionError(f"Faltan columnas requeridas para entrenar: {faltan}")

    df = coerce_numeric_columns(df, NUMERIC_FEATURES)
    y = pd.to_numeric(df[target], errors="coerce")
    no_numericos = df[target].notna() & y.isna()
    if no_numericos.any():
        lineas = [int(i) + 2 for i in df.index[no_numericos][:5]]
        raise AplicacionError(
            f"La variable objetivo '{target}' tiene valores no numericos "
            f"(lineas del CSV: {lineas})."
        )
    df[target] = y
    for col in CATEGORICAL_FEATURES:
        df[col] = _to_text(df[col])

    antes = len(df)
    df = df[df[target].notna()].reset_index(drop=True)
    descartadas = antes - len(df)
    if len(df) < MIN_TRAINING_ROWS:
        raise AplicacionError(
            f"Hay muy pocas filas con objetivo valido ({len(df)}); minimo {MIN_TRAINING_ROWS}."
        )
    return df, descartadas


def prepare_prediction_frame(df: pd.DataFrame, schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Valida y ordena el CSV de entrada segun el esquema guardado con el modelo.

    Devuelve (X con las columnas en el orden del entrenamiento, lista de advertencias).
    Columnas adicionales: se ignoran con advertencia.
    Categorias no vistas: se codifican con todas las columnas one-hot en cero, con advertencia.
    Celdas vacias: se imputan con la estrategia aprendida en el entrenamiento.
    """
    numericas = list(schema["numeric"])
    categoricas = list(schema["categorical"])
    requeridas = numericas + categoricas

    faltan = [c for c in requeridas if c not in df.columns]
    if faltan:
        raise AplicacionError(
            f"Faltan columnas requeridas en el archivo de entrada: {faltan}. "
            f"Columnas esperadas: {requeridas}"
        )

    advertencias: list[str] = []
    extras = [c for c in df.columns if c not in requeridas]
    if extras:
        advertencias.append(f"Columnas adicionales ignoradas por el modelo: {extras}")

    X = coerce_numeric_columns(df[requeridas], numericas)
    for col in categoricas:
        X[col] = _to_text(X[col])
        conocidas = set(schema.get("known_categories", {}).get(col, []))
        nuevas = sorted(set(X[col].dropna()) - conocidas)
        if nuevas:
            n = int(X[col].isin(nuevas).sum())
            advertencias.append(
                f"Categorias no vistas en el entrenamiento para '{col}': {nuevas} "
                f"({n} fila(s)); se codifican con todas las columnas one-hot en cero."
            )
    return X[requeridas], advertencias
