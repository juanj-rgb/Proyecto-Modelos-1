"""Hace importable api.py (y, a traves de el, el codigo de fase-2) al correr
`python -m pytest` desde fase-3/.

Tambien ofrece las piezas que comparten las pruebas: un conjunto de datos sinteticos
con el esquema real del proyecto, un modelo entrenado de verdad en una carpeta temporal
y un cliente HTTP contra la aplicacion.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

RAIZ = Path(__file__).resolve().parent.parent          # fase-3/
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Registro valido de referencia: es el mismo ejemplo_solicitud.json que acompana a la API.
REGISTRO_VALIDO = {
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41.0,
    "total_rooms": 880.0,
    "total_bedrooms": 129.0,
    "population": 322.0,
    "households": 126.0,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY",
}


def hacer_datos(n=240, seed=0):
    """Datos sinteticos con el esquema real: ocho numericas, una categorica y el objetivo.

    Incluye nulos en total_bedrooms, que es la unica variable con faltantes en el
    conjunto original, para que el imputador del pipeline tenga algo que aprender.
    """
    rng = np.random.default_rng(seed)
    datos = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, n).round(2),
        "latitude": rng.uniform(32, 42, n).round(2),
        "housing_median_age": rng.integers(1, 53, n).astype(float),
        "total_rooms": rng.lognormal(7.5, 0.5, n).round(),
        "total_bedrooms": rng.lognormal(6.0, 0.5, n).round(),
        "population": rng.lognormal(7.0, 0.5, n).round(),
        "households": rng.lognormal(6.0, 0.5, n).round(),
        "median_income": rng.lognormal(1.2, 0.3, n).round(4),
        "ocean_proximity": rng.choice(["<1H OCEAN", "INLAND", "NEAR BAY", "NEAR OCEAN"], n),
    })
    datos.loc[rng.choice(n, max(1, n // 50), replace=False), "total_bedrooms"] = np.nan
    datos["median_house_value"] = (
        50_000 + 35_000 * datos["median_income"] + rng.normal(0, 8_000, n)
    ).round(1)
    return datos


@pytest.fixture
def csv_datos(tmp_path):
    """CSV de entrenamiento en una carpeta temporal."""
    ruta = tmp_path / "datos.csv"
    hacer_datos().to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def entorno(tmp_path, csv_datos, monkeypatch):
    """Apunta la API a una carpeta de modelos y un CSV temporales.

    Las funciones de api.py leen las variables de entorno en cada solicitud, de modo
    que basta con definirlas antes de llamar al cliente.
    """
    modelos = tmp_path / "models"
    modelos.mkdir()
    monkeypatch.setenv("MODELS_DIR", str(modelos))
    monkeypatch.setenv("DATA_PATH", str(csv_datos))
    monkeypatch.setenv("MODEL_VERSION", "v1")
    monkeypatch.setenv("SEED", "42")
    return {"modelos": modelos, "datos": csv_datos}


@pytest.fixture
def cliente(entorno):
    """Cliente HTTP contra la aplicacion, sin levantar un servidor real."""
    import api
    api._CACHE.clear()          # el cache es global: se limpia entre pruebas
    return TestClient(api.app)


@pytest.fixture
def modelo_entrenado(entorno, cliente):
    """Entrena de verdad una version 'v1' y devuelve la carpeta de modelos.

    Se entrena por la propia API para no duplicar la logica de fase-2.
    """
    r = cliente.post("/train", json={"version": "v1", "n_estimators": 5})
    assert r.status_code == 201, r.text
    return entorno["modelos"]
