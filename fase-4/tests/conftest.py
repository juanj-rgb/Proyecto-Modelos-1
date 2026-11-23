"""Fixtures: un modelo pequeno entrenado sobre datos sinteticos, en una carpeta temporal.

Las pruebas no tocan fase-2/models ni necesitan housing.csv ni red.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import api  # noqa: E402  (agrega fase-2 al sys.path)
import train  # noqa: E402
from src import config  # noqa: E402

CATEGORIAS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"]
VERSION_PRUEBA = "vtest"

# Un registro valido, con la forma exacta de la solicitud JSON.
REGISTRO_VALIDO = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
    "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
    "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
}


def hacer_datos(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, n),
        "latitude": rng.uniform(32.5, 42, n),
        "housing_median_age": rng.integers(1, 53, n).astype(float),
        "total_rooms": rng.integers(100, 8000, n).astype(float),
        "total_bedrooms": rng.integers(50, 1500, n).astype(float),
        "population": rng.integers(100, 4000, n).astype(float),
        "households": rng.integers(50, 1500, n).astype(float),
        "median_income": rng.uniform(0.5, 12, n),
        "ocean_proximity": rng.choice(CATEGORIAS, n),
    })
    df[config.DEFAULT_TARGET] = df["median_income"] * 30000 + rng.normal(0, 20000, n)
    df.loc[rng.choice(n, 5, replace=False), "total_bedrooms"] = np.nan
    return df


@pytest.fixture(scope="session")
def entorno_modelo(tmp_path_factory):
    """Entrena UNA vez un modelo pequeno ('vtest') y devuelve las rutas de trabajo."""
    base = tmp_path_factory.mktemp("api")
    csv = base / "housing.csv"
    hacer_datos().to_csv(csv, index=False)
    modelos = base / "models"
    codigo = train.main(["--data", str(csv), "--target", config.DEFAULT_TARGET,
                         "--output-dir", str(modelos), "--version", VERSION_PRUEBA,
                         "--seed", "42", "--n-estimators", "10", "--cv-folds", "2"])
    assert codigo == 0
    return {"modelos": modelos, "csv": csv}


@pytest.fixture
def cliente(entorno_modelo, monkeypatch, tmp_path):
    """Cliente de pruebas de FastAPI apuntando al modelo pequeno."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("MODELS_DIR", str(entorno_modelo["modelos"]))
    monkeypatch.setenv("DATA_PATH", str(entorno_modelo["csv"]))
    monkeypatch.setenv("MODEL_VERSION", VERSION_PRUEBA)
    monkeypatch.setenv("LOG_PATH", str(tmp_path / "logs" / "solicitudes.csv"))  # FASE 4
    return TestClient(api.app)


@pytest.fixture
def ruta_log(cliente):  # FASE 4
    """Ruta del registro que usa el cliente de pruebas (aun no existe hasta la 1.a llamada)."""
    import os
    return Path(os.environ["LOG_PATH"])
