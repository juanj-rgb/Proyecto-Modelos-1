"""Pruebas de la aplicacion. Datos sinteticos pequenos, sin red, en directorios temporales.

Ejecutar desde fase-2:  python -m pytest
"""
import hashlib
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import predict
import train
from src import config
from src.data import prepare_prediction_frame, read_csv, prepare_training_frame
from src.errors import AplicacionError
from src.modeling import split_data, train_and_evaluate

RAIZ = Path(__file__).resolve().parent.parent
TARGET = config.DEFAULT_TARGET
CATEGORIAS = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY"]


def hacer_datos(n=300, seed=0) -> pd.DataFrame:
    """Datos sinteticos con el mismo esquema que California Housing (con nulos en total_bedrooms)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "longitude": rng.uniform(-124, -114, n),
            "latitude": rng.uniform(32.5, 42, n),
            "housing_median_age": rng.integers(1, 53, n).astype(float),
            "total_rooms": rng.integers(100, 8000, n).astype(float),
            "total_bedrooms": rng.integers(50, 1500, n).astype(float),
            "population": rng.integers(100, 4000, n).astype(float),
            "households": rng.integers(50, 1500, n).astype(float),
            "median_income": rng.uniform(0.5, 12, n),
            "ocean_proximity": rng.choice(CATEGORIAS, n),
        }
    )
    df[TARGET] = df["median_income"] * 30000 + rng.normal(0, 20000, n)
    if n >= 100:  # solo los conjuntos de entrenamiento llevan nulos naturales
        df.loc[rng.choice(n, 5, replace=False), "total_bedrooms"] = np.nan
    return df


@pytest.fixture
def csv_entrenamiento(tmp_path):
    ruta = tmp_path / "datos.csv"
    hacer_datos().to_csv(ruta, index=False)
    return ruta


@pytest.fixture
def csv_entrada(tmp_path):
    ruta = tmp_path / "entrada.csv"
    hacer_datos(n=12, seed=1).drop(columns=[TARGET]).to_csv(ruta, index=False)
    return ruta


def entrenar(csv, models_dir, version, extra=None):
    argv = ["--data", str(csv), "--target", TARGET, "--output-dir", str(models_dir),
            "--version", version, "--seed", "42", "--n-estimators", "10", "--cv-folds", "2"]
    return train.main(argv + (extra or []))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# 1. Entrenamiento y almacenamiento
def test_entrenamiento_genera_artefactos_cargables(csv_entrenamiento, tmp_path):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0

    carpeta = models / "v1"
    for nombre in (config.MODEL_FILENAME, config.METRICS_FILENAME, config.METADATA_FILENAME):
        assert (carpeta / nombre).is_file()

    pipeline = joblib.load(carpeta / config.MODEL_FILENAME)
    assert hasattr(pipeline, "predict")

    metrics = pd.read_json(carpeta / config.METRICS_FILENAME, typ="series")
    assert metrics["modelo"]["MAE"] > 0
    assert metrics["modelo_base"]["MAE"] > 0
    # el modelo debe mejorar al base en estos datos (la senal es fuerte)
    assert metrics["modelo"]["MAE"] < metrics["modelo_base"]["MAE"]

    metadata = pd.read_json(carpeta / config.METADATA_FILENAME, typ="series")
    assert metadata["version"] == "v1"
    assert metadata["seed"] == 42
    assert metadata["target"] == TARGET
    assert metadata["input_schema"]["numeric"] == config.NUMERIC_FEATURES
    assert metadata["data"]["n_rows_used"] == 300


# 2. Carga y prediccion
def test_carga_y_prediccion_conserva_resultados_del_modelo(csv_entrenamiento, csv_entrada, tmp_path):
    # Resultado del modelo EN MEMORIA, antes de guardarlo
    df, _ = prepare_training_frame(read_csv(csv_entrenamiento), TARGET)
    res = train_and_evaluate(df, TARGET, "random_forest", seed=42, test_size=0.2,
                             spatial=False, cv_folds=0, n_estimators=10)
    X_nuevo, _ = prepare_prediction_frame(
        read_csv(csv_entrada),
        {"numeric": config.NUMERIC_FEATURES, "categorical": config.CATEGORICAL_FEATURES,
         "known_categories": {"ocean_proximity": CATEGORIAS}},
    )
    esperado = res["pipeline"].predict(X_nuevo)

    # Guardar, recargar y comparar
    ruta = tmp_path / "m.joblib"
    joblib.dump(res["pipeline"], ruta)
    np.testing.assert_allclose(joblib.load(ruta).predict(X_nuevo), esperado, rtol=1e-9)

    # Y por linea de comandos: una prediccion por fila y mismo orden
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0
    salida = tmp_path / "out" / "pred.csv"
    codigo = predict.main(["--model", str(models / "v1" / config.MODEL_FILENAME),
                           "--input", str(csv_entrada), "--output", str(salida)])
    assert codigo == 0
    entrada, pred = pd.read_csv(csv_entrada), pd.read_csv(salida)
    assert len(pred) == len(entrada) == 12
    assert np.isfinite(pred[config.PREDICTION_COLUMN]).all()
    pd.testing.assert_frame_equal(pred.drop(columns=[config.PREDICTION_COLUMN]), entrada)


# 3. Valores faltantes
def test_valores_faltantes_se_imputan_segun_la_estrategia(csv_entrenamiento, tmp_path):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0
    pipeline = joblib.load(models / "v1" / config.MODEL_FILENAME)

    # Mediana calculada de forma INDEPENDIENTE, solo con las filas de entrenamiento
    df, _ = prepare_training_frame(read_csv(csv_entrenamiento), TARGET)
    X_tr, _, _, _ = split_data(df[config.FEATURES], df[TARGET], 0.2, 42, False)
    mediana_train = X_tr["total_bedrooms"].median()
    mediana_todo = df["total_bedrooms"].median()
    assert mediana_train != mediana_todo  # el test distingue "solo train" de "todos los datos"

    num = pipeline.named_steps["prep"].named_transformers_["num"]
    aprendida = dict(zip(config.NUMERIC_FEATURES, num.statistics_))["total_bedrooms"]
    assert aprendida == pytest.approx(mediana_train)  # mediana, y solo con entrenamiento

    base = hacer_datos(n=3, seed=5).drop(columns=[TARGET])
    con_nulo = base.copy()
    con_nulo.loc[0, "total_bedrooms"] = np.nan          # nulo numerico
    con_nulo.loc[1, "ocean_proximity"] = np.nan         # nulo categorico
    entrada = tmp_path / "con_nulos.csv"
    con_nulo.to_csv(entrada, index=False)
    salida = tmp_path / "pred.csv"
    assert predict.main(["--model", str(models / "v1"), "--input", str(entrada),
                         "--output", str(salida)]) == 0
    pred = pd.read_csv(salida)
    assert len(pred) == 3                                # no se elimina ninguna fila
    assert np.isfinite(pred[config.PREDICTION_COLUMN]).all()

    # La fila con nulo se predice igual que la misma fila con la mediana de entrenamiento
    rellena = base.copy()
    rellena.loc[0, "total_bedrooms"] = mediana_train
    esperado = pipeline.predict(rellena[config.FEATURES])[0]
    assert pred.loc[0, config.PREDICTION_COLUMN] == pytest.approx(esperado)


# 4. Entrada invalida
def test_columna_ausente_produce_error_claro_y_codigo_distinto_de_cero(csv_entrenamiento, csv_entrada, tmp_path):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0

    # (a) a nivel de funcion: pytest.raises con mensaje que nombra la columna
    df = read_csv(csv_entrada).drop(columns=["median_income"])
    with pytest.raises(AplicacionError, match="median_income"):
        prepare_prediction_frame(df, {"numeric": config.NUMERIC_FEATURES,
                                      "categorical": config.CATEGORICAL_FEATURES})

    # (b) por linea de comandos real: codigo != 0, mensaje en stderr y SIN archivo de salida
    entrada_mala = tmp_path / "mala.csv"
    df.to_csv(entrada_mala, index=False)
    salida = tmp_path / "no_debe_existir.csv"
    proc = subprocess.run(
        [sys.executable, "predict.py", "--model", str(models / "v1" / config.MODEL_FILENAME),
         "--input", str(entrada_mala), "--output", str(salida)],
        cwd=RAIZ, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "median_income" in proc.stderr
    assert not salida.exists()


# 5. Versionamiento
def test_segunda_version_conserva_la_primera_y_no_se_sobrescribe(csv_entrenamiento, tmp_path):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0
    huella_v1 = {n: sha(models / "v1" / n) for n in
                 (config.MODEL_FILENAME, config.METRICS_FILENAME, config.METADATA_FILENAME)}

    assert entrenar(csv_entrenamiento, models, "v2", extra=["--algorithm", "linear_regression"]) == 0
    for v in ("v1", "v2"):
        assert hasattr(joblib.load(models / v / config.MODEL_FILENAME), "predict")
    assert {n: sha(models / "v1" / n) for n in huella_v1} == huella_v1  # v1 intacta

    # Repetir un identificador NO sobrescribe y falla con codigo distinto de cero
    assert entrenar(csv_entrenamiento, models, "v1") != 0
    assert {n: sha(models / "v1" / n) for n in huella_v1} == huella_v1


# Extras
def test_categoria_no_vista_y_columnas_extra_no_rompen(csv_entrenamiento, tmp_path, capsys):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0
    df = hacer_datos(n=4, seed=9).drop(columns=[TARGET])
    df.loc[0, "ocean_proximity"] = "ISLAND"       # nunca vista en el entrenamiento
    df["columna_extra"] = 1
    df = df[df.columns[::-1]]                     # orden de columnas distinto al del entrenamiento
    entrada = tmp_path / "e.csv"
    df.to_csv(entrada, index=False)
    salida = tmp_path / "p.csv"
    assert predict.main(["--model", str(models / "v1"), "--input", str(entrada),
                         "--output", str(salida)]) == 0
    avisos = capsys.readouterr().err
    assert "ISLAND" in avisos and "columna_extra" in avisos
    assert len(pd.read_csv(salida)) == 4


def test_valor_no_numerico_en_columna_numerica_es_error(csv_entrenamiento, tmp_path):
    models = tmp_path / "models"
    assert entrenar(csv_entrenamiento, models, "v1") == 0
    df = hacer_datos(n=3, seed=2).drop(columns=[TARGET]).astype({"total_rooms": object})
    df.loc[1, "total_rooms"] = "muchas"
    entrada = tmp_path / "e.csv"
    df.to_csv(entrada, index=False)
    salida = tmp_path / "p.csv"
    codigo = predict.main(["--model", str(models / "v1"), "--input", str(entrada), "--output", str(salida)])
    assert codigo != 0 and not salida.exists()


def test_help_funciona_en_ambos_scripts():
    for script in ("train.py", "predict.py"):
        proc = subprocess.run([sys.executable, script, "--help"], cwd=RAIZ,
                              capture_output=True, text=True)
        assert proc.returncode == 0
        assert "--help" in proc.stdout or "usage" in proc.stdout.lower()
