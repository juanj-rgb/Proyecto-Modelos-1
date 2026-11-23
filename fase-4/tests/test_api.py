"""Pruebas de la API (Fase 3). Ejecutar desde fase-3:  python -m pytest

Las tres primeras son las exigidas por la especificacion; el resto cubre el comportamiento
descrito en el README (nulos, tipos, categorias, 503 y /train).
"""
import copy
import shutil

import joblib
import pandas as pd

from conftest import REGISTRO_VALIDO, VERSION_PRUEBA


# --- Las tres pruebas exigidas -------------------------------------------------
def test_health_responde_200(cliente):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_entrada_valida_devuelve_prediccion_y_version(cliente):
    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"prediccion", "version"}
    assert isinstance(cuerpo["prediccion"], float)
    assert cuerpo["version"] == VERSION_PRUEBA


def test_entrada_con_campo_obligatorio_ausente_se_rechaza(cliente):
    registro = copy.deepcopy(REGISTRO_VALIDO)
    del registro["median_income"]
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422
    assert "median_income" in r.json()["mensaje"]  # el mensaje nombra la variable que falta


# --- Comportamiento adicional ------------------------------------------------
def test_la_api_da_el_mismo_valor_que_el_modelo_guardado(cliente, entorno_modelo):
    esperado = joblib.load(entorno_modelo["modelos"] / VERSION_PRUEBA / "modelo.joblib").predict(
        pd.DataFrame([REGISTRO_VALIDO]))[0]
    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert abs(r.json()["prediccion"] - esperado) < 0.01


def test_la_variable_objetivo_no_es_necesaria_y_si_llega_se_ignora(cliente):
    base = cliente.post("/predict", json=REGISTRO_VALIDO).json()["prediccion"]
    con_objetivo = {**REGISTRO_VALIDO, "median_house_value": 999999.0}
    r = cliente.post("/predict", json=con_objetivo)
    assert r.status_code == 200 and r.json()["prediccion"] == base


def test_total_bedrooms_acepta_null_y_se_imputa(cliente):
    r = cliente.post("/predict", json={**REGISTRO_VALIDO, "total_bedrooms": None})
    assert r.status_code == 200


def test_otro_campo_numerico_no_admite_null(cliente):
    r = cliente.post("/predict", json={**REGISTRO_VALIDO, "median_income": None})
    assert r.status_code == 422
    assert "median_income" in r.json()["mensaje"] and "null" in r.json()["mensaje"]


def test_tipo_incompatible_se_rechaza_con_el_nombre_del_campo(cliente):
    for malo in ("24.5", True, [1], {"a": 1}):
        r = cliente.post("/predict", json={**REGISTRO_VALIDO, "latitude": malo})
        assert r.status_code == 422, malo
        assert "latitude" in r.json()["mensaje"], malo
    r = cliente.post("/predict", json={**REGISTRO_VALIDO, "ocean_proximity": 5})
    assert r.status_code == 422 and "ocean_proximity" in r.json()["mensaje"]


def test_categoria_desconocida_se_rechaza(cliente):
    r = cliente.post("/predict", json={**REGISTRO_VALIDO, "ocean_proximity": "NEAR_BAY"})
    assert r.status_code == 422
    assert "ocean_proximity" in r.json()["mensaje"] and "NEAR BAY" in r.json()["mensaje"]


def test_cuerpo_que_no_es_json_o_no_es_objeto_se_rechaza(cliente):
    r = cliente.post("/predict", content="esto no es json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    r = cliente.post("/predict", json=[1, 2, 3])
    assert r.status_code == 422


def test_modelo_no_disponible_devuelve_503_y_explica_como_elegir_version(cliente, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v-inexistente")
    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 503
    cuerpo = r.json()
    assert VERSION_PRUEBA in cuerpo["versiones_disponibles"]
    assert "MODEL_VERSION" in cuerpo["como_seleccionar"]
    assert cliente.get("/health").json()["modelo_disponible"] is False  # /health sigue en 200


def test_train_guarda_version_nueva_sin_tocar_las_anteriores(cliente, entorno_modelo):
    modelos = entorno_modelo["modelos"]
    antes = (modelos / VERSION_PRUEBA / "modelo.joblib").read_bytes()
    r = cliente.post("/train", json={"version": "v-nueva", "n_estimators": 5})
    try:
        assert r.status_code == 201
        assert r.json()["version"] == "v-nueva"
        for nombre in ("modelo.joblib", "metrics.json", "metadata.json"):
            assert (modelos / "v-nueva" / nombre).is_file()
        assert (modelos / VERSION_PRUEBA / "modelo.joblib").read_bytes() == antes
        # repetir la misma version no sobrescribe: 409
        assert cliente.post("/train", json={"version": "v-nueva"}).status_code == 409
        # /predict sigue con la version configurada hasta que se cambie MODEL_VERSION
        assert cliente.post("/predict", json=REGISTRO_VALIDO).json()["version"] == VERSION_PRUEBA
    finally:
        shutil.rmtree(modelos / "v-nueva", ignore_errors=True)


def test_train_con_cuerpo_vacio_genera_un_identificador_disponible(cliente, entorno_modelo):
    modelos = entorno_modelo["modelos"]
    r = cliente.post("/train")
    nueva = r.json().get("version")
    try:
        assert r.status_code == 201 and nueva and nueva != VERSION_PRUEBA
        assert (modelos / nueva / "modelo.joblib").is_file()
        assert nueva in cliente.get("/versiones").json()["versiones_disponibles"]
    finally:
        if nueva:
            shutil.rmtree(modelos / nueva, ignore_errors=True)


def test_train_sin_datos_devuelve_503(cliente, monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "no_existe.csv"))
    r = cliente.post("/train")
    assert r.status_code == 503 and "DATA_PATH" in r.json()["mensaje"]


def test_train_rechaza_identificador_invalido(cliente):
    r = cliente.post("/train", json={"version": "../x"})
    assert r.status_code == 422
