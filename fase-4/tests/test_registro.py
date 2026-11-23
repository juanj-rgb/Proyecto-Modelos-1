"""Pruebas del registro de /predict y del script de monitoreo (Fase 4)."""
import copy
import csv

import monitor
import registro
from conftest import REGISTRO_VALIDO, VERSION_PRUEBA


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def enviar_ensayo(cliente):
    """Dos entradas validas y una a la que le falta una variable (el ensayo de la especificacion)."""
    incompleta = copy.deepcopy(REGISTRO_VALIDO)
    del incompleta["median_income"]
    return [cliente.post("/predict", json=REGISTRO_VALIDO).status_code,
            cliente.post("/predict", json={**REGISTRO_VALIDO, "median_income": 3.5}).status_code,
            cliente.post("/predict", json=incompleta).status_code]


def test_cada_intento_genera_una_sola_fila_incluido_el_rechazado(cliente, ruta_log):
    assert enviar_ensayo(cliente) == [200, 200, 422]
    filas = leer(ruta_log)
    assert [f["estado_http"] for f in filas] == ["200", "200", "422"]
    assert [f["version"] for f in filas] == [VERSION_PRUEBA, VERSION_PRUEBA, ""]
    assert filas[0]["prediccion"] != "" and filas[1]["prediccion"] != ""
    assert filas[2]["prediccion"] == ""  # el error deja la prediccion vacia
    assert filas[0]["fecha_hora"].count(":") == 2  # fecha y hora ISO


def test_la_prediccion_registrada_es_la_que_devolvio_la_api(cliente, ruta_log):
    respuesta = cliente.post("/predict", json=REGISTRO_VALIDO).json()
    fila = leer(ruta_log)[0]
    assert float(fila["prediccion"]) == respuesta["prediccion"]
    assert fila["version"] == respuesta["version"]


def test_encabezado_una_sola_vez_y_el_historial_se_conserva_entre_ejecuciones(cliente, ruta_log):
    from fastapi.testclient import TestClient
    import api
    enviar_ensayo(cliente)
    # "reinicio": un cliente nuevo sobre el mismo archivo agrega filas, no lo reescribe
    TestClient(api.app).post("/predict", json=REGISTRO_VALIDO)
    lineas = ruta_log.read_text(encoding="utf-8").splitlines()
    assert lineas[0] == "fecha_hora,estado_http,version,prediccion"
    assert lineas.count(lineas[0]) == 1
    assert len(lineas) == 1 + 4


def test_503_y_categoria_desconocida_tambien_se_registran(cliente, ruta_log, monkeypatch):
    # categoria desconocida: el modelo si se uso -> version registrada, sin prediccion
    assert cliente.post("/predict", json={**REGISTRO_VALIDO, "ocean_proximity": "MARTE"}).status_code == 422
    # modelo no disponible: no se uso ningun modelo -> version vacia
    monkeypatch.setenv("MODEL_VERSION", "v-inexistente")
    assert cliente.post("/predict", json=REGISTRO_VALIDO).status_code == 503
    filas = leer(ruta_log)
    assert [(f["estado_http"], f["version"], f["prediccion"]) for f in filas] == [
        ("422", VERSION_PRUEBA, ""), ("503", "", "")]


def test_las_otras_rutas_no_se_registran(cliente, ruta_log):
    cliente.get("/health")
    cliente.get("/versiones")
    assert not ruta_log.exists()


def test_monitor_muestra_3_solicitudes_y_33_3_por_ciento(cliente, ruta_log, capsys):
    enviar_ensayo(cliente)
    assert monitor.main(["--log", str(ruta_log)]) == 0
    salida = capsys.readouterr().out
    assert "Total de solicitudes    : 3" in salida
    assert "Porcentaje de fallos    : 33,3 %" in salida
    r = monitor.calcular(ruta_log)
    assert (r["total"], r["fallidas"]) == (3, 1)


def test_monitor_con_registro_vacio_indica_sin_solicitudes(tmp_path, capsys):
    vacio = tmp_path / "vacio.csv"
    vacio.write_text("fecha_hora,estado_http,version,prediccion\n", encoding="utf-8")
    for ruta in (vacio, tmp_path / "no_existe.csv"):
        assert monitor.main(["--log", str(ruta)]) == 0
        salida = capsys.readouterr().out
        assert "Total de solicitudes    : 0" in salida and "sin solicitudes" in salida


def test_monitor_cuenta_como_fallidos_los_4xx_y_5xx(tmp_path, monkeypatch):
    ruta = tmp_path / "r.csv"
    monkeypatch.setenv("LOG_PATH", str(ruta))
    for codigo in (200, 201, 399, 400, 422, 503, 599):
        registro.registrar(codigo)
    r = monitor.calcular(ruta)
    assert (r["total"], r["fallidas"]) == (7, 4)
    assert monitor.formatear_porcentaje(r["porcentaje"]) == "57,1 %"
