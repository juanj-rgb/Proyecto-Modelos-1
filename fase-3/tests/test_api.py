"""Pruebas de la API REST de la Fase 3.

Cada prueba usa una carpeta de modelos y un CSV temporales, de modo que no dependen
de los artefactos del repositorio ni dejan rastro. El modelo que usan las pruebas de
prediccion se entrena de verdad a traves de la propia API.
"""
import json
from pathlib import Path

from conftest import REGISTRO_VALIDO


# --- 1. Estado del servicio ---------------------------------------------------
def test_health_responde_siempre_aunque_no_haya_modelo(cliente):
    """/health informa que el proceso vive y si la version configurada esta disponible.

    Debe responder 200 incluso sin ningun modelo guardado: sirve para distinguir
    "la API no esta arriba" de "la API esta arriba pero sin modelo".
    """
    r = cliente.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["version_configurada"] == "v1"
    assert cuerpo["modelo_disponible"] is False      # todavia no se ha entrenado nada


def test_health_reporta_el_modelo_cuando_ya_existe(cliente, modelo_entrenado):
    r = cliente.get("/health")
    assert r.status_code == 200
    assert r.json()["modelo_disponible"] is True


def test_versiones_lista_lo_guardado(cliente, modelo_entrenado):
    r = cliente.get("/versiones")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["version_activa"] == "v1"
    assert "v1" in cuerpo["versiones_disponibles"]


# --- 2. Prediccion ------------------------------------------------------------
def test_predict_devuelve_un_valor_numerico(cliente, modelo_entrenado):
    """Caso feliz: un registro valido produce una prediccion y dice con que version."""
    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 200, r.text
    cuerpo = r.json()
    assert isinstance(cuerpo["prediccion"], (int, float))
    assert cuerpo["prediccion"] > 0
    assert cuerpo["version"] == "v1"


def test_predict_admite_null_solo_en_total_bedrooms(cliente, modelo_entrenado):
    """total_bedrooms es la unica variable con faltantes en los datos, y la unica
    que la API acepta en null: el imputador del pipeline la completa."""
    registro = dict(REGISTRO_VALIDO, total_bedrooms=None)
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 200, r.text
    assert r.json()["prediccion"] > 0

    # En cualquier otra variable, null es entrada invalida.
    registro = dict(REGISTRO_VALIDO, median_income=None)
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422
    assert "median_income" in r.text

    # Enviar null y omitir el campo NO son lo mismo: null es un dato ausente conocido
    # y se imputa; omitirlo es una solicitud incompleta y se rechaza. Sin esta
    # comprobacion, darle un valor por defecto a total_bedrooms pasaria inadvertido
    # y la API predeciria sobre un dato inventado.
    registro = {k: v for k, v in REGISTRO_VALIDO.items() if k != "total_bedrooms"}
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422, r.text
    assert "total_bedrooms" in r.text


def test_predict_ignora_campos_de_mas(cliente, modelo_entrenado):
    """Un cliente puede mandar la variable objetivo o columnas extra: se ignoran
    en vez de romper, porque el esquema del modelo manda sobre la entrada."""
    registro = dict(REGISTRO_VALIDO, median_house_value=500_001.0, columna_inventada="x")
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 200, r.text
    assert r.json()["prediccion"] > 0


# --- 3. Entradas invalidas ----------------------------------------------------
def test_predict_sin_un_campo_obligatorio_responde_422_y_lo_nombra(cliente, modelo_entrenado):
    """El mensaje tiene que decir QUE campo falta: un 422 sin detalle no sirve
    para que quien consume la API corrija su solicitud."""
    registro = {k: v for k, v in REGISTRO_VALIDO.items() if k != "median_income"}
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422
    cuerpo = r.json()
    assert cuerpo["error"] == "entrada_invalida"
    campos = [d["campo"] for d in cuerpo["detalle"]]
    assert "median_income" in campos
    assert "median_income" in cuerpo["mensaje"]


def test_predict_con_tipo_incompatible_responde_422(cliente, modelo_entrenado):
    """Un texto donde se espera un numero no debe llegar al modelo."""
    registro = dict(REGISTRO_VALIDO, median_income="mucho")
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422
    cuerpo = r.json()
    assert cuerpo["error"] == "entrada_invalida"
    assert any(d["campo"] == "median_income" for d in cuerpo["detalle"])


def test_predict_rechaza_cualquier_valor_que_no_sea_un_numero_finito(cliente, modelo_entrenado):
    """Prueba agregada en la revision del PR.

    La prueba anterior cubre un caso puntual: un texto en median_income. Esta comprueba
    que la regla vale para otro campo (longitude) y para seis formas distintas de valor
    invalido, dos de las cuales no vigilaba ninguna prueba:

      - booleano: en JSON, true no es 1. El esquema esta declarado estricto (Field
        strict=True) justamente para que no se convierta solo.
      - NaN e Infinity: JSON estricto no los admite, pero varios clientes los emiten.
        Si uno pasara, el modelo predeciria sobre un valor sin sentido en silencio,
        que es el peor desenlace posible: respuesta 200 con un numero inventado.

    Los seis casos se envian como cuerpo crudo, no con json=, porque NaN e Infinity
    no se pueden construir desde un diccionario de Python sin que el cliente los
    rechace antes de salir.
    """
    def cuerpo_con_longitude(literal: str) -> str:
        partes = [f'"{clave}": ' + (literal if clave == "longitude" else json.dumps(valor))
                  for clave, valor in REGISTRO_VALIDO.items()]
        return "{" + ", ".join(partes) + "}"

    casos = {
        "texto": '"abc"',
        "booleano": "true",
        "NaN": "NaN",
        "Infinity": "Infinity",
        "lista": "[1, 2]",
        "nulo": "null",
    }
    for etiqueta, literal in casos.items():
        r = cliente.post("/predict", content=cuerpo_con_longitude(literal),
                         headers={"Content-Type": "application/json"})
        assert r.status_code == 422, f"{etiqueta}: se esperaba 422, llego {r.status_code}: {r.text}"
        cuerpo = r.json()
        assert cuerpo["error"] == "entrada_invalida", f"{etiqueta}: {cuerpo}"
        campos = [d["campo"] for d in cuerpo["detalle"]]
        assert "longitude" in campos, f"{etiqueta}: el error no nombra el campo: {cuerpo}"
        assert "longitude" in cuerpo["mensaje"], f"{etiqueta}: {cuerpo['mensaje']}"


def test_predict_con_categoria_desconocida_responde_422_y_lista_las_validas(
        cliente, modelo_entrenado):
    """El codificador se ajusto con las categorias del entrenamiento. Una categoria
    nueva se rechaza con la lista de las validas, en vez de predecir en silencio
    sobre un vector de ceros."""
    registro = dict(REGISTRO_VALIDO, ocean_proximity="MARTE")
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 422
    cuerpo = r.json()
    assert "ocean_proximity" in cuerpo["mensaje"]
    assert cuerpo["detalle"][0]["validas"]        # viene la lista de categorias conocidas


def test_predict_sin_modelo_responde_503_y_explica_como_resolverlo(cliente):
    """Sin ninguna version guardada, /predict no puede trabajar. Debe responder 503
    (el servicio no esta listo), no 500, y decir que versiones hay y como elegirlas."""
    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 503
    cuerpo = r.json()
    assert cuerpo["error"] == "modelo_no_disponible"
    assert cuerpo["versiones_disponibles"] == []
    assert "MODEL_VERSION" in cuerpo["como_seleccionar"]


# --- 4. Entrenamiento ---------------------------------------------------------
def test_train_crea_una_version_nueva_con_sus_artefactos(cliente, entorno):
    """/train guarda modelo, metricas y metadatos, y responde 201 con las metricas
    del modelo y del modelo base."""
    r = cliente.post("/train", json={"version": "v_nueva", "n_estimators": 5})
    assert r.status_code == 201, r.text
    cuerpo = r.json()
    assert cuerpo["version"] == "v_nueva"
    assert "modelo" in cuerpo["metricas_prueba"]
    assert "modelo_base" in cuerpo["metricas_prueba"]

    carpeta = entorno["modelos"] / "v_nueva"
    assert (carpeta / "modelo.joblib").is_file()
    assert (carpeta / "metrics.json").is_file()
    assert (carpeta / "metadata.json").is_file()


def test_train_no_sobrescribe_una_version_existente(cliente, entorno, modelo_entrenado):
    """Regla heredada de la Fase 2: una version guardada es inmutable. Un segundo
    entrenamiento con el mismo nombre falla y deja intacto el modelo anterior."""
    modelo = entorno["modelos"] / "v1" / "modelo.joblib"
    antes = modelo.stat().st_mtime_ns, modelo.stat().st_size

    r = cliente.post("/train", json={"version": "v1", "n_estimators": 5})
    assert r.status_code == 409
    assert r.json()["error"] == "version_invalida"

    assert (modelo.stat().st_mtime_ns, modelo.stat().st_size) == antes


def test_train_sin_datos_responde_503(cliente, entorno, monkeypatch, tmp_path):
    """Si el CSV no esta montado, /train avisa que faltan los datos en vez de reventar."""
    monkeypatch.setenv("DATA_PATH", str(tmp_path / "no_existe.csv"))
    r = cliente.post("/train", json={"version": "v_sin_datos"})
    assert r.status_code == 503
    assert r.json()["error"] == "datos_no_disponibles"


def test_train_y_predict_usan_el_mismo_esquema(cliente, monkeypatch):
    """Lo que entrena /train debe poder consumirlo /predict: se entrena una version,
    se apunta la API a ella y se predice con el ejemplo que acompana al proyecto.

    Es la prueba de que el ciclo completo cierra sobre un mismo esquema de entrada.
    """
    import api

    r = cliente.post("/train", json={"version": "v_ciclo", "n_estimators": 5})
    assert r.status_code == 201, r.text

    monkeypatch.setenv("MODEL_VERSION", "v_ciclo")   # monkeypatch lo revierte al terminar
    api._CACHE.clear()

    r = cliente.post("/predict", json=REGISTRO_VALIDO)
    assert r.status_code == 200, r.text
    assert r.json()["version"] == "v_ciclo"


# --- 5. Coherencia con los archivos que acompanan a la API --------------------
def test_el_ejemplo_de_solicitud_es_valido(cliente, modelo_entrenado):
    """ejemplo_solicitud.json se entrega como referencia para quien use la API:
    tiene que funcionar tal cual, sin editarlo."""
    ruta = Path(__file__).resolve().parent.parent / "ejemplo_solicitud.json"
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    r = cliente.post("/predict", json=registro)
    assert r.status_code == 200, r.text
    assert r.json()["prediccion"] > 0
