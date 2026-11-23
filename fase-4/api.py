"""API REST de la Fase 4: la API de la Fase 3 + registro de cada llamada a /predict.

Cambios respecto a fase-3/api.py (todos marcados con "FASE 4"): el registro de solicitudes
(registro.py, un middleware que anota una fila por intento) y LOG_PATH.

Reutiliza el codigo de la Fase 2 sin modificarlo:
  - train.run_training  -> POST /train
  - src.data.prepare_prediction_frame + modelo.joblib + metadata.json -> POST /predict

Arranque local (desde fase-3/, con el entorno virtual activo):
    python -m uvicorn api:app --host 127.0.0.1 --port 8000
El registro queda en LOG_PATH (por defecto logs/solicitudes.csv).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Annotated, Any

# --- Hace importable el codigo de fase-2 ------------------------------------
# En el repositorio: fase-3/api.py importa de ../fase-2 (src/ y train.py).
# En Docker: src/ y train.py se copian junto a api.py, asi que no hace falta.
_RAIZ = Path(__file__).resolve().parent
_FASE2 = _RAIZ.parent / "fase-2"
if (_FASE2 / "src").is_dir() and str(_FASE2) not in sys.path:
    sys.path.insert(0, str(_FASE2))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import train as entrenamiento  # noqa: E402  (fase-2/train.py)
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

import registro  # noqa: E402  (FASE 4)
from src import config  # noqa: E402
from src.data import prepare_prediction_frame  # noqa: E402
from src.errors import AplicacionError  # noqa: E402
from src.versioning import (  # noqa: E402
    default_version_id,
    ensure_version_free,
    read_json,
    validate_version_id,
)

# Un numero JSON estricto: acepta enteros y decimales; rechaza textos, booleanos, NaN e Infinity.
Numero = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Texto = Annotated[str, Field(strict=True)]

# Unica variable con nulos en los datos de entrenamiento; la unica que admite null.
CAMPOS_CON_NULO = {"total_bedrooms"}


# --- Configuracion (variables de entorno, leidas en cada solicitud) ----------
def _primera_existente(*rutas: Path) -> Path:
    for r in rutas:
        if r.exists():
            return r
    return rutas[-1]


def modelos_dir() -> Path:
    """Directorio con las versiones guardadas (MODELS_DIR). Por defecto fase-2/models."""
    env = os.environ.get("MODELS_DIR")
    return Path(env) if env else _primera_existente(_FASE2 / "models", _RAIZ / "models")


def datos_path() -> Path:
    """CSV con el que /train entrena (DATA_PATH). Por defecto fase-2/data/housing.csv."""
    env = os.environ.get("DATA_PATH")
    return Path(env) if env else _primera_existente(_FASE2 / "data" / "housing.csv",
                                                    _RAIZ / "data" / "housing.csv")


def version_activa() -> str:
    """Version que usa /predict (MODEL_VERSION). Por defecto v1."""
    return os.environ.get("MODEL_VERSION", "v1").strip() or "v1"


def objetivo() -> str:
    return os.environ.get("TARGET", config.DEFAULT_TARGET)


def semilla() -> int:
    return int(os.environ.get("SEED", "42"))


# --- Errores con formato uniforme ---------------------------------------------
class ErrorApi(Exception):
    """Error previsto: se responde con el codigo HTTP y un JSON con mensaje."""

    def __init__(self, status: int, error: str, mensaje: str, **extra: Any):
        super().__init__(mensaje)
        self.status = status
        self.cuerpo = {"error": error, "mensaje": mensaje, **extra}


def versiones_disponibles() -> list[str]:
    base = modelos_dir()
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if (d / config.MODEL_FILENAME).is_file() and (d / config.METADATA_FILENAME).is_file()
    )


COMO_SELECCIONAR = ("Defina la variable de entorno MODEL_VERSION con una version disponible "
                    "y reinicie la API (ver el README, seccion 'Seleccionar la version').")

# --- Carga del modelo (con cache) ---------------------------------------------
_CACHE: dict[tuple[str, int], tuple[Any, dict]] = {}
_CACHE_LOCK = threading.Lock()


def cargar_modelo(version: str) -> tuple[Any, dict]:
    """Devuelve (pipeline, metadata) de una version guardada, o lanza ErrorApi 503."""
    try:
        validate_version_id(version)
    except AplicacionError as exc:
        raise ErrorApi(503, "modelo_no_disponible", f"Version configurada invalida: {exc}",
                       versiones_disponibles=versiones_disponibles(),
                       como_seleccionar=COMO_SELECCIONAR) from exc
    carpeta = modelos_dir() / version
    modelo = carpeta / config.MODEL_FILENAME
    meta = carpeta / config.METADATA_FILENAME
    if not modelo.is_file() or not meta.is_file():
        raise ErrorApi(503, "modelo_no_disponible",
                       f"La version '{version}' no esta disponible en {modelos_dir()}.",
                       version_solicitada=version,
                       versiones_disponibles=versiones_disponibles(),
                       como_seleccionar=COMO_SELECCIONAR)
    clave = (str(modelo.resolve()), modelo.stat().st_mtime_ns)
    with _CACHE_LOCK:
        if clave not in _CACHE:
            try:
                metadata = read_json(meta)
                metadata["input_schema"]
                pipeline = joblib.load(modelo)
            except Exception as exc:  # artefacto corrupto o incompatible
                raise ErrorApi(503, "modelo_no_disponible",
                               f"No se pudo cargar la version '{version}': {exc}",
                               versiones_disponibles=versiones_disponibles(),
                               como_seleccionar=COMO_SELECCIONAR) from exc
            _CACHE.clear()  # solo se conserva la version cargada mas reciente
            _CACHE[clave] = (pipeline, metadata)
        return _CACHE[clave]


# --- Esquemas de entrada -------------------------------------------------------
class RegistroVivienda(BaseModel):
    """Un distrito de California. Las claves son las de fase-2 (config.FEATURES).

    Todas son obligatorias. Solo total_bedrooms acepta null (se imputa con la mediana
    aprendida en el entrenamiento). La variable objetivo no hace falta; si llega, se ignora.
    """
    model_config = ConfigDict(extra="ignore")

    longitude: Numero
    latitude: Numero
    housing_median_age: Numero
    total_rooms: Numero
    total_bedrooms: Numero | None
    population: Numero
    households: Numero
    median_income: Numero
    ocean_proximity: Texto


class SolicitudEntrenamiento(BaseModel):
    """Cuerpo opcional de /train. Un cuerpo vacio usa la configuracion del proyecto."""
    model_config = ConfigDict(extra="ignore")

    version: str | None = None
    seed: int | None = None
    n_estimators: int | None = Field(default=None, ge=1, le=1000)


# --- Aplicacion ----------------------------------------------------------------
app = FastAPI(title="API de prediccion de precios de vivienda (Fase 4)", version="4.0")


# --- FASE 4: una fila en el registro por cada llamada a /predict -------------
@app.middleware("http")
async def registrar_llamada_a_predict(request: Request, call_next):
    """Anota fecha, codigo HTTP, version y prediccion de cada intento sobre /predict.

    Esta por encima de la validacion: tambien registra las entradas rechazadas (422), los
    503 y los errores inesperados, y cada intento genera exactamente una fila.
    """
    if request.url.path != "/predict":
        return await call_next(request)
    try:
        respuesta = await call_next(request)
    except Exception:
        registro.registrar(500, getattr(request.state, "version", None), None)
        raise
    prediccion = getattr(request.state, "prediccion", None) if respuesta.status_code == 200 else None
    registro.registrar(respuesta.status_code, getattr(request.state, "version", None), prediccion)
    return respuesta


@app.exception_handler(ErrorApi)
async def _manejar_error_api(_: Request, exc: ErrorApi) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.cuerpo)


def _describir(err: dict) -> dict:
    """Traduce un error de validacion de pydantic a un mensaje que nombra el campo."""
    loc = [str(p) for p in err.get("loc", ()) if p not in ("body",)]
    campo = ".".join(loc) if loc else "(cuerpo)"
    tipo, entrada = err.get("type", ""), err.get("input")
    if tipo == "missing":
        problema = "campo obligatorio ausente"
    elif tipo == "json_invalid":
        problema, campo = "el cuerpo no es un JSON valido", "(cuerpo)"
    elif tipo in ("model_attributes_type", "dict_type"):
        problema, campo = "el cuerpo debe ser un objeto JSON", "(cuerpo)"
    elif entrada is None and campo.split(".")[-1] not in CAMPOS_CON_NULO:
        problema = "no admite null (solo total_bedrooms admite null)"
    elif tipo.endswith("_type") or tipo.startswith("float") or tipo.startswith("string"):
        esperado = "texto" if tipo.startswith("string") else "numero"
        recibido = type(entrada).__name__
        problema = f"tipo incompatible: se esperaba {esperado} (se recibio {recibido})"
    else:
        problema = err.get("msg", "valor invalido")
    return {"campo": campo, "problema": problema}


@app.exception_handler(RequestValidationError)
async def _manejar_validacion(_: Request, exc: RequestValidationError) -> JSONResponse:
    detalle = [_describir(e) for e in exc.errors()]
    resumen = "; ".join(f"{d['campo']}: {d['problema']}" for d in detalle)
    return JSONResponse(status_code=422, content={
        "error": "entrada_invalida",
        "mensaje": f"La entrada no es valida. {resumen}",
        "detalle": detalle,
    })


@app.get("/health")
def health() -> dict:
    """Indica que la API esta en ejecucion (siempre 200 mientras el proceso responda)."""
    activa = version_activa()
    return {
        "status": "ok",
        "mensaje": "La API esta en ejecucion",
        "version_configurada": activa,
        "modelo_disponible": activa in versiones_disponibles(),
    }


@app.get("/versiones")
def versiones() -> dict:
    """Lista las versiones guardadas y cual esta configurada para predecir."""
    return {"version_activa": version_activa(), "versiones_disponibles": versiones_disponibles()}


@app.post("/predict")
def predict(registro: RegistroVivienda, request: Request) -> dict:
    """Predice el valor mediano de vivienda de un distrito con la version configurada."""
    version = version_activa()
    pipeline, metadata = cargar_modelo(version)
    request.state.version = version  # el modelo existe y se va a usar

    fila = registro.model_dump()
    esquema = metadata["input_schema"]
    for col, conocidas in esquema.get("known_categories", {}).items():
        if fila[col] not in conocidas:
            raise ErrorApi(422, "entrada_invalida",
                           f"La entrada no es valida. {col}: categoria desconocida "
                           f"'{fila[col]}'. Categorias validas: {conocidas}",
                           detalle=[{"campo": col, "problema": "categoria desconocida",
                                     "validas": conocidas}])
    tabla = pd.DataFrame([fila], columns=config.FEATURES)  # una fila, columnas en el orden del esquema
    X, _ = prepare_prediction_frame(tabla, esquema)  # mismo preprocesamiento de fase-2
    try:
        valor = float(pipeline.predict(X)[0])
    except Exception as exc:
        raise ErrorApi(500, "fallo_de_prediccion",
                       f"Fallo al predecir ({type(exc).__name__}): {exc}") from exc

    valor = round(valor, 2)
    request.state.prediccion = valor
    return {"prediccion": valor, "version": metadata.get("version", version)}


_ENTRENANDO = threading.Lock()


def _id_disponible() -> str:
    base = default_version_id()
    candidato, n = base, 1
    while (modelos_dir() / candidato).exists():
        n += 1
        candidato = f"{base}-{n}"
    return candidato


@app.post("/train", status_code=201)
def train(solicitud: SolicitudEntrenamiento | None = None) -> dict:
    """Entrena con los datos y la configuracion de fase-2 y guarda una version NUEVA.

    No sobrescribe versiones anteriores. La version nueva no se usa para predecir hasta
    que se cambie MODEL_VERSION y se reinicie la API.
    """
    solicitud = solicitud or SolicitudEntrenamiento()
    datos = datos_path()
    if not datos.is_file():
        raise ErrorApi(503, "datos_no_disponibles",
                       f"No se encuentra el archivo de datos {datos}. Monte la carpeta de datos "
                       "o defina DATA_PATH (ver el README).")
    version = solicitud.version
    try:
        if version:
            ensure_version_free(modelos_dir(), version)
    except AplicacionError as exc:
        existe = (modelos_dir() / version).exists()
        raise ErrorApi(409 if existe else 422, "version_invalida", str(exc)) from exc

    if not _ENTRENANDO.acquire(blocking=False):
        raise ErrorApi(409, "entrenamiento_en_curso",
                       "Ya hay un entrenamiento en ejecucion; espere a que termine.")
    try:
        version = version or _id_disponible()
        argv = ["--data", str(datos), "--target", objetivo(), "--output-dir", str(modelos_dir()),
                "--version", version,
                "--seed", str(solicitud.seed if solicitud.seed is not None else semilla())]
        if solicitud.n_estimators is not None:
            argv += ["--n-estimators", str(solicitud.n_estimators)]
        try:
            salida = entrenamiento.run_training(entrenamiento.build_parser().parse_args(argv))
        except AplicacionError as exc:
            raise ErrorApi(500, "fallo_de_entrenamiento", str(exc)) from exc
    finally:
        _ENTRENANDO.release()

    m = salida["metrics"]
    return {
        "version": m["version"],
        "ruta": f"{modelos_dir().name}/{m['version']}",
        "metricas_prueba": {"modelo": m["modelo"], "modelo_base": m["modelo_base"]},
        "version_activa": version_activa(),
        "nota": "La version nueva esta guardada pero /predict sigue usando la version activa. "
                "Para usarla, defina MODEL_VERSION y reinicie la API.",
    }
