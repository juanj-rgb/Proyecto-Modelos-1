"""Registro de actividad de la API (Fase 4).

Toma la aplicacion de la Fase 3 sin modificarla y le agrega un middleware que anota
una fila por cada llamada a POST /predict --incluidas las rechazadas por validacion--
en un CSV que se conserva entre ejecuciones.

Columnas del registro: fecha_hora, estado_http, version, prediccion.
En un rechazo la prediccion queda vacia y, si no se llego a usar un modelo, la version
tambien. El archivo se abre en modo de agregar y el encabezado se escribe al crearlo.

Arranque (desde fase-4/, con el entorno virtual activo):
    python -m uvicorn registro:app --host 127.0.0.1 --port 8000

Ubicacion del registro: fase-4/logs/solicitudes.csv, o la ruta de la variable de
entorno LOG_PATH (en Docker, una carpeta montada).
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

# --- Hace importable la API de la Fase 3 -------------------------------------
# En el repositorio: fase-4/registro.py importa ../fase-3/api.py.
# En Docker: api.py se copia junto a este archivo, asi que se encuentra primero.
_RAIZ = Path(__file__).resolve().parent
for _candidata in (_RAIZ, _RAIZ.parent / "fase-3"):
    if (_candidata / "api.py").is_file():
        if str(_candidata) not in sys.path:
            sys.path.insert(0, str(_candidata))
        break

import api  # noqa: E402  (fase-3/api.py)

app = api.app  # la misma aplicacion de la Fase 3, con el registro activado abajo

COLUMNAS = ("fecha_hora", "estado_http", "version", "prediccion")
RUTA_POR_DEFECTO = _RAIZ / "logs" / "solicitudes.csv"
RUTA_VIGILADA = "/predict"

_ESCRITURA = Lock()  # una sola fila a la vez aunque haya solicitudes simultaneas


def ruta_registro() -> Path:
    """Archivo donde se anotan las solicitudes (LOG_PATH). Por defecto fase-4/logs/solicitudes.csv."""
    env = os.environ.get("LOG_PATH")
    return Path(env) if env else RUTA_POR_DEFECTO


def anotar(estado_http: int, version: Any = None, prediccion: Any = None,
           ruta: Path | str | None = None) -> dict:
    """Agrega una fila al registro y la devuelve.

    Crea la carpeta y el encabezado si el archivo no existe todavia. Un valor ausente
    (version o prediccion) se escribe como celda vacia.
    """
    fila = {
        "fecha_hora": datetime.now().isoformat(timespec="seconds"),
        "estado_http": int(estado_http),
        "version": "" if version is None else version,
        "prediccion": "" if prediccion is None else prediccion,
    }
    destino = Path(ruta) if ruta is not None else ruta_registro()
    with _ESCRITURA:
        destino.parent.mkdir(parents=True, exist_ok=True)
        nuevo = not destino.exists() or destino.stat().st_size == 0
        with open(destino, "a", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=COLUMNAS)
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)
    return fila


def _es_prediccion(request) -> bool:
    return request.method == "POST" and request.url.path.rstrip("/") == RUTA_VIGILADA


@app.middleware("http")
async def registrar_solicitudes(request, call_next):
    """Anota una sola fila por intento de prediccion, sea cual sea el resultado.

    El middleware envuelve a los manejadores de error de la API, de modo que tambien ve
    las respuestas 422 que produce la validacion antes de llamar a la funcion de prediccion.
    api.py deja en request.state la version y la prediccion cuando llega a usarlas; si la
    solicitud se rechazo antes, esos atributos no existen y las celdas quedan vacias.
    """
    vigilada = _es_prediccion(request)
    try:
        respuesta = await call_next(request)
    except Exception:
        if vigilada:
            anotar(500)  # fallo no previsto: el servidor respondera 500
        raise
    if vigilada:
        estado = request.state
        anotar(respuesta.status_code,
               getattr(estado, "version", None),
               getattr(estado, "prediccion", None))
    return respuesta
