"""Registro de solicitudes a /predict en un CSV que se conserva entre ejecuciones.

Campos (los unicos que pide la especificacion):
    fecha_hora   hora local del servidor, ISO 8601 sin zona (p. ej. 2026-11-17T14:30:05)
    estado_http  codigo HTTP devuelto (200, 422, 503, ...)
    version      version del modelo que se uso; vacio si no se uso ningun modelo
    prediccion   valor devuelto; vacio si hubo error
"""
from __future__ import annotations

import csv
import os
import sys
import threading
from datetime import datetime
from pathlib import Path

CAMPOS = ["fecha_hora", "estado_http", "version", "prediccion"]
RUTA_POR_DEFECTO = "logs/solicitudes.csv"
_CANDADO = threading.Lock()


def ruta_registro() -> Path:
    """Ruta del archivo de registro (variable de entorno LOG_PATH)."""
    return Path(os.environ.get("LOG_PATH", RUTA_POR_DEFECTO))


def registrar(estado_http: int, version: str | None = None, prediccion: float | None = None,
              ahora: datetime | None = None) -> None:
    """Agrega UNA fila al registro. Abre en modo 'a' (conserva el historial) y escribe el
    encabezado solo cuando el archivo no existe o esta vacio. Un fallo al escribir no tumba
    la API: se avisa por stderr y la respuesta al cliente sigue su curso."""
    ruta = ruta_registro()
    fila = [
        (ahora or datetime.now()).isoformat(timespec="seconds"),
        int(estado_http),
        version or "",
        "" if prediccion is None else prediccion,
    ]
    try:
        with _CANDADO:
            ruta.parent.mkdir(parents=True, exist_ok=True)
            nuevo = not ruta.exists() or ruta.stat().st_size == 0
            with open(ruta, "a", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                if nuevo:
                    escritor.writerow(CAMPOS)
                escritor.writerow(fila)
    except OSError as exc:
        print(f"ADVERTENCIA: no se pudo escribir el registro {ruta}: {exc}", file=sys.stderr)
