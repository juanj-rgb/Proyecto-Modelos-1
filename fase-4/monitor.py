"""Metricas de operacion de la API (Fase 4).

Lee el registro que escribe registro.py y muestra dos indicadores:
  - total de solicitudes de prediccion recibidas;
  - porcentaje de solicitudes fallidas (respuestas HTTP 4xx o 5xx).

Si el registro esta vacio o todavia no existe, muestra total 0 e indica
«sin solicitudes» para el porcentaje.

Uso (desde fase-4/, con el entorno virtual activo):
    python monitor.py
    python monitor.py --registro logs/solicitudes.csv
    python monitor.py --registro evidencia/registro_muestra.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
RUTA_POR_DEFECTO = RAIZ / "logs" / "solicitudes.csv"
SIN_MODELO = "(sin modelo)"


def leer_registro(ruta: Path) -> tuple[list[dict], int]:
    """Devuelve (filas validas, numero de filas descartadas).

    Se descarta una fila cuyo estado_http no sea un numero entero: es un renglon
    corrupto y contarlo falsearia los indicadores.
    """
    if not ruta.is_file():
        return [], 0
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    validas, descartadas = [], 0
    for fila in filas:
        try:
            fila["estado_http"] = int(str(fila.get("estado_http", "")).strip())
        except (TypeError, ValueError):
            descartadas += 1
            continue
        validas.append(fila)
    return validas, descartadas


def calcular(filas: list[dict]) -> dict:
    """Total, fallidas y tasa de fallos. La tasa es None cuando no hay solicitudes."""
    total = len(filas)
    fallidas = sum(1 for f in filas if 400 <= f["estado_http"] <= 599)
    return {
        "total": total,
        "fallidas": fallidas,
        "tasa": (fallidas / total * 100) if total else None,
    }


def formato_porcentaje(tasa: float | None) -> str:
    return "sin solicitudes" if tasa is None else f"{tasa:.1f}".replace(".", ",") + " %"


def informe(ruta: Path, filas: list[dict], descartadas: int) -> str:
    r = calcular(filas)
    lineas = [
        f"Registro: {ruta}" + ("" if ruta.is_file() else "  (no existe todavia)"),
        "",
        f"Total de solicitudes : {r['total']}",
        f"Solicitudes fallidas : {r['fallidas']}   (respuestas HTTP 4xx o 5xx)",
        f"Tasa de fallos       : {formato_porcentaje(r['tasa'])}",
    ]
    if r["total"]:
        lineas += ["", "Por codigo HTTP:"]
        for codigo, n in sorted(Counter(f["estado_http"] for f in filas).items()):
            lineas.append(f"  {codigo}  {n}")
        lineas += ["", "Por version utilizada:"]
        versiones = Counter((f.get("version") or "").strip() or SIN_MODELO for f in filas)
        ancho = max(len(v) for v in versiones)
        for version, n in sorted(versiones.items()):
            lineas.append(f"  {version.ljust(ancho)}  {n}")
    if descartadas:
        lineas += ["", f"Aviso: se descartaron {descartadas} fila(s) sin un codigo HTTP valido."]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Indicadores de operacion de la API de prediccion.")
    p.add_argument("--registro", default=str(RUTA_POR_DEFECTO),
                   help="Archivo CSV del registro (por defecto logs/solicitudes.csv).")
    args = p.parse_args(argv)

    ruta = Path(args.registro)
    try:
        filas, descartadas = leer_registro(ruta)
    except OSError as exc:
        print(f"ERROR: no se pudo leer el registro {ruta}: {exc}", file=sys.stderr)
        return 1
    print(informe(ruta, filas, descartadas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
