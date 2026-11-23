"""Metricas de operacion a partir del registro de la API.

Muestra el total de solicitudes y el porcentaje de solicitudes fallidas (HTTP 4xx o 5xx).
No mide la calidad de las predicciones: eso exigiria conocer los resultados reales.

Uso (desde fase-4/):
    python monitor.py
    python monitor.py --log logs/solicitudes.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

RUTA_POR_DEFECTO = "logs/solicitudes.csv"


def calcular(ruta) -> dict:
    """Lee el registro y devuelve total, fallidas, porcentaje (None si no hay solicitudes),
    solicitudes por version y filas descartadas por tener un codigo ilegible."""
    ruta = Path(ruta)
    total = fallidas = ilegibles = 0
    por_version: Counter = Counter()
    existe = ruta.is_file()
    if existe:
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f):
                try:
                    codigo = int(str(fila.get("estado_http", "")).strip())
                except ValueError:
                    ilegibles += 1
                    continue
                total += 1
                if 400 <= codigo <= 599:
                    fallidas += 1
                if fila.get("version"):
                    por_version[fila["version"]] += 1
    return {
        "existe": existe,
        "total": total,
        "fallidas": fallidas,
        "porcentaje": (fallidas / total * 100) if total else None,
        "por_version": dict(por_version),
        "ilegibles": ilegibles,
    }


def formatear_porcentaje(valor) -> str:
    if valor is None:
        return "sin solicitudes"
    return f"{valor:.1f}".replace(".", ",") + " %"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Total de solicitudes y porcentaje de fallos de la API.")
    p.add_argument("--log", default=os.environ.get("LOG_PATH", RUTA_POR_DEFECTO),
                   help=f"Archivo de registro (por defecto LOG_PATH o {RUTA_POR_DEFECTO}).")
    args = p.parse_args(argv)

    r = calcular(args.log)
    print(f"Registro                : {args.log}")
    if not r["existe"]:
        print("AVISO: el archivo no existe todavia; se trata como un registro vacio.", file=sys.stderr)
    if r["ilegibles"]:
        print(f"AVISO: {r['ilegibles']} fila(s) con codigo HTTP ilegible no se cuentan.", file=sys.stderr)
    print(f"Total de solicitudes    : {r['total']}")
    print(f"Solicitudes fallidas    : {r['fallidas']} (HTTP 4xx o 5xx)")
    print(f"Porcentaje de fallos    : {formatear_porcentaje(r['porcentaje'])}")
    if r["por_version"]:
        usadas = ", ".join(f"{v}: {n}" for v, n in sorted(r["por_version"].items()))
        print(f"Solicitudes por version : {usadas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
