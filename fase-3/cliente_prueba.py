"""Cliente de prueba sencillo: consulta /health y envia una prediccion a la API.

Solo usa la libreria estandar (no necesita instalar nada).

Ejemplos:
    python cliente_prueba.py
    python cliente_prueba.py --url http://localhost:8000 --archivo ejemplo_solicitud.json
    python cliente_prueba.py --csv ../fase-2/data/ejemplo_prediccion.csv --fila 2
    python cliente_prueba.py --quitar median_income        # prueba de validacion: debe dar 422
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

CATEGORICAS = {"ocean_proximity"}


def pedir(url, datos=None):
    """Hace GET (sin datos) o POST (con JSON). Devuelve (codigo_http, cuerpo_json)."""
    cuerpo = None if datos is None else json.dumps(datos).encode("utf-8")
    req = urllib.request.Request(url, data=cuerpo, method="POST" if datos is not None else "GET",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        texto = e.read().decode("utf-8")
        try:
            return e.code, json.loads(texto)
        except ValueError:
            return e.code, {"respuesta": texto}


def fila_csv_a_json(ruta, fila):
    """Convierte una fila (1 = primera) de un CSV en el JSON de la solicitud: los numeros se
    envian como numeros, la categoria como texto y una celda vacia como null."""
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    if not 1 <= fila <= len(filas):
        sys.exit(f"ERROR: --fila debe estar entre 1 y {len(filas)}.")
    registro = {}
    for clave, valor in filas[fila - 1].items():
        if clave in CATEGORICAS:
            registro[clave] = valor
        else:
            registro[clave] = float(valor) if valor.strip() else None
    return registro


def main():
    p = argparse.ArgumentParser(description="Cliente de prueba de la API de prediccion.")
    p.add_argument("--url", default="http://localhost:8000", help="URL base de la API.")
    p.add_argument("--archivo", default=str(Path(__file__).with_name("ejemplo_solicitud.json")),
                   help="JSON con un registro (por defecto ejemplo_solicitud.json).")
    p.add_argument("--csv", help="En vez del JSON, usar una fila de este CSV (p. ej. el de fase-2).")
    p.add_argument("--fila", type=int, default=1, help="Fila del CSV (1 = primera). Con --csv.")
    p.add_argument("--quitar", help="Nombre de una variable a quitar del registro (prueba de validacion).")
    args = p.parse_args()

    if args.csv:
        registro = fila_csv_a_json(args.csv, args.fila)
    else:
        registro = json.loads(Path(args.archivo).read_text(encoding="utf-8"))
    if args.quitar:
        registro.pop(args.quitar, None)

    try:
        codigo, cuerpo = pedir(args.url.rstrip("/") + "/health")
        print(f"GET  /health  -> {codigo} {json.dumps(cuerpo, ensure_ascii=False)}")
        print("POST /predict  solicitud:", json.dumps(registro, ensure_ascii=False))
        codigo, cuerpo = pedir(args.url.rstrip("/") + "/predict", registro)
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: no se pudo conectar con {args.url} ({e.reason}). Esta la API en ejecucion?")
    print(f"POST /predict  -> {codigo} {json.dumps(cuerpo, ensure_ascii=False)}")
    return 0 if codigo == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
