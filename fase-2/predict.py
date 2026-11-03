"""Prediccion: carga una version guardada y predice sobre un CSV nuevo.

No entrena ni reajusta transformaciones: todo viene en el artefacto guardado.

Ejemplo (desde la carpeta fase-2, en una sola linea):
    python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import sklearn

from src import config
from src.data import prepare_prediction_frame, read_csv
from src.errors import AplicacionError
from src.versioning import read_json, resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Genera predicciones con una version guardada del modelo. "
        "La salida conserva el orden y las columnas de la entrada y agrega la columna 'prediccion'.",
    )
    p.add_argument("--model", required=True,
                   help="Ruta de modelo.joblib o del directorio de la version (p. ej. models/v1).")
    p.add_argument("--input", required=True,
                   help="CSV con las variables de entrada (la variable objetivo no es necesaria).")
    p.add_argument("--output", required=True, help="Ruta del CSV de salida.")
    return p


def run_prediction(args: argparse.Namespace) -> dict:
    modelo_path = resolve_model_path(args.model)
    metadata_path = modelo_path.parent / config.METADATA_FILENAME
    if not metadata_path.exists():
        raise AplicacionError(
            f"Falta {config.METADATA_FILENAME} junto al modelo ({metadata_path.parent}); "
            "sin el no se puede validar el esquema de entrada."
        )
    try:
        metadata = read_json(metadata_path)
        schema = metadata["input_schema"]
    except (OSError, ValueError, KeyError) as exc:
        raise AplicacionError(f"metadata.json ilegible o incompleto: {exc}") from exc

    entrada = Path(args.input)
    salida = Path(args.output)
    if entrada.resolve() == salida.resolve():
        raise AplicacionError("--output no puede ser el mismo archivo que --input.")

    df = read_csv(entrada)
    if config.PREDICTION_COLUMN in df.columns:
        raise AplicacionError(
            f"El archivo de entrada ya tiene una columna '{config.PREDICTION_COLUMN}'; renombrala."
        )
    X, advertencias = prepare_prediction_frame(df, schema)

    version_sklearn = metadata.get("environment", {}).get("scikit-learn")
    if version_sklearn and version_sklearn != sklearn.__version__:
        advertencias.append(
            f"El modelo se entreno con scikit-learn {version_sklearn} y se carga con "
            f"{sklearn.__version__}; los resultados podrian diferir."
        )

    try:
        pipeline = joblib.load(modelo_path)
    except Exception as exc:  # artefacto corrupto o incompatible
        raise AplicacionError(f"No se pudo cargar el modelo {modelo_path}: {exc}") from exc

    try:
        predicciones = pipeline.predict(X)
    except Exception as exc:
        raise AplicacionError(f"Fallo al predecir ({type(exc).__name__}): {exc}") from exc

    resultado = df.copy()  # mismo orden y mismas filas que la entrada
    resultado[config.PREDICTION_COLUMN] = predicciones

    salida.parent.mkdir(parents=True, exist_ok=True)
    tmp = salida.with_name(salida.name + ".tmp")
    try:
        resultado.to_csv(tmp, index=False)
        os.replace(tmp, salida)  # el archivo final solo existe si todo salio bien
    finally:
        if tmp.exists():
            tmp.unlink()

    return {"n": len(resultado), "version": metadata.get("version"),
            "salida": salida, "advertencias": advertencias}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        r = run_prediction(args)
    except AplicacionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    for aviso in r["advertencias"]:
        print(f"ADVERTENCIA: {aviso}", file=sys.stderr)
    print(f"Version del modelo : {r['version']}")
    print(f"Predicciones       : {r['n']} fila(s)")
    print(f"Archivo generado   : {r['salida']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
