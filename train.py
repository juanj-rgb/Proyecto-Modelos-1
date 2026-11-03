"""Entrenamiento: construye y guarda una version reutilizable del modelo.

Ejemplo (desde la carpeta fase-2, en una sola linea):
    python train.py --data data/housing.csv --target median_house_value --output-dir models --version v1 --seed 42
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from src import config
from src.data import file_sha256, prepare_training_frame, read_csv
from src.errors import AplicacionError
from src.modeling import estimator_params, train_and_evaluate
from src.versioning import (
    code_reference,
    create_version_dir,
    default_version_id,
    ensure_version_free,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train.py",
        description="Entrena un modelo de regresion y guarda una version en <output-dir>/<version>/ "
        "(modelo.joblib, metrics.json, metadata.json).",
    )
    p.add_argument("--data", required=True, help="Ruta del CSV con datos etiquetados.")
    p.add_argument("--target", default=config.DEFAULT_TARGET,
                   help=f"Nombre de la variable objetivo (por defecto: {config.DEFAULT_TARGET}).")
    p.add_argument("--output-dir", default="models",
                   help="Directorio donde se crean las versiones (por defecto: models).")
    p.add_argument("--version", default=None,
                   help="Identificador de la version (p. ej. v1). Si se omite se genera uno con fecha y hora. "
                   "Si ya existe, el script falla y NO sobrescribe.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria (por defecto: 42).")
    p.add_argument("--test-size", type=float, default=0.2,
                   help="Fraccion de datos para prueba (por defecto: 0.2).")
    p.add_argument("--spatial-split", action="store_true",
                   help="Separa por zonas geograficas (celdas de 1 grado) en vez de aleatoriamente.")
    p.add_argument("--algorithm", choices=config.ALGORITHMS, default="random_forest",
                   help="Algoritmo (por defecto: random_forest).")
    p.add_argument("--n-estimators", type=int, default=100,
                   help="Numero de arboles de random_forest (por defecto: 100).")
    p.add_argument("--max-depth", type=int, default=None,
                   help="Profundidad maxima (random_forest / decision_tree).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Folds de validacion cruzada sobre entrenamiento; 0 la desactiva (por defecto: 5).")
    return p


def run_training(args: argparse.Namespace) -> dict:
    version = args.version or default_version_id()
    ensure_version_free(args.output_dir, version)  # falla antes de entrenar si ya existe

    df_raw = read_csv(args.data)
    df, descartadas = prepare_training_frame(df_raw, args.target)

    resultado = train_and_evaluate(
        df=df,
        target=args.target,
        algorithm=args.algorithm,
        seed=args.seed,
        test_size=args.test_size,
        spatial=args.spatial_split,
        cv_folds=args.cv_folds,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    X_train = resultado["X_train"]
    mae_m = resultado["metricas_modelo"]["MAE"]
    mae_b = resultado["metricas_base"]["MAE"]

    metrics = {
        "version": version,
        "metrica_principal": "MAE",
        "modelo": resultado["metricas_modelo"],
        "modelo_base": {"descripcion": "DummyRegressor (media del entrenamiento)",
                        **resultado["metricas_base"]},
        "mejora_MAE_vs_base_pct": round((mae_b - mae_m) / mae_b * 100, 2),
        "validacion_cruzada_entrenamiento": resultado["validacion_cruzada"],
        "n_entrenamiento": resultado["n_train"],
        "n_prueba": resultado["n_test"],
        "nota": "La prueba se usa una sola vez, para evaluar; no interviene en el ajuste "
                "ni en la seleccion.",
    }

    metadata = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "problem_type": "regression",
        "target": args.target,
        "algorithm": args.algorithm,
        "model_class": type(resultado["estimator"]).__name__,
        "model_params": estimator_params(resultado["estimator"]),
        "seed": args.seed,
        "input_schema": {
            "numeric": config.NUMERIC_FEATURES,
            "categorical": config.CATEGORICAL_FEATURES,
            "dtypes": {c: str(df[c].dtype) for c in config.FEATURES},
            "known_categories": {
                c: sorted(X_train[c].dropna().unique().tolist())
                for c in config.CATEGORICAL_FEATURES
            },
        },
        "missing_values_strategy": {
            "numeric": "mediana aprendida solo con el entrenamiento",
            "categorical": "moda aprendida solo con el entrenamiento; categorias no vistas "
                           "se codifican con todas las columnas one-hot en cero",
        },
        "data": {
            "file": Path(args.data).name,
            "sha256": file_sha256(args.data),
            "n_rows_read": int(len(df_raw)),
            "n_rows_used": int(len(df)),
            "n_rows_dropped_missing_target": int(descartadas),
        },
        "split": {
            "strategy": "por zonas (GroupShuffleSplit, celdas de 1 grado)"
                        if args.spatial_split else "aleatoria (train_test_split)",
            "test_size": args.test_size,
            "n_train": resultado["n_train"],
            "n_test": resultado["n_test"],
        },
        "evaluation_metric": "MAE (principal); tambien RMSE y R2",
        "environment": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
            "requirements_file": "requirements.txt",
        },
        "code_reference": code_reference(),
        "reproducibility_notes": "Con los mismos datos, parametros, semilla y versiones de "
                                 "librerias se esperan los mismos resultados.",
    }

    version_dir = create_version_dir(args.output_dir, version)
    try:
        joblib.dump(resultado["pipeline"], version_dir / config.MODEL_FILENAME, compress=3)
        write_json(version_dir / config.METRICS_FILENAME, metrics)
        write_json(version_dir / config.METADATA_FILENAME, metadata)
    except Exception:
        shutil.rmtree(version_dir, ignore_errors=True)  # no dejar versiones a medias
        raise
    return {"version_dir": version_dir, "metrics": metrics, "metadata": metadata}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        salida = run_training(args)
    except AplicacionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    m, base = salida["metrics"]["modelo"], salida["metrics"]["modelo_base"]
    vd = salida["version_dir"]
    print(f"Version entrenada: {salida['metrics']['version']}")
    print(f"  Algoritmo        : {args.algorithm}")
    print(f"  Particion        : {salida['metadata']['split']['strategy']} "
          f"(train={salida['metrics']['n_entrenamiento']}, test={salida['metrics']['n_prueba']})")
    print(f"  Modelo  (prueba) : MAE={m['MAE']:,.2f}  RMSE={m['RMSE']:,.2f}  R2={m['R2']:.4f}")
    print(f"  Base    (prueba) : MAE={base['MAE']:,.2f}  RMSE={base['RMSE']:,.2f}  R2={base['R2']:.4f}")
    cv = salida["metrics"]["validacion_cruzada_entrenamiento"]
    if cv:
        print(f"  Val. cruzada     : MAE={cv['MAE_media']:,.2f} (+/- {cv['MAE_desv']:,.2f}), {cv['folds']} folds, {cv['esquema']}")
    print("Archivos generados:")
    for nombre in (config.MODEL_FILENAME, config.METRICS_FILENAME, config.METADATA_FILENAME):
        print(f"  {os.path.join(str(vd), nombre)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
