"""Versionamiento de artefactos del modelo mediante carpetas: models/<version>/."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

from .config import MODEL_FILENAME
from .errors import AplicacionError

# Letras, numeros, punto, guion y guion bajo. Evita rutas como "../x" o "a/b".
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def default_version_id() -> str:
    return datetime.now().strftime("v%Y%m%d-%H%M%S")


def validate_version_id(version: str) -> None:
    if not _VERSION_RE.match(version):
        raise AplicacionError(
            f"Identificador de version invalido: '{version}'. Usa letras, numeros, '.', '-' o '_' "
            "(maximo 64 caracteres, sin rutas)."
        )


def ensure_version_free(models_dir, version: str) -> None:
    """Falla pronto (antes de entrenar) si la version ya existe."""
    validate_version_id(version)
    destino = Path(models_dir) / version
    if destino.exists():
        raise AplicacionError(
            f"La version '{version}' ya existe en {models_dir}. No se sobrescribe: "
            "usa otro --version o omitelo para generar un identificador nuevo."
        )


def create_version_dir(models_dir, version: str) -> Path:
    """Crea models/<version>/. mkdir sin exist_ok es atomico: rechaza la repeticion."""
    validate_version_id(version)
    base = Path(models_dir)
    base.mkdir(parents=True, exist_ok=True)
    destino = base / version
    try:
        destino.mkdir()
    except FileExistsError as exc:
        raise AplicacionError(
            f"La version '{version}' ya existe en {models_dir}. No se sobrescribe."
        ) from exc
    return destino


def write_json(path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def read_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_model_path(path) -> Path:
    """Acepta la ruta al .joblib o al directorio de la version (models/v1)."""
    p = Path(path)
    if p.is_dir():
        p = p / MODEL_FILENAME
    if not p.exists():
        raise AplicacionError(f"No existe el modelo: {p}")
    if not p.is_file():
        raise AplicacionError(f"La ruta del modelo no es un archivo: {p}")
    return p


def code_reference() -> str:
    """Identifica el codigo usado: variable GIT_COMMIT (Docker) o hash del commit actual.

    Se agrega '-dirty' si hay cambios sin confirmar en archivos versionados.
    """
    env = os.environ.get("GIT_COMMIT", "").strip()
    if env and env != "unknown":
        return env
    repo = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        sucio = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return commit + ("-dirty" if sucio else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"
