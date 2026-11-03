"""Hace importable train.py / predict.py / src/ al correr `python -m pytest` desde fase-2."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
