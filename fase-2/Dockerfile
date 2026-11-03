# Imagen base con version explicita (no "latest"): misma version menor de Python que el entorno local (3.14).
# Si "docker build" falla en esta linea (la etiqueta no se descarga), usar python:3.12-slim:
# las mismas versiones de requirements.txt tambien se verificaron con Python 3.12.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias primero: aprovecha la cache de capas si solo cambia el codigo
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de la aplicacion (los datos y los modelos NO van en la imagen: se montan al ejecutar)
COPY src/ ./src/
COPY train.py predict.py pytest.ini ./
COPY tests/ ./tests/

# Referencia al commit del codigo; se pasa con --build-arg GIT_COMMIT=$(git rev-parse HEAD)
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Sin ENTRYPOINT: el comando (train.py, predict.py o pytest) se indica en "docker run".
# El entrenamiento NO ocurre durante "docker build".
CMD ["python", "train.py", "--help"]
