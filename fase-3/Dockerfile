# Se construye desde la RAIZ del repositorio (el contexto incluye fase-2/ y fase-3/):
#   docker build -f fase-3/Dockerfile -t api-vivienda:3.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
#
# Imagen base con version explicita (no "latest"): la misma version menor de Python que el entorno local (3.14).
# Si "docker build" falla en esta linea (la etiqueta no se descarga), usar python:3.12-slim.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencias primero: aprovecha la cache de capas si solo cambia el codigo
COPY fase-3/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Codigo de fase-2 que reutiliza la API (preprocesamiento, versionamiento y entrenamiento)...
COPY fase-2/src/ ./src/
COPY fase-2/train.py ./
# ...y el codigo de la API. Los datos y los modelos NO van en la imagen: se montan al ejecutar.
COPY fase-3/api.py fase-3/pytest.ini fase-3/cliente_prueba.py fase-3/ejemplo_solicitud.json ./
COPY fase-3/tests/ ./tests/

# Referencia al commit del codigo (se guarda en metadata.json de las versiones que entrene /train)
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

# Rutas dentro del contenedor (se corresponden con los montajes del README) y version activa
ENV MODELS_DIR=/app/models \
    DATA_PATH=/app/data/housing.csv \
    MODEL_VERSION=v1

EXPOSE 8000

# 0.0.0.0 para que el puerto publicado con -p 8000:8000 sea alcanzable desde el equipo anfitrion.
# Para pruebas: docker run --rm <imagen> python -m pytest
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
