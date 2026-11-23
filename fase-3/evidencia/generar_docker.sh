#!/usr/bin/env bash
# Genera fase-3/evidencia/docker.txt con la evidencia de ejecucion de la Fase 3 en Docker.
# Ejecutar desde la RAIZ del repositorio:   bash fase-3/evidencia/generar_docker.sh
# Requiere Docker y fase-2/data/housing.csv en su sitio (en GitHub Codespaces, Docker ya viene).

set -uo pipefail

SALIDA="fase-3/evidencia/docker.txt"
IMAGEN="api-vivienda:3.0"
NOMBRE="api-fase3"

if [ ! -f "fase-3/api.py" ]; then
  echo "ERROR: ejecute este script desde la raiz del repositorio (donde estan fase-2/ y fase-3/)." >&2
  exit 1
fi
if [ ! -f "fase-2/data/housing.csv" ]; then
  echo "ERROR: falta fase-2/data/housing.csv. Subalo a esa carpeta y vuelva a ejecutar." >&2
  exit 1
fi

mkdir -p fase-3/evidencia
: > "$SALIDA"

registrar() { printf '%s\n' "$*" | tee -a "$SALIDA"; }
paso() {
  local titulo="$1"; shift
  registrar ""
  registrar "=============================================================="
  registrar "\$ $titulo"
  registrar "=============================================================="
  "$@" 2>&1 | tee -a "$SALIDA"
}

registrar "Evidencia de ejecucion de la Fase 3 en Docker"
registrar "Fecha: $(date '+%Y-%m-%d %H:%M:%S %Z')"
registrar "Entorno: $(uname -srm)  ·  Terminal: bash"
registrar "Commit: $(git rev-parse HEAD)"

docker rm -f "$NOMBRE" >/dev/null 2>&1 || true

paso "docker --version" docker --version

paso "docker build -f fase-3/Dockerfile -t $IMAGEN --build-arg GIT_COMMIT=\$(git rev-parse HEAD) ." \
  docker build -f fase-3/Dockerfile -t "$IMAGEN" --build-arg GIT_COMMIT="$(git rev-parse HEAD)" .

paso "docker run -d --rm --name $NOMBRE -p 8000:8000 -v datos:ro -v modelos -e MODEL_VERSION=v1 $IMAGEN" \
  docker run -d --rm --name "$NOMBRE" -p 8000:8000 \
    -v "$PWD/fase-2/data:/app/data:ro" \
    -v "$PWD/fase-2/models:/app/models" \
    -e MODEL_VERSION=v1 \
    "$IMAGEN"

echo "Esperando a que la API responda..."
listo=0
for _ in $(seq 1 40); do
  sleep 2
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then listo=1; break; fi
done
if [ "$listo" -ne 1 ]; then
  registrar ""
  registrar "ERROR: la API no respondio en 80 segundos. Registro del contenedor:"
  docker logs "$NOMBRE" 2>&1 | tee -a "$SALIDA"
  docker stop "$NOMBRE" >/dev/null 2>&1 || true
  exit 1
fi

paso "curl -s http://localhost:8000/health" \
  bash -c "curl -s http://localhost:8000/health | python3 -m json.tool"

paso "curl -s http://localhost:8000/versiones" \
  bash -c "curl -s http://localhost:8000/versiones | python3 -m json.tool"

paso "python3 fase-3/cliente_prueba.py   (prediccion con un registro real del proyecto)" \
  python3 fase-3/cliente_prueba.py --url http://localhost:8000

paso "python3 fase-3/cliente_prueba.py --quitar median_income   (entrada invalida: debe dar 422)" \
  python3 fase-3/cliente_prueba.py --url http://localhost:8000 --quitar median_income

paso "curl -s -X POST http://localhost:8000/train -d '{}'   (entrena y guarda una version nueva)" \
  bash -c "curl -s -X POST http://localhost:8000/train -H 'Content-Type: application/json' -d '{}' --max-time 900 | python3 -m json.tool"

paso "curl -s http://localhost:8000/versiones   (la version nueva ya aparece)" \
  bash -c "curl -s http://localhost:8000/versiones | python3 -m json.tool"

paso "ls -la fase-2/models   (fuera del contenedor: la version entrenada persiste)" \
  ls -la fase-2/models

paso "docker stop $NOMBRE" docker stop "$NOMBRE"

paso "docker run --rm $IMAGEN python -m pytest   (las pruebas dentro de la imagen)" \
  docker run --rm "$IMAGEN" python -m pytest

registrar ""
registrar "=============================================================="
registrar "Fin de la evidencia."
echo ""
echo "Listo. Evidencia guardada en $SALIDA"
