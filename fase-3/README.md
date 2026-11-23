# Fase 3 — API REST: Predicción de precios de vivienda en California

**Proyecto Integrador — Modelos y Simulación de Sistemas I (2026-II)** · Docente: Andrés Parra

**Integrantes:** Oscar Plaza · Samuel Velásquez · Juan José Durango

La Fase 3 expone por HTTP el mismo modelo de la Fase 2. No se reentrena nada ni se cambia el pipeline: la API carga una versión ya guardada en `fase-2/models/` y le aplica el preprocesamiento registrado en su `metadata.json`. Las carpetas `fase-1/` y `fase-2/` se conservan sin modificar.

## 1. Qué hace la API

| Operación | Respuesta | Qué hace |
|---|---|---|
| `GET /health` | 200 | Indica que la API está en ejecución; informa la versión configurada y si está disponible. |
| `GET /versiones` | 200 | Lista las versiones guardadas y cuál se usa para predecir. |
| `POST /predict` | 200 | Recibe un registro JSON con las 9 variables predictoras, carga la versión configurada, aplica sus transformaciones y devuelve el valor estimado y la versión usada. |
| `POST /train` | 201 | Entrena con los datos y la configuración de la Fase 2 y guarda una **versión nueva** con su modelo, métricas y metadatos. No sobrescribe las anteriores. |

`GET /versiones` no la pide la especificación; se agregó porque es la forma de averiguar qué identificadores existen antes de cambiar la versión activa.

Reutilización del código de la Fase 2, sin copiarlo ni modificarlo:

| La API usa | De la Fase 2 |
|---|---|
| Preprocesamiento de datos nuevos | `src.data.prepare_prediction_frame` con el `input_schema` de `metadata.json` |
| Carga del modelo | `modelo.joblib` (pipeline completo: imputación + codificación + modelo) |
| Entrenamiento | `train.run_training`, la misma función que usa `train.py` |
| Validación de identificadores de versión | `src.versioning` |

Documentación interactiva generada por FastAPI: `http://localhost:8000/docs`.

## 2. Entorno

```powershell
cd fase-3
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell. En Bash: source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` agrega al conjunto de la Fase 2 las dependencias del servicio con versiones fijas: `fastapi`, `starlette`, `pydantic`, `uvicorn` y `httpx2` (esta última la necesita el cliente de pruebas de Starlette). Probado con Python 3.13 y 3.14; la imagen Docker usa `python:3.14-slim`.

**Los comandos de las secciones 3 a 5 se ejecutan desde `fase-3/`.** Los de Docker (sección 6) se ejecutan desde la **raíz del repositorio**, porque la imagen necesita también `fase-2/`.

## 3. Arranque local

```powershell
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Sin variables de entorno, la API toma los modelos de `fase-2/models/`, los datos de `fase-2/data/housing.csv` y predice con `v1`.

## 4. Configuración

Se configura con variables de entorno; se leen en cada solicitud, de modo que basta reiniciar el proceso para cambiar de versión.

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `MODEL_VERSION` | `v1` | Versión que usa `POST /predict`. |
| `MODELS_DIR` | `fase-2/models` | Carpeta con las versiones guardadas. |
| `DATA_PATH` | `fase-2/data/housing.csv` | CSV con el que entrena `POST /train`. |
| `TARGET` | `median_house_value` | Variable objetivo del entrenamiento. |
| `SEED` | `42` | Semilla del entrenamiento. |

### Seleccionar la versión disponible

`GET /versiones` lista los identificadores existentes. Para predecir con otro, se define `MODEL_VERSION` y se reinicia la API:

```powershell
# Local (PowerShell), desde fase-3/
$env:MODEL_VERSION = "v2"
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# Docker: se pasa al contenedor
docker run --rm -p 8000:8000 -e MODEL_VERSION=v2 ... api-vivienda:3.0
```

Para volver a la versión anterior se repite el procedimiento con `v1`. Si la versión configurada no existe o sus archivos no se pueden cargar, la API responde **503** y el cuerpo incluye `versiones_disponibles` y la indicación de cómo seleccionarla; `GET /health` sigue respondiendo 200 con `"modelo_disponible": false`.

## 5. Contrato de entrada y salida

Un registro de `POST /predict` lleva las 9 variables predictoras de la Fase 2, con los mismos nombres y tipos usados al entrenar: los números como números JSON y la categoría como texto. La variable objetivo no hace falta; si llega, se ignora, igual que cualquier otro campo de más.

| Campo | Tipo | Nulo |
|---|---|---|
| `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `population`, `households`, `median_income` | número | no |
| `total_bedrooms` | número | **sí** (`null`): se imputa con la mediana aprendida en el entrenamiento, la regla guardada en la Fase 2 |
| `ocean_proximity` | texto: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN` | no |

Solo `total_bedrooms` admite `null` porque es la única variable con valores faltantes en los datos de entrenamiento; para las demás, un `null` sería un dato inventado y no un dato ausente conocido.

**Ejemplo de solicitud** (es el contenido de `ejemplo_solicitud.json`, tomado de la primera fila del CSV del proyecto):

```json
{
  "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0,
  "total_rooms": 880.0, "total_bedrooms": 129.0, "population": 322.0,
  "households": 126.0, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"
}
```

**Respuesta** (HTTP 200): el valor mediano de vivienda estimado, en USD, y la versión que lo produjo.

```json
{"prediccion": 431942.36, "version": "v1"}
```

### Errores

Todos devuelven un JSON con `error` y `mensaje`; los de validación agregan `detalle` con el campo y el problema.

| Situación | Código | Mensaje |
|---|---|---|
| Falta un campo obligatorio | 422 | `median_income: campo obligatorio ausente` |
| Tipo incompatible | 422 | `median_income: tipo incompatible: se esperaba numero (se recibio str)` |
| `null` en un campo que no lo admite | 422 | `no admite null (solo total_bedrooms admite null)` |
| Categoría no vista en entrenamiento | 422 | `ocean_proximity: categoria desconocida 'NEAR LAKE'. Categorias validas: [...]` |
| El cuerpo no es un objeto JSON válido | 422 | `(cuerpo): el cuerpo no es un JSON valido` |
| La versión configurada no está disponible | 503 | `La version 'vX' no esta disponible en ...` + `versiones_disponibles` |
| `POST /train` sin el archivo de datos | 503 | `No se encuentra el archivo de datos ...` |
| `POST /train` con un identificador ya usado | 409 | la versión existe y no se sobrescribe |

A diferencia de `predict.py` de la Fase 2, una **categoría no vista se rechaza** en vez de advertirse: en un servicio, devolver una predicción calculada sobre una fila de ceros sin que el cliente se entere es peor que un error explícito.

### Entrenar una versión nueva

```powershell
curl.exe -s -X POST http://localhost:8000/train -H "Content-Type: application/json" -d "{}"
```

El cuerpo es opcional y admite `version`, `seed` y `n_estimators`. Sin `version` se genera un identificador con fecha y hora (`v20261122-213000`). La respuesta (201) trae el identificador, la ruta donde quedó y las métricas de prueba del modelo y del modelo base. **La versión nueva no se usa para predecir hasta que se cambie `MODEL_VERSION` y se reinicie la API**; así lo informa el campo `nota` de la respuesta. Las versiones anteriores quedan intactas: repetir un identificador existente falla con 409.

## 6. Docker

La imagen se construye **desde la raíz del repositorio**, porque el contexto debe incluir `fase-2/src/` y `fase-2/train.py` además de `fase-3/`. Los datos y los modelos **no** van dentro de la imagen: se montan al ejecutar, igual que en la Fase 2, de modo que lo que entrene `POST /train` persiste en el equipo.

```powershell
# Desde la raiz del repositorio
docker build -f fase-3/Dockerfile -t api-vivienda:3.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .

docker run --rm -p 8000:8000 `
  -v "${PWD}/fase-2/data:/app/data:ro" `
  -v "${PWD}/fase-2/models:/app/models" `
  -e MODEL_VERSION=v1 `
  api-vivienda:3.0
```

```bash
# Bash
docker build -f fase-3/Dockerfile -t api-vivienda:3.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .

docker run --rm -p 8000:8000 \
  -v "$PWD/fase-2/data:/app/data:ro" \
  -v "$PWD/fase-2/models:/app/models" \
  -e MODEL_VERSION=v1 \
  api-vivienda:3.0
```

| Aspecto | Valor |
|---|---|
| Puerto | el contenedor escucha en `0.0.0.0:8000`; se publica con `-p 8000:8000` y se consulta en `http://localhost:8000` |
| Acceso a los datos | `fase-2/data` montado en `/app/data` en solo lectura; `DATA_PATH=/app/data/housing.csv` |
| Montaje de los modelos | `fase-2/models` montado en `/app/models` con escritura; `MODELS_DIR=/app/models`. Lo que guarde `POST /train` aparece en `fase-2/models/` del equipo |
| Versión activa | `-e MODEL_VERSION=v1` (por defecto `v1`) |

Las pruebas también corren dentro de la imagen: `docker run --rm api-vivienda:3.0 python -m pytest`.

`fase-3/Dockerfile.dockerignore` excluye del contexto el historial de Git, los entornos virtuales, las cachés, los datos y los modelos. Es la convención de BuildKit para un `.dockerignore` propio de un Dockerfile con nombre explícito, necesaria aquí porque el contexto es la raíz del repositorio y el `.dockerignore` de la Fase 2 no aplica.

> Si `docker build` falla al descargar `python:3.14-slim`, usar `python:3.12-slim` en la primera línea del Dockerfile. Las versiones de `requirements.txt` funcionan en ambas.

## 7. Cliente de prueba

`cliente_prueba.py` consulta `/health` y envía una predicción. Solo usa la librería estándar, así que corre sin instalar nada.

```powershell
python cliente_prueba.py                                            # usa ejemplo_solicitud.json
python cliente_prueba.py --csv ..\fase-2\data\ejemplo_prediccion.csv --fila 2
python cliente_prueba.py --quitar median_income                     # prueba de validacion: 422
```

Salida con la API en ejecución:

```
GET  /health  -> 200 {"status": "ok", "mensaje": "La API esta en ejecucion", "version_configurada": "v1", "modelo_disponible": true}
POST /predict  solicitud: {"longitude": -122.23, "latitude": 37.88, ...}
POST /predict  -> 200 {"prediccion": 431942.36, "version": "v1"}
```

Devuelve código de salida 0 si la predicción fue 200 y 1 en cualquier otro caso, de modo que sirve en un script.

## 8. Pruebas (pytest)

```powershell
python -m pytest
```

16 pruebas, sin red y sin el dataset completo: entrenan un modelo pequeño sobre datos sintéticos con el mismo esquema, en un directorio temporal, y apuntan la API a él con variables de entorno. Resultado: `16 passed`, registrado en [`evidencia/pytest.txt`](evidencia/pytest.txt), cuyo encabezado deja constancia del entorno (Windows, Python 3.13.7, pytest 9.1.1).

Los **tres casos que exige la especificación** son estos:

| Caso exigido | Prueba |
|---|---|
| `/health` responde 200 | `test_health_responde_siempre_aunque_no_haya_modelo` |
| Una entrada válida devuelve predicción y versión | `test_predict_devuelve_un_valor_numerico` |
| Falta un campo obligatorio y se rechaza | `test_predict_sin_un_campo_obligatorio_responde_422_y_lo_nombra` |

Las trece restantes:

| Prueba | Qué comprueba |
|---|---|
| `test_health_reporta_el_modelo_cuando_ya_existe` | `/health` distingue entre «en ejecución» y «con modelo disponible». |
| `test_versiones_lista_lo_guardado` | `/versiones` lista lo que hay en disco y señala la activa. |
| `test_predict_admite_null_solo_en_total_bedrooms` | `null` en `total_bedrooms` se imputa y predice; en otra variable es 422. Además, enviar `null` y **omitir** el campo no son lo mismo: lo primero es un dato ausente conocido y lo segundo una solicitud incompleta, que se rechaza. |
| `test_predict_ignora_campos_de_mas` | Un campo extra (incluida la variable objetivo) no altera la predicción. |
| `test_predict_con_tipo_incompatible_responde_422` | Un texto donde se espera un número se rechaza nombrando el campo. |
| `test_predict_rechaza_cualquier_valor_que_no_sea_un_numero_finito` | Seis formas de valor inválido en un campo numérico (`"abc"`, `true`, `NaN`, `Infinity`, una lista y `null`) dan 422 y nombran el campo. Cubre `strict=True` y `allow_inf_nan=False`: sin ellos, `true` se convertiría en `1.0` y un `NaN` llegaría al modelo, y la API respondería 200 con un número sin sentido. |
| `test_predict_con_categoria_desconocida_responde_422_y_lista_las_validas` | Una categoría no vista se rechaza y el mensaje lista las válidas. |
| `test_predict_sin_modelo_responde_503_y_explica_como_resolverlo` | Sin la versión configurada: 503 con las versiones disponibles y cómo seleccionarla. |
| `test_train_crea_una_version_nueva_con_sus_artefactos` | `/train` responde 201 y deja `modelo.joblib`, `metrics.json` y `metadata.json`. |
| `test_train_no_sobrescribe_una_version_existente` | Repetir un identificador falla sin tocar los archivos de la versión anterior. |
| `test_train_sin_datos_responde_503` | Sin el CSV de entrenamiento: 503 con un mensaje que lo explica. |
| `test_train_y_predict_usan_el_mismo_esquema` | Una versión recién entrenada sirve para predecir: `/train` y `/predict` coinciden. |
| `test_el_ejemplo_de_solicitud_es_valido` | `ejemplo_solicitud.json` es aceptado por la API, de modo que el ejemplo del README no se desactualiza. |

## 9. Evidencia de ejecución

| Archivo | Contenido |
|---|---|
| [`evidencia/pytest.txt`](evidencia/pytest.txt) | `python -m pytest` en el entorno local |
| `evidencia/generar_docker.ps1` | el script que genera la evidencia de Docker: ejecuta la secuencia completa desde la raíz del repositorio (build, arranque del contenedor, `/health`, `/versiones`, el cliente de prueba, una entrada inválida, `POST /train`, el listado de `fase-2/models/` mostrando que la versión entrenada persiste, y las pruebas dentro de la imagen) y deja la salida en `evidencia/docker.txt` |
| `evidencia/docker.txt` | **pendiente.** Lo aporta el integrante que tiene Docker instalado, ejecutando el script anterior |

La versión que crea `POST /train` durante esa comprobación queda en `fase-2/models/` con un identificador de fecha y hora (`v20261122-...`) y **no se versiona** en Git: es evidencia de ejecución, no una entrega.

## 10. Estructura

```
fase-3/
├── api.py                     # la API: /health, /versiones, /predict, /train
├── cliente_prueba.py          # cliente sencillo (solo libreria estandar)
├── ejemplo_solicitud.json     # registro de ejemplo, tomado del CSV del proyecto
├── requirements.txt           # dependencias con versiones fijas
├── Dockerfile                 # se construye desde la raiz del repositorio
├── Dockerfile.dockerignore    # contexto de construccion (convencion de BuildKit)
├── pytest.ini
├── tests/                     # conftest.py y test_api.py (16 pruebas)
├── evidencia/                 # salidas reales de terminal
└── README.md                  # este archivo
```

## 11. Fuentes consultadas y código externo adaptado

- Documentación de FastAPI (routing, modelos de pydantic, manejadores de excepciones) y de Starlette (`TestClient`).
- Documentación de pytest (fixtures, `monkeypatch`, `tmp_path`).
- El pipeline, el esquema de entrada y la función de entrenamiento se reutilizan de la Fase 2 de este mismo repositorio; no se reescribieron.
- **Asistencia de IA generativa.** La estructura inicial de `api.py`, del `Dockerfile`, de las pruebas y de este README se elaboró con asistencia de Claude (Anthropic). El equipo revisó el código, lo ejecutó en sus propios entornos y lo comprende; las cifras y las salidas de este README provienen de ejecuciones propias registradas en `evidencia/`.
