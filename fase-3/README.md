# Fase 3 — API REST: consultar el modelo y entrenar versiones por HTTP

**Proyecto Integrador — Modelos y Simulación de Sistemas I (2026-II)** · Docente: Andrés Parra

**Integrantes:** Oscar Plaza · Samuel Velásquez · Juan José Durango

**Versión entregada (Entrega 3):** etiqueta de Git `entrega-3` (`git checkout entrega-3` deja el repositorio en el estado evaluado). Es distinta de los identificadores de modelo (`v1`, `v2`).

## 1. Qué es y cómo se relaciona con las fases anteriores

Una API (FastAPI) que pone el modelo de la Fase 2 detrás de tres operaciones HTTP. **No reimplementa nada**: reutiliza el código de `fase-2/` sin modificarlo.

| Operación | Qué hace | Qué reutiliza de la Fase 2 |
|---|---|---|
| `GET /health` | Indica que la API está en ejecución (HTTP 200). | — |
| `POST /predict` | Recibe un distrito en JSON, carga la versión configurada y devuelve la predicción y la versión usada. | `modelo.joblib` (el `Pipeline` con el preprocesamiento guardado), `metadata.json` (esquema de entrada) y `src.data.prepare_prediction_frame`. |
| `POST /train` | Entrena con los datos y la configuración de la Fase 2 y guarda una **versión nueva** (modelo, métricas y metadatos). | `train.run_training`, la misma función que usa `train.py`. |

Además hay `GET /versiones` (lista las versiones guardadas y la configurada) y la documentación interactiva de FastAPI en `/docs`.

`fase-1/` y `fase-2/` se conservan sin cambios.

## 2. Estructura de `fase-3/`

```
fase-3/
├── api.py                  # la API (rutas, validación, carga del modelo)
├── cliente_prueba.py       # cliente sencillo: consulta /health y envía una predicción
├── ejemplo_solicitud.json  # primera fila de fase-2/data/ejemplo_prediccion.csv, como JSON
├── requirements.txt        # dependencias con versiones fijas
├── Dockerfile              # se construye desde la RAIZ del repositorio
├── Dockerfile.dockerignore
├── pytest.ini
├── tests/                  # conftest.py y test_api.py
├── evidencia/              # salidas reales de terminal (pytest, docker)
└── README.md
```

## 3. Entorno local (PowerShell, desde `fase-3/`)

Python **3.14.5** (el del equipo; la imagen Docker usa `python:3.14-slim`).

```powershell
cd fase-3
python -m venv venv
venv\Scripts\Activate.ps1            # si se bloquea: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements.txt
```

El entorno virtual está ignorado por Git. La API importa `src/` y `train.py` de `../fase-2` por sí sola; no hay que copiar nada.

## 4. Configuración

Se hace con variables de entorno, que se leen en cada solicitud. Los valores por defecto sirven para trabajar desde el repositorio.

| Variable | Qué controla | Por defecto (local) | En Docker |
|---|---|---|---|
| `MODEL_VERSION` | **Versión que usa `/predict`** | `v1` | `v1` |
| `MODELS_DIR` | Carpeta con las versiones guardadas (`<MODELS_DIR>/<version>/`) | `fase-2/models` | `/app/models` (montaje) |
| `DATA_PATH` | CSV con el que entrena `/train` | `fase-2/data/housing.csv` | `/app/data/housing.csv` (montaje) |
| `TARGET` | Variable objetivo | `median_house_value` | igual |
| `SEED` | Semilla de `/train` | `42` | igual |

## 5. Ejecutar

### Local

```powershell
# desde fase-3/, con el entorno virtual activo
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Datos de entrenamiento: `housing.csv` debe estar en `fase-2\data\` (está ignorado por Git; se copia de la Fase 1: `Copy-Item ..\fase-1\data\housing.csv ..\fase-2\data\housing.csv`). Para predecir basta con que existan `fase-2\models\v1` y `v2`, que ya están en el repositorio.

### Docker

Se construye **desde la raíz del repositorio** (el contexto debe incluir `fase-2/` y `fase-3/`):

```powershell
cd <raiz del repositorio>
docker build -f fase-3/Dockerfile -t api-vivienda:3.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm -p 8000:8000 `
  -v "${PWD}/fase-2/models:/app/models" `
  -v "${PWD}/fase-2/data:/app/data:ro" `
  api-vivienda:3.0
```

(En Bash, el mismo comando con `\` en lugar de `` ` `` y `$(pwd)` en lugar de `${PWD}`.)

| Tema | Cómo queda |
|---|---|
| Puerto | `8000` del contenedor publicado en `8000` del equipo (`-p 8000:8000`). El servidor escucha en `0.0.0.0` dentro del contenedor. |
| Datos | `fase-2/data` se monta en `/app/data` **solo lectura**; `/train` lee de ahí `housing.csv`. |
| Modelos | `fase-2/models` se monta en `/app/models` **con escritura**: `/predict` lee de ahí y `/train` guarda ahí las versiones nuevas, que **persisten** al detener el contenedor. |
| Entrenamiento | No ocurre en `docker build`. |

Las versiones que crea `/train` con identificador automático (`v2026…`) quedan en `fase-2/models/` pero **Git las ignora** (`fase-2/models/v20*/`), para no subir por accidente otro modelo de decenas de MB.

Si la etiqueta `python:3.14-slim` no se descarga, cambiar la primera línea del `Dockerfile` a `python:3.12-slim`.

**Dónde se demostró.** La demostración en Docker se ejecutó en **GitHub Actions** (máquina Ubuntu con Docker), con los mismos comandos de esta sección, porque los integrantes presentes no tenían Docker instalado. El flujo está en `.github/workflows/docker.yml` y el registro completo en `evidencia/docker.txt` (sección 12). Cualquiera puede repetirla desde la pestaña *Actions* del repositorio o en un equipo con Docker con los comandos de arriba.

## 6. Operaciones

### `GET /health`

```json
{"status": "ok", "mensaje": "La API esta en ejecucion", "version_configurada": "v1", "modelo_disponible": true}
```

Siempre responde 200 mientras el proceso funcione; `modelo_disponible` avisa si la versión configurada existe.

### `POST /predict`

**Del CSV al JSON.** Cada columna predictora de `fase-2/data/ejemplo_prediccion.csv` es una clave; un número se envía como **número**, la categoría como **texto** y un dato faltante como `null`. Se envían las 9 variables:

| Variable | Tipo JSON | ¿Admite `null`? |
|---|---|---|
| `longitude`, `latitude`, `housing_median_age`, `total_rooms`, `population`, `households`, `median_income` | número | No |
| `total_bedrooms` | número | **Sí** (se imputa con la mediana aprendida en el entrenamiento, la regla de la Fase 2) |
| `ocean_proximity` | texto: `<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY` o `NEAR OCEAN` | No |

La variable objetivo (`median_house_value`) **no hace falta**; si llega, se ignora. La clave de cada variable es obligatoria aunque su valor sea `null`.

Solicitud y respuesta (con el modelo real `v1`; el valor coincide con el del modelo de la Fase 1 para ese distrito):

```json
{"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41.0, "total_rooms": 880.0,
 "total_bedrooms": 129.0, "population": 322.0, "households": 126.0, "median_income": 8.3252,
 "ocean_proximity": "NEAR BAY"}
```
```json
{"prediccion": 431942.36, "version": "v1"}
```

Con `curl` (en PowerShell hay que escribir `curl.exe`, porque `curl` es un alias de otro comando):

```powershell
curl.exe -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "@ejemplo_solicitud.json"
```

**Errores** (siempre un JSON con `error` y `mensaje`):

| Caso | Código | Ejemplo de mensaje |
|---|---|---|
| Falta una variable | **422** | `La entrada no es valida. median_income: campo obligatorio ausente` |
| Tipo incompatible (texto por número, booleano, lista…) | **422** | `latitude: tipo incompatible: se esperaba numero (se recibio str)` |
| `null` en una variable que no lo admite | **422** | `median_income: no admite null (solo total_bedrooms admite null)` |
| Categoría que el modelo no conoce | **422** | `ocean_proximity: categoria desconocida 'NEAR_BAY'. Categorias validas: [...]` |
| Cuerpo que no es JSON ni un objeto | **422** | `(cuerpo): el cuerpo no es un JSON valido` |
| La versión configurada no existe o no se puede cargar | **503** | `La version 'v99' no esta disponible en ...` + `versiones_disponibles` + `como_seleccionar` |

### `POST /train`

Entrena con `DATA_PATH`, `TARGET` y `SEED` (los de la Fase 2) y acepta un **cuerpo vacío**:

```powershell
curl.exe -s -X POST http://localhost:8000/train
```

Cuerpo opcional: `{"version": "v3", "seed": 7, "n_estimators": 50}`. Si no se da `version`, genera un identificador disponible con fecha y hora (`v20261117-142453`). Responde **201** con la versión, su ruta, las métricas de prueba y la versión activa:

```json
{"version": "v20261117-142453", "ruta": "models/v20261117-142453",
 "metricas_prueba": {"modelo": {"MAE": ..., "RMSE": ..., "R2": ...}, "modelo_base": {...}},
 "version_activa": "v1",
 "nota": "La version nueva esta guardada pero /predict sigue usando la version activa. ..."}
```

- **No se sobrescribe nada**: una versión que ya existe responde **409**; un identificador inválido, 422; si falta el CSV, **503** (`datos_no_disponibles`).
- **La versión nueva no se activa sola**: `/predict` sigue con `MODEL_VERSION` hasta que se cambie (sección 7).
- El entrenamiento es síncrono y con el dataset completo tarda unos 30 segundos; un segundo `/train` simultáneo recibe 409.

## 7. Seleccionar la versión

`/predict` usa la versión de la variable `MODEL_VERSION` (por defecto `v1`). Para ver cuáles hay:

```powershell
curl.exe -s http://localhost:8000/versiones
```

Para usar otra, **reinicie la API** con la variable definida:

```powershell
# local
$env:MODEL_VERSION = "v2"
python -m uvicorn api:app --host 127.0.0.1 --port 8000

# Docker: añadir  -e MODEL_VERSION=v2  al docker run
```

Si la versión indicada no existe, `/predict` devuelve **503** con la lista de versiones disponibles y esta misma instrucción.

## 8. Cliente de prueba

`cliente_prueba.py` usa solo la biblioteca estándar. Consulta `/health` y envía una predicción con el registro de `ejemplo_solicitud.json`:

```powershell
python cliente_prueba.py
python cliente_prueba.py --csv ..\fase-2\data\ejemplo_prediccion.csv --fila 3     # otra fila del CSV de la Fase 2
python cliente_prueba.py --quitar median_income                                    # entrada inválida: debe dar 422
```

Termina con código 0 si `/predict` responde 200 y con 1 en caso contrario.

## 9. Pruebas

```powershell
# desde fase-3/, con el entorno virtual activo
python -m pytest
```

Resultado: **16 passed** (Windows con Python 3.13.7 y 3.14.5, y dentro de la imagen Docker). Salida completa en `evidencia/pytest.txt`.

Entrenan un modelo **pequeño** (5 árboles) sobre datos sintéticos en una carpeta temporal (no tocan `fase-2/models` ni necesitan `housing.csv`). Las tres exigidas:

| Prueba | Qué comprueba |
|---|---|
| `test_health_responde_siempre_aunque_no_haya_modelo` | `/health` responde 200 y `status` es `ok`. |
| `test_predict_devuelve_un_valor_numerico` | Un registro válido devuelve 200 con `prediccion` numérica y `version`. |
| `test_predict_sin_un_campo_obligatorio_responde_422_y_lo_nombra` | Sin `median_income` responde 422 y el mensaje nombra la variable. |

Las otras trece comprueban: `/health` cuando el modelo existe; `/versiones`; que `null` solo se admite en `total_bedrooms` y que **omitir** ese campo da 422 (antes daba 200 con una predicción inventada); que los campos de más se ignoran; los tipos incompatibles; que en `longitude` se rechaza cualquier valor que no sea un número finito (texto, booleano, `NaN`, `Infinity`, lista, `null`); la categoría desconocida con la lista de válidas; el 503 sin modelo; que `/train` guarda modelo, métricas y metadatos, no sobrescribe (409) y responde 503 sin datos; que una versión entrenada por `/train` se usa en `/predict`; y que `ejemplo_solicitud.json` funciona tal cual.

Para comprobar que las pruebas detectan fallos reales se modificó `api.py` a propósito (quitar los tipos estrictos, no nombrar el campo en el error, dar valor por defecto a `total_bedrooms`): en cada caso falló al menos una prueba, y los cambios se revirtieron.

En Docker: `docker run --rm api-vivienda:3.0 python -m pytest`.

## 10. Comprobación desde una copia nueva del repositorio

| Paso | Qué hacer | Qué debe verse |
|---|---|---|
| 01 | Construir la imagen y arrancar el contenedor (sección 5, Docker). | La API escucha en `http://localhost:8000`. |
| 02 | `curl.exe -s http://localhost:8000/health` y luego `python cliente_prueba.py` (desde `fase-3/`). | `/health` 200; `/predict` devuelve `{"prediccion": 431942.36, "version": "v1"}`. |
| 03 | `curl.exe -s -X POST http://localhost:8000/train`; después `Get-ChildItem fase-2\models -Name`; y `python -m pytest` (desde `fase-3/`). | Aparece una carpeta `v2026…` nueva junto a `v1` y `v2`; pasan las pruebas. |

## 11. Decisiones y limitaciones

- **Solo `total_bedrooms` admite `null`**, porque es la única variable con valores faltantes en los datos de entrenamiento. Para las demás, imputar la mediana de una coordenada o de un ingreso produciría una predicción sin sentido, así que se rechaza con 422.
- **Una categoría desconocida se rechaza** (la Fase 2 la aceptaba con una advertencia): en una API un error de escritura (`NEAR_BAY`) devolvería en silencio una predicción equivocada.
- Los tipos son estrictos: `"24.5"` (texto) no se acepta como número.
- `/train` es síncrono, no tiene autenticación y entrena con los datos que haya en `DATA_PATH`: es adecuado para una demostración local, no para exponerlo en Internet.
- El modelo conserva las limitaciones de la Fase 2 (ver `fase-2/README.md`, sección 9): estimaciones para **zonas ya representadas en los datos**, datos de 1990 y valores truncados en 500.001 USD.
- Los modelos se entrenaron con scikit-learn 1.9.1; cargarlos con otra versión puede dar resultados distintos, por eso `requirements.txt` fija las versiones.

## 12. Evidencia de ejecución

Salidas reales de terminal, guardadas en `evidencia/`:

| Archivo | Contenido |
|---|---|
| [`evidencia/pytest.txt`](evidencia/pytest.txt) | `python -m pytest` desde `fase-3/` |
| [`evidencia/docker.txt`](evidencia/docker.txt) | Registro de GitHub Actions: `docker build`, pytest dentro de la imagen, arranque del contenedor, `/health`, el cliente (dos predicciones y una entrada inválida), `/train` y el listado de `fase-2/models` |

Extracto de la salida del cliente contra el contenedor:

```
GET  /health  -> 200 {"status": "ok", "mensaje": "La API esta en ejecucion", "version_configurada": "v1", "modelo_disponible": true}
POST /predict  -> 200 {"prediccion": 431942.36, "version": "v1"}
POST /predict  (fila 2 de ejemplo_prediccion.csv) -> 200 {"prediccion": 388912.02, "version": "v1"}
POST /predict  (sin median_income) -> 422 {"error": "entrada_invalida", "mensaje": "La entrada no es valida. median_income: campo obligatorio ausente", ...}
```

En esa ejecución `housing.csv` no está disponible (no se versiona en GitHub), así que `POST /train` responde **503 `datos_no_disponibles`**, que es el comportamiento esperado y no crea ninguna versión. El entrenamiento por `/train` se comprueba en las pruebas automáticas con datos sintéticos (sección 9) y se repite a mano con el `housing.csv` real en `fase-2/data` (paso 03 de la sección 10).

## 13. Fuentes y asistencia de IA

- Documentación de FastAPI, Pydantic, scikit-learn, Docker y GitHub Actions.
- Código reutilizado: `fase-2/train.py` (`run_training`), `fase-2/src/` y los artefactos de `fase-2/models/`.
- **Asistencia de IA generativa.** La estructura inicial de `api.py`, las pruebas, el cliente, el `Dockerfile` y el flujo de GitHub Actions se elaboró con asistencia de Claude (Anthropic). El equipo revisó el código, lo ejecutó en sus propios entornos, amplió las pruebas y lo comprende; las cifras y salidas de este README provienen de ejecuciones propias registradas en `evidencia/`.
