# Fase 4 — Monitoreo: cómo se usa la API y qué versión predice

**Proyecto Integrador — Modelos y Simulación de Sistemas I (2026-II)** · Docente: Andrés Parra

**Integrantes:** Oscar Plaza · Samuel Velásquez · Juan José Durango

**Versión entregada (Entrega 4):** etiqueta de Git `entrega-4` (`git checkout entrega-4` deja el repositorio en el estado evaluado).

## 1. Qué cambia respecto a la Fase 3

`fase-3/` se conserva sin cambios. `fase-4/` contiene la misma API con **el registro de actividad** y el script que calcula las métricas de operación.

| Archivo | Qué hace |
|---|---|
| `api.py` | La API de la Fase 3 más un *middleware* que anota **una fila por cada llamada a `/predict`**, incluidas las rechazadas por validación (las partes nuevas están marcadas con `FASE 4`). |
| `registro.py` | Código del registro: abre el CSV en modo de agregar y escribe el encabezado solo al crearlo. |
| `monitor.py` | Lee el registro y muestra el **total de solicitudes** y el **porcentaje de fallos**. |
| `Dockerfile` | Igual que el de la Fase 3, más `registro.py`, `monitor.py` y `LOG_PATH=/app/logs/solicitudes.csv`. |
| `tests/` | Las pruebas de la API de la Fase 3 más las del registro y del monitoreo. |
| `evidencia/` | Salidas reales de terminal y la muestra del registro utilizado. |

Las rutas, la validación, la configuración (`MODEL_VERSION`, `MODELS_DIR`, `DATA_PATH`…) y el cliente de prueba son los de `fase-3/README.md`.

## 2. Arrancar la API con el registro habilitado

El registro siempre está habilitado: su ubicación la fija la variable **`LOG_PATH`**.

| Entorno | Ubicación del registro persistente |
|---|---|
| Local | `fase-4/logs/solicitudes.csv` (por defecto) |
| Docker | `fase-4/logs/solicitudes.csv` del equipo, montado en `/app/logs` |

**Local** (PowerShell, desde `fase-4/`, con el entorno virtual activo y `pip install -r requirements.txt`):

```powershell
$env:MODEL_VERSION = "v1"                      # versión identificada que usará la API
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

**Docker** (desde la raíz del repositorio):

```powershell
docker build -f fase-4/Dockerfile -t api-vivienda:4.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
docker run --rm -p 8000:8000 `
  -e MODEL_VERSION=v1 `
  -v "${PWD}/fase-2/models:/app/models" `
  -v "${PWD}/fase-2/data:/app/data:ro" `
  -v "${PWD}/fase-4/logs:/app/logs" `
  api-vivienda:4.0
```

La carpeta `fase-4/logs/` se crea sola si no existe. El registro **se conserva entre ejecuciones**: al reiniciar la API se siguen agregando filas al mismo archivo. Git ignora `fase-4/logs/`; la muestra que se entrega está en `evidencia/registro_muestra.csv`.

## 3. Formato del registro

Un CSV con encabezado, una fila por intento sobre `/predict` y solo estos campos:

| Campo | Contenido |
|---|---|
| `fecha_hora` | Hora local del servidor, ISO 8601 sin zona (`2026-11-17T14:27:32`). En Docker es la hora del contenedor (UTC). |
| `estado_http` | Código HTTP devuelto (`200`, `422`, `503`…). |
| `version` | Versión del modelo que se usó. **Vacía** si no se llegó a usar ningún modelo (entrada rechazada por validación, o versión no disponible). |
| `prediccion` | Valor devuelto. **Vacía** si hubo error. |

Ejemplo (ensayo con dos entradas válidas y una con un campo ausente):

```
fecha_hora,estado_http,version,prediccion
2026-11-22T22:56:47,200,v1,431942.36
2026-11-22T22:56:52,200,v1,388912.02
2026-11-22T22:56:56,422,,
```

Solo `/predict` se registra; `/health`, `/versiones` y `/train` no.

## 4. Métricas de operación

```powershell
# desde fase-4/ (usa LOG_PATH o logs\solicitudes.csv)
python monitor.py
python monitor.py --log evidencia\registro_muestra.csv
```

```
Total de solicitudes    : 3
Solicitudes fallidas    : 1 (HTTP 4xx o 5xx)
Porcentaje de fallos    : 33,3 %
Solicitudes por version : v1: 2
```

Cuenta como **fallida** una respuesta HTTP 400 a 599; el porcentaje es `fallidas / total × 100`. Si el registro está vacío (o aún no existe) muestra total 0 y «sin solicitudes» como porcentaje. Este porcentaje describe el **funcionamiento del servicio**, no la calidad de las predicciones: medirla exigiría conocer los resultados reales.

En Docker: `docker run --rm -v "${PWD}/fase-4/logs:/app/logs" api-vivienda:4.0 python monitor.py`.

## 5. Versiones y cómo cambiar de una a otra

| Versión | Qué es | Ubicación |
|---|---|---|
| **`v1`** | **La que usa la API** (`MODEL_VERSION` por defecto). Random Forest de 100 árboles, partición aleatoria. | `fase-2/models/v1/` |
| `v2` | La versión anterior de comparación, de la Fase 2: Random Forest de 50 árboles, partición por zonas. | `fase-2/models/v2/` |

Cada carpeta tiene `modelo.joblib`, `metrics.json` y `metadata.json`. También se listan con `GET /versiones`. Una versión creada con `POST /train` aparece como `fase-2/models/v2026…/`.

**Cambiar de versión y volver a la anterior:** se cambia `MODEL_VERSION` y se reinicia la API. Por ejemplo, para pasar a `v2` y luego regresar a `v1`:

```powershell
$env:MODEL_VERSION = "v2"; python -m uvicorn api:app --host 127.0.0.1 --port 8000     # activa v2
$env:MODEL_VERSION = "v1"; python -m uvicorn api:app --host 127.0.0.1 --port 8000     # vuelve a v1
```

(En Docker, `-e MODEL_VERSION=v2` en el `docker run`.) La versión que respondió cada solicitud queda en la columna `version` del registro.

## 6. Política de reentrenamiento

Cuando se reúnan **500 distritos nuevos con su valor mediano real conocido**, el equipo revisará el modelo: un integrante designado ejecutará `POST /train` sobre el CSV ampliado con 300 de esos casos y reservará los 200 restantes. Comparará la versión candidata y la versión activa con el **MAE** de la Fase 2 sobre esos mismos 200 casos reservados, y **solo activará la candidata** (cambiando `MODEL_VERSION` y reiniciando la API) si su MAE es menor; si no, conservará la versión activa.

Un campo ausente o mal enviado **no** dispara esta revisión: se corrige en quien envía los datos (aparece como 422 en el registro).

## 7. Reporte

*Procedimiento (se ejecuta una vez, sobre un registro vacío):* arrancar la API como en la sección 2 con `MODEL_VERSION=v1`; con el cliente de la Fase 3 enviar dos solicitudes válidas y una con un campo obligatorio ausente; ejecutar `python monitor.py` y comparar con las filas guardadas.

```powershell
python cliente_prueba.py                                                       # válida 1
python cliente_prueba.py --csv ..\fase-2\data\ejemplo_prediccion.csv --fila 2  # válida 2
python cliente_prueba.py --quitar median_income                                # inválida: falta median_income
python monitor.py
```

**Solicitudes enviadas**

| # | Solicitud | Resultado |
|---|---|---|
| 1 | `ejemplo_solicitud.json`: distrito `NEAR BAY` con las 9 variables (`longitude` −122.23, `latitude` 37.88, `median_income` 8.3252…) | **200** · versión `v1` · predicción **431.942,36 USD** |
| 2 | Fila 2 de `fase-2/data/ejemplo_prediccion.csv`: distrito `NEAR BAY`, 7.099 habitaciones, `median_income` 8.3014 | **200** · versión `v1` · predicción **388.912,02 USD** |
| 3 | El registro 1 **sin `median_income`** (`cliente_prueba.py --quitar median_income`) | **422** · `La entrada no es valida. median_income: campo obligatorio ausente` · en el registro queda sin versión ni predicción |

**Indicadores obtenidos** (salida de `monitor.py` sobre `evidencia/registro_muestra.csv`)

| Indicador | Valor |
|---|---|
| Total de solicitudes | 3 |
| Porcentaje de solicitudes fallidas | 33,3 % (1 de 3) |

**Interpretación:** la API procesó dos distritos con la versión `v1` y devolvió 431.942,36 y 388.912,02 USD; rechazó el tercero con **422** porque le faltaba `median_income`, y esa fila quedó en el registro sin versión ni predicción, al no haberse llegado a cargar ningún modelo. La tasa de fallos fue 1/3 (**33,3 %**). El rechazo era el resultado esperado de la prueba de validación, no una falla del servicio: un campo ausente se corrige en quien envía los datos. El indicador describe el funcionamiento del servicio y no la calidad predictiva, que exigiría conocer el valor real de esos distritos.

**Versiones:** la API usó `v1`; la versión anterior de comparación es `v2` (ver sección 5). **Política de reentrenamiento:** la de la sección 6.

La **muestra del registro utilizado** es [`evidencia/registro_muestra.csv`](evidencia/registro_muestra.csv): una copia del `solicitudes.csv` generado en esta ejecución.

## 8. Pruebas

```powershell
# desde fase-4/, con el entorno virtual activo
python -m pytest
```

Resultado: **23 passed**, registrado en [`evidencia/pytest.txt`](evidencia/pytest.txt) (Windows, Python 3.13.7, pytest 9.1.1). Incluyen las tres de la Fase 3 y las del registro: una sola fila por intento (también para la entrada rechazada), el encabezado una sola vez, la conservación del historial entre ejecuciones, el 503, y el cálculo de `monitor.py` (3 solicitudes y 33,3 %; registro vacío → «sin solicitudes»; 4xx y 5xx como fallos).

## 9. Comprobación desde una copia nueva del repositorio

| Paso | Qué hacer |
|---|---|
| 04 | Construir y arrancar como en la sección 2 (Docker). Comando de inicio y ubicación del registro: ahí mismo (`fase-4/logs/solicitudes.csv`). |
| 05 | Enviar dos solicitudes válidas y una con un campo ausente (sección 7) y ejecutar `python monitor.py`; comparar el resultado con las filas de `fase-4/logs/solicitudes.csv`. |
| 06 | Comprobar que el reporte de la sección 7 corresponde a esa ejecución, que la política (sección 6) está escrita y que las versiones `v1` y `v2` se localizan en `fase-2/models/`. |

## 10. Evidencia y limitaciones

| Archivo | Contenido |
|---|---|
| [`evidencia/pytest.txt`](evidencia/pytest.txt) | `python -m pytest` desde `fase-4/` |
| [`evidencia/ejecucion.txt`](evidencia/ejecucion.txt) | arranque de la API, el cliente, el contenido del registro y la salida de `monitor.py` |
| [`evidencia/registro_muestra.csv`](evidencia/registro_muestra.csv) | el registro utilizado |
| [`evidencia/docker.txt`](evidencia/docker.txt) | Demostración en Docker ejecutada en GitHub Actions (flujo `.github/workflows/docker-fase-4.yml`): arranque con el registro montado, ensayo de 3 solicitudes, `monitor.py` (3 solicitudes, 33,3 %) y conservación del registro tras reiniciar el contenedor (4 solicitudes, 25,0 %) |

- El registro es un CSV local: sirve para esta demostración, no para varias réplicas de la API ni para volúmenes grandes.
- Si no se puede escribir en el registro, la API sigue respondiendo y avisa por la salida de error.
- No se mide la calidad predictiva ni hay reentrenamiento automático: la política de la sección 6 solo está documentada.

## 11. Asistencia de IA generativa

La estructura inicial de `registro.py`, `monitor.py`, el *middleware* y las pruebas se elaboró con asistencia de Claude (Anthropic). El equipo revisó el código, lo ejecutó en sus propios entornos y lo comprende; el reporte y la muestra del registro provienen de una ejecución propia.
