# Fase 2 — Scripts y Docker: Predicción de precios de vivienda en California

> **Antes de entregar:** buscar el texto `COMPLETAR` en este archivo (PowerShell: `Select-String -Path README.md -Pattern COMPLETAR`; Bash: `grep -n COMPLETAR README.md`) y reemplazar cada ocurrencia. No debe quedar ninguna.

**Proyecto Integrador — Modelos y Simulación de Sistemas I (2026-II)** · Docente: Andrés Parra

**Integrantes:** Oscar Plaza · Samuel Velásquez · Juan José Durango

**Versión entregada (Entrega 2):** etiqueta de Git `entrega-2` (`git checkout entrega-2` deja el repositorio en el estado evaluado). Es distinta de los identificadores de modelo (`v1`, `v2`) de `models/`. La Fase 2 parte de la etiqueta `fase-1-v1.1` (Fase 1 corregida, commit `08199fd` en `main`).

## 1. Problema y relación con la Fase 1

Se predice el **valor mediano de vivienda (`median_house_value`, en USD) de un distrito de California** a partir de variables demográficas y geográficas (problema de **regresión**). Se conservan el problema, los datos aprobados y el enfoque de la Fase 1: la carpeta `fase-1/` se conserva **sin modificar en esta fase** (su versión corregida es la etiqueta `fase-1-v1.1`). La Fase 2 convierte el notebook en una aplicación que se ejecuta por línea de comandos:

- `train.py`: entrena y guarda una **versión** del modelo (`models/<version>/`).
- `predict.py`: carga una versión guardada y predice sobre un CSV nuevo, sin reentrenar.
- `Dockerfile`: ejecuta ambos scripts en un contenedor.
- `tests/`: pruebas automatizadas con pytest.

`fase-1/modelo.joblib` y `fase-1/modelo_info.json` pertenecen a la Fase 1: **no son una versión de la Fase 2** y `predict.py` no los acepta (necesita el `metadata.json` que genera `train.py`).

El pipeline de `fase-2` es el mismo de la Fase 1 v1.1 (`SimpleImputer` por mediana en las 8 variables numéricas + `OneHotEncoder(handle_unknown='ignore')` en `ocean_proximity` + Random Forest de 100 árboles, `random_state=42`), con la misma partición 80/20 (`random_state=42`) y la misma validación cruzada (5 folds, `KFold` con mezcla y semilla 42). Por eso `v1` debe reproducir las cifras de la Fase 1 v1.1 (sección 9).

## 2. Correcciones respecto a la entrega anterior (Fase 1)

Las correcciones de la retroalimentación se aplicaron primero en `fase-1/` (versión `fase-1-v1.1`, PR #9 a #12) y se trasladan a esta aplicación:

| Observación de la retroalimentación | Qué se hizo |
|---|---|
| La prueba (`X_test`) participaba en la selección del modelo | La configuración se elige con **validación cruzada solo sobre entrenamiento** y la prueba se usa **una sola vez**, para evaluar. En `train.py`, el algoritmo es un parámetro (`--algorithm`); la validación cruzada se guarda en `metrics.json` (`validacion_cruzada_entrenamiento`) y es la cifra para comparar configuraciones. |
| `get_dummies(drop_first=True)` por lote cambiaba la categoría de referencia | `OneHotEncoder(handle_unknown='ignore')` dentro de un `Pipeline`, ajustado solo con entrenamiento. Una categoría no vista se codifica con ceros y se avisa. |
| `modelo.joblib` guardaba solo el bosque | Se guarda el **`Pipeline` completo** (imputación + codificación + modelo). `predict.py` acepta datos crudos. |
| Dependencias sin versiones | `requirements.txt` con versiones fijas (el de `fase-2/` es mínimo, sin jupyter, matplotlib ni seaborn); las versiones usadas quedan en `metadata.json`. |
| Independencia espacial no discutida | El alcance declarado es estimar distritos en **zonas ya representadas en los datos**; no se pretende generalizar a regiones nuevas. En la Fase 1 v1.1 se midió la sensibilidad con validación por zonas: MAE de validación cruzada de 32.309,46 (aleatoria) a 48.698,39 (por zonas), **+51 %**. En esta fase la opción `--spatial-split` separa por zonas (celdas de 1° de latitud/longitud) y la versión `v2` la usa. |
| Afirmaciones sobre el tope de 500.001 USD | En la Fase 1 v1.1 se compararon los errores con y sin tope (ver sección 9) y se matizaron las afirmaciones. **No se excluyen ni se tratan** esos registros; `fase-2` no repite ese diagnóstico. |
| Nombre del tercer integrante inconsistente | Unificado como **Juan José Durango** en el notebook y en los README. |
| Pull Requests sin revisión formal | En la Fase 1 v1.1 los PR #9 a #12 se fusionaron con aprobación de otro integrante. En esta fase, cada PR hacia `develop` lo aprobó un integrante distinto de su autor. COMPLETAR: confirmar este enunciado con la lista de PR de GitHub antes de entregar y ajustarlo si no es cierto. |

## 3. Entorno

**Python utilizado: 3.14.5** (entorno local de Oscar, con el que se entrenaron los modelos entregados). La imagen Docker usa `python:3.14-slim`. Las mismas versiones de `requirements.txt` se verificaron además con Python 3.12 y con Python 3.14.10 (las 8 pruebas pasan en ambas).

```powershell
cd fase-2
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell. En Bash: source venv/bin/activate
pip install -r requirements.txt
```

`fase-2/requirements.txt` contiene `numpy`, `pandas`, `scikit-learn`, `joblib`, `scipy`, `threadpoolctl` y `pytest`, con versiones fijas. El `requirements.txt` de la raíz del repositorio es el de la Fase 1 (incluye jupyter, matplotlib y seaborn, que solo necesita el notebook).

**Todos los comandos de este README se ejecutan desde la carpeta `fase-2/`**, con rutas relativas.

## 4. Datos

- **Fuente:** [California Housing Prices (Kaggle, Cam Nugent)](https://www.kaggle.com/datasets/camnugent/california-housing-prices). El archivo completo **no se versiona** en GitHub (los `*.csv` están en `.gitignore`).
- **Cómo obtenerlo:** descargar `housing.csv` desde la página de Kaggle (botón *Download*) y copiarlo a **`fase-2/data/housing.csv`**. Quien ya lo tiene en la Fase 1 puede copiarlo: `Copy-Item ..\fase-1\data\housing.csv data\housing.csv` (PowerShell, desde `fase-2/`). No se requieren credenciales en el repositorio.
- **Archivo de ejemplo para predecir:** `data/ejemplo_prediccion.csv` (5 distritos reales, sin la variable objetivo; sí se versiona).

**Esquema del CSV de entrenamiento:** las 9 columnas de entrada de la tabla siguiente más `median_house_value` (objetivo). Son las mismas 9 columnas, en el mismo orden, que `fase-1/modelo_info.json` (`columnas_entrada`).

**Esquema del CSV de predicción** (el orden de las columnas no importa; la objetivo no es necesaria):

| Columna | Tipo | Nulos |
|---|---|---|
| `longitude`, `latitude` | numérico | se imputa con la mediana del entrenamiento |
| `housing_median_age`, `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income` | numérico | se imputa con la mediana del entrenamiento |
| `ocean_proximity` | texto (`<1H OCEAN`, `INLAND`, `ISLAND`, `NEAR BAY`, `NEAR OCEAN`) | se imputa con la moda del entrenamiento |

Comportamiento de `predict.py`: **columna requerida ausente** o **valor no numérico en una columna numérica** → error con mensaje claro, código de salida 1 y **no se genera archivo de salida**. **Columnas adicionales** → se ignoran con advertencia. **Categoría no vista** → todas las columnas one-hot en cero, con advertencia. Una celda vacía **no** es una columna ausente: se imputa. Se devuelve una predicción por fila, **en el orden original**, sin eliminar observaciones. Se validan columnas y tipos; **no se validan rangos de valores**.

## 5. Uso (ejecutar desde `fase-2/`, cada comando en una sola línea)

Ayuda de cada script:

```bash
python train.py --help
python predict.py --help
```

Entrenar una versión:

```bash
python train.py --data data/housing.csv --target median_house_value --output-dir models --version v1 --seed 42
```

Entrenar una segunda versión (partición por zonas, 50 árboles):

```bash
python train.py --data data/housing.csv --target median_house_value --output-dir models --version v2 --seed 42 --spatial-split --n-estimators 50
```

Predecir con una versión (se elige la versión con `--model`; también acepta `models/v1`):

```bash
python predict.py --model models/v1/modelo.joblib --input data/ejemplo_prediccion.csv --output outputs/predicciones.csv
```

Argumentos de `train.py`: `--data`, `--target`, `--output-dir`, `--version`, `--seed`, `--test-size` (0.2), `--spatial-split`, `--algorithm` (`random_forest`, `decision_tree`, `linear_regression`), `--n-estimators`, `--max-depth`, `--cv-folds` (5; 0 la desactiva). Repetir un `--version` existente **falla sin sobrescribir**; si se omite, se genera un identificador con fecha y hora.

## 6. Versiones, métricas y resultados

```
fase-2/
├── train.py  predict.py  requirements.txt  Dockerfile  .dockerignore  pytest.ini  README.md
├── src/          # lógica reutilizable: config, data, preprocessing, modeling, versioning, errors
├── tests/        # test_aplicacion.py (pruebas con pytest)
├── data/         # ejemplo_prediccion.csv (y housing.csv, no versionado)
├── models/
│   ├── v1/       # modelo.joblib · metrics.json · metadata.json
│   └── v2/       # modelo.joblib · metrics.json · metadata.json
├── evidencia/    # salidas de terminal: pytest, ejecución local y Docker
└── outputs/      # resultados generados (no versionado)
```

- `modelo.joblib`: `Pipeline` completo (preprocesamiento ajustado + estimador).
- `metrics.json`: métricas del modelo y del modelo base en prueba, y validación cruzada de entrenamiento.
- `metadata.json`: versión, fecha, algoritmo, parámetros, semilla, esquema de entrada y tipos, datos usados (archivo, huella SHA-256, nº de filas), partición, métrica, versión de Python y librerías, y referencia al commit del código.

Cada `modelo.joblib` pesa unos 15 a 30 MB y se versiona en Git junto con las métricas y los metadatos.

## 7. Docker

Imagen base `python:3.14-slim`. El entrenamiento **no** ocurre en `docker build`; se ejecuta al iniciar el contenedor. Los datos se montan desde el equipo y los modelos y resultados **persisten fuera del contenedor**. Terminal usada por el equipo: COMPLETAR (Bash / PowerShell).

**PowerShell** (desde `fase-2/`):

```powershell
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
New-Item -ItemType Directory -Force models, outputs | Out-Null

docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models" `
  modelos-fase2:1.0 python train.py `
  --data data/housing.csv --target median_house_value `
  --output-dir models --version v-docker --seed 42

docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/models:/app/models:ro" `
  -v "${PWD}/outputs:/app/outputs" `
  modelos-fase2:1.0 python predict.py `
  --model models/v-docker/modelo.joblib `
  --input data/ejemplo_prediccion.csv `
  --output outputs/predicciones.csv
```

**Bash** (desde `fase-2/`):

```bash
docker build -t modelos-fase2:1.0 --build-arg GIT_COMMIT=$(git rev-parse HEAD) .
mkdir -p models outputs

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  modelos-fase2:1.0 python train.py \
  --data data/housing.csv --target median_house_value \
  --output-dir models --version v-docker --seed 42

docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models:ro" \
  -v "$PWD/outputs:/app/outputs" \
  modelos-fase2:1.0 python predict.py \
  --model models/v-docker/modelo.joblib \
  --input data/ejemplo_prediccion.csv \
  --output outputs/predicciones.csv
```

Las pruebas también se pueden correr en la imagen: `docker run --rm modelos-fase2:1.0 python -m pytest`.

## 8. Pruebas (pytest)

```bash
python -m pytest
```

Usan datos sintéticos pequeños, directorios temporales y no requieren red ni el dataset completo.

| Prueba | Qué comprueba |
|---|---|
| `test_entrenamiento_genera_artefactos_cargables` | El entrenamiento crea `modelo.joblib`, `metrics.json` y `metadata.json`; el modelo carga, mejora al base y los metadatos son correctos. |
| `test_carga_y_prediccion_conserva_resultados_del_modelo` | El modelo recargado da las mismas predicciones que antes de guardarlo; `predict.py` devuelve una predicción por fila y conserva el orden. |
| `test_valores_faltantes_se_imputan_segun_la_estrategia` | Un nulo se imputa con la mediana calculada **solo con entrenamiento**; no se elimina ninguna fila. |
| `test_columna_ausente_produce_error_claro_y_codigo_distinto_de_cero` | Una columna ausente da un error que la nombra; por línea de comandos el código de salida es ≠ 0 y no se crea archivo. |
| `test_segunda_version_conserva_la_primera_y_no_se_sobrescribe` | Crear `v2` no altera `v1`; repetir un identificador falla sin tocar los archivos. |
| `test_categoria_no_vista_y_columnas_extra_no_rompen` | Categorías no vistas y columnas extra generan advertencia, no error; el orden de las columnas del CSV no importa. |
| `test_valor_no_numerico_en_columna_numerica_es_error` | Un valor no numérico en una columna numérica es un error y no genera salida. |
| `test_help_funciona_en_ambos_scripts` | `--help` funciona en `train.py` y `predict.py`. |

## 9. Resultados, comparación con el modelo base y limitaciones

**Métrica principal: MAE** (error medio en USD, directo de interpretar). Se reportan también RMSE (penaliza errores grandes) y R².

| Versión | Algoritmo y partición | MAE prueba (USD) | RMSE prueba (USD) | R² prueba | MAE modelo base | MAE val. cruzada (train) |
|---|---|---:|---:|---:|---:|---:|
| `v1` | Random Forest (100 árboles), aleatoria | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR |
| `v2` | Random Forest (50 árboles), por zonas | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR | COMPLETAR |

Copiar las cifras de `models/v1/metrics.json` y `models/v2/metrics.json`. **El modelo base** (predecir siempre la media del entrenamiento) usa la misma partición y las mismas métricas. COMPLETAR: una o dos frases comparando el modelo con el base (cuánto baja el MAE).

**Referencia de la Fase 1 v1.1** (misma partición, semilla y pipeline): Random Forest con preprocesamiento completo, MAE de prueba 31.636,35, RMSE 48.977,06 y R² 0,8169; MAE de validación cruzada 32.309,46; modelo base con MAE 90.606,85. `v1` usa el mismo pipeline y la misma partición, por lo que con los mismos datos y las mismas versiones de librerías se espera que reproduzca esas cifras. COMPLETAR: indicar si coinciden y, si no, explicar la diferencia.

**`v1` y `v2` no son comparables entre sí**: usan particiones distintas y miden escenarios distintos (zonas conocidas vs. zonas nuevas). Que `v2` tenga un error mayor o menor que `v1` es información sobre la sensibilidad espacial, no sobre cuál modelo es “mejor”. COMPLETAR: una frase interpretando la diferencia observada.

**Alcance y limitaciones conocidas**

- **Alcance:** estimar el valor mediano de distritos en **zonas ya representadas en los datos**. No se pretende generalizar a regiones nuevas: en la Fase 1 v1.1 el error de validación cruzada subió un 51 % al separar por zonas (32.309,46 a 48.698,39).
- Los datos son de **1990** y cada fila es un **distrito**, no una vivienda: no representan precios actuales.
- El valor objetivo está **truncado en 500.001 USD**, de modo que no se conoce el valor real de esos distritos. En la Fase 1 v1.1 (prueba, modelo final) el MAE fue de 73.644,8 USD en los 179 distritos con tope (4,3 % de la prueba; 786 en entrenamiento, 4,8 %) frente a 29.732,2 USD en el resto, y esos 179 distritos aportaron el 21,5 % del error cuadrático. No se excluyeron ni se trataron.
- Distritos vecinos se parecen entre sí; por eso la partición aleatoria no mide el desempeño en regiones nuevas.
- `ISLAND` es una categoría muy poco frecuente: el modelo aprende muy poco sobre ella.
- No se evaluó ajuste de hiperparámetros ni ingeniería de variables.
- Random Forest usa componentes aleatorios: se fija la semilla (`--seed`), y con los mismos datos, parámetros y versiones de librerías se esperan los mismos resultados; no se exige igualdad exacta entre sistemas distintos.

## 10. Evidencia de ejecución

Salidas reales de terminal, guardadas en `evidencia/`:

| Archivo | Contenido |
|---|---|
| [`evidencia/pytest.txt`](evidencia/pytest.txt) | `python -m pytest` (resultado de las pruebas) |
| [`evidencia/local.txt`](evidencia/local.txt) | entrenamiento y predicción en el entorno local, y el error de una entrada inválida con su código de salida |
| [`evidencia/docker.txt`](evidencia/docker.txt) | `docker build`, entrenamiento y predicción dentro del contenedor, y el listado de `models/` y `outputs/` en el equipo anfitrión mostrando que los resultados persisten |

COMPLETAR: pegar aquí un extracto breve (3-5 líneas) de la salida del entrenamiento en Docker.

## 11. Fuentes consultadas y código externo adaptado

- Documentación de scikit-learn (Pipeline, ColumnTransformer, SimpleImputer, OneHotEncoder, GroupShuffleSplit, cross_validate).
- Dataset: California Housing Prices (Kaggle, Cam Nugent).
- La lógica del pipeline se adaptó del notebook de la Fase 1 v1.1 (`fase-1/notebook.ipynb`).
- COMPLETAR: declarar el uso de asistencia de IA generativa si la política del curso lo exige (por ejemplo: *“La estructura inicial de train.py, predict.py y las pruebas se elaboró con asistencia de Claude; el equipo revisó, ejecutó y comprende el código.”*).
