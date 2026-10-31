# Fase 1 — Modelo Predictivo: Precios de Vivienda en California

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

> **Versión 1.1.** Corrige la metodología según la retroalimentación del docente. En la entrega original, los 6 modelos se compararon sobre el conjunto de prueba; en esta versión la selección se hace con validación cruzada sobre el entrenamiento y la prueba se usa una sola vez. Por eso las cifras difieren de las de la entrega original.

## Integrantes

| Integrante | Responsabilidad principal | Modelo |
|---|---|---|
| Oscar Plaza | Análisis exploratorio | Regresión Lineal |
| Samuel Velásquez | Preparación de datos y prevención de fuga de información | Árbol de Decisión |
| Juan José Durango | Partición, modelo base y administración del repositorio | Random Forest |

## Descripción del problema

Se busca estimar el valor mediano de las viviendas de un distrito censal de California a partir de características demográficas y geográficas del distrito: ubicación, antigüedad de las viviendas, número de habitaciones, población, hogares, ingreso mediano y cercanía al océano.

Es un problema de regresión supervisada. La variable objetivo, `median_house_value`, es numérica y continua.

Cada observación corresponde a un distrito completo, no a una vivienda individual, lo que impone un límite a la precisión alcanzable: el modelo no dispone de atributos del inmueble concreto.

## Alcance del modelo

El modelo estima el valor mediano de distritos de California **ya representados en los datos** (censo de 1990, que cubre todo el estado); **no pretende generalizar a regiones nuevas**. Que cada fila sea un distrito distinto no demuestra independencia espacial: distritos vecinos comparten características y precios, y una partición aleatoria puede dejar vecinos en entrenamiento y en prueba. Por eso la partición aleatoria se justifica solo para este alcance, y se midió cuánto cambia el error al separar por zonas (ver Limitaciones).

## Fuente de los datos

California Housing Prices, de Cam Nugent, publicado en Kaggle y derivado del censo de California de 1990.

https://www.kaggle.com/datasets/camnugent/california-housing-prices

| Característica | Valor |
|---|---|
| Observaciones | 20.640 |
| Variables predictoras | 9 |
| Variable objetivo | `median_house_value` |
| Valores faltantes | `total_bedrooms`, aproximadamente 1% |
| Variable categórica | `ocean_proximity` |

El archivo `housing.csv` no se versiona en el repositorio por su tamaño; debe descargarse de Kaggle y ubicarse en `fase-1/data/housing.csv`.

## Objetivo del modelo

Construir un modelo reproducible que prediga `median_house_value` con un error medio sustancialmente menor al de un modelo base trivial, documentando y justificando cada decisión de preparación y evitando la fuga de información.

## Preparación de los datos

| Decisión | Estrategia adoptada | Justificación |
|---|---|---|
| Valores faltantes en `total_bedrooms` | Imputación por mediana (`SimpleImputer`, dentro de un `Pipeline`) | La distribución es asimétrica a la derecha; la mediana es robusta a los valores extremos. Eliminar filas descartaría distritos completos sin razón que lo justifique. |
| Variable categórica `ocean_proximity` | One-Hot Encoding con `OneHotEncoder(handle_unknown='ignore')` | Es una variable nominal sin orden natural. Se descartó `get_dummies(drop_first=True)` porque decide las columnas y la categoría de referencia con cada lote: un registro aislado de `INLAND` quedaba como una fila de ceros, igual que la categoría omitida. El `OneHotEncoder` conserva las categorías aprendidas en entrenamiento. |
| Partición de los datos | 80/20, aleatoria, `random_state=42` | Fija antes de cualquier transformación, de modo que el conjunto de prueba no influya en los parámetros aprendidos. |

Imputación y codificación forman parte de un `Pipeline` y se ajustan únicamente con datos de entrenamiento, también dentro de cada fold de la validación cruzada. La sección 5 del notebook detalla las verificaciones de prevención de fuga de información.

## Algoritmo utilizado

Se compararon tres algoritmos: Regresión Lineal, Árbol de Decisión y Random Forest, además del modelo base. Para la Regresión Lineal y el Random Forest se evaluó también una versión sin las columnas One-Hot, con el fin de medir el aporte real de `ocean_proximity`.

**Selección.** Cada candidato es un `Pipeline` completo y se evalúa con validación cruzada de 5 folds (`KFold` con mezcla y `random_state=42`) usando solo el conjunto de entrenamiento. Se elige el de menor MAE medio de validación cruzada. **Evaluación final.** El modelo elegido se reajusta con todo el entrenamiento y se mide una sola vez en el conjunto de prueba.

El modelo almacenado en `modelo.joblib` es el Random Forest con preprocesamiento completo, elegido por validación cruzada.

## Métrica empleada

Se reportan MAE, RMSE y R², y se utiliza el MAE como criterio de selección.

El MAE está expresado en dólares y se interpreta directamente como el error promedio del modelo. Al no elevar los errores al cuadrado, es menos sensible a unos pocos errores muy grandes, como los que podrían darse en distritos con el precio truncado en el valor máximo del censo. El RMSE se reporta porque sí es sensible a esos casos, y el R² permite comparar contra el modelo base en una escala independiente de las unidades.

## Principales resultados

**Selección: validación cruzada de 5 folds sobre el entrenamiento** (de menor a mayor MAE):

| Modelo | Estrategia | MAE_cv | MAE_std | RMSE_cv | R2_cv |
|---|---|---|---|---|---|
| Random Forest | Preprocesamiento completo | 32.309,46 | 548,23 | 49.422,38 | 0,8172 |
| Random Forest | Solo numéricas | 32.769,70 | 512,82 | 50.125,40 | 0,8120 |
| Árbol de Decisión | Preprocesamiento completo | 43.925,54 | 440,21 | 64.061,84 | 0,6929 |
| Regresión Lineal | Preprocesamiento completo | 49.667,16 | 1.005,30 | 68.604,16 | 0,6477 |
| Regresión Lineal | Solo numéricas | 50.674,03 | 1.012,03 | 69.490,01 | 0,6385 |
| Baseline | DummyRegressor (media) | 91.391,47 | 1.008,24 | 115.617,19 | -0,0002 |

**Evaluación final: conjunto de prueba, una sola vez:**

| Modelo | Estrategia | MAE | RMSE | R2 |
|---|---|---|---|---|
| Random Forest | Preprocesamiento completo | 31.636,35 | 48.977,06 | 0,8169 |
| Baseline | DummyRegressor (media) | 90.606,85 | 114.485,64 | -0,0002 |

El baseline se mide en prueba solo como referencia; no participa en la selección.

El modelo base, que predice siempre la media, obtiene un R² de -0,0002: como corresponde a un modelo que no usa ninguna variable predictora, no explica nada de la variabilidad del precio. Todos los modelos entrenados lo superan con amplitud.

El Random Forest con preprocesamiento completo es el mejor: en la prueba reduce el error absoluto medio de 90.606,85 a 31.636,35 dólares, es decir un 65% menos que el modelo base, y explica el 81,69% de la variabilidad. El resultado en prueba es coherente con el de la validación cruzada (32.309,46). La Regresión Lineal queda en un R² de validación cruzada de 0,6477 y el Árbol de Decisión en 0,6929. Que los modelos basados en árboles superen claramente al lineal es compatible con una relación no lineal entre las predictoras y el precio, algo que ya sugería el diagrama de dispersión de `median_income`.

La comparación con y sin One-Hot cuantifica el aporte de `ocean_proximity`: reduce el MAE de validación cruzada un 1,99% en la Regresión Lineal y un 1,40% en el Random Forest. Es una mejora modesta; en el Random Forest la diferencia (unos 460 dólares) es del mismo orden que la variación entre folds, por lo que no permite afirmar una mejora clara. Es compatible con que `latitude` y `longitude` ya recojan buena parte de esa señal.

## Limitaciones

- **Zonas nuevas.** Se repitió la validación cruzada del modelo elegido separando por zonas geográficas (celdas de 1 grado de latitud y longitud, 55 zonas en entrenamiento). El MAE sube de 32.309,46 con partición aleatoria a 48.698,39 con partición por zonas, alrededor de un 51% más. El modelo rinde mejor en zonas ya representadas que en zonas que no vio, por lo que no debe usarse para estimar precios en regiones nuevas.
- **Tope de 500.001.** La variable objetivo está truncada en 500.001, por lo que no se conoce el valor real de esos distritos. En la prueba hay 179 distritos con tope (4,3% de 4.128): el MAE del modelo es de 73.644,80 dólares en ellos frente a 29.732,20 en el resto, y el modelo predice por debajo del tope en todos. Aportan alrededor del 21,5% del error cuadrático. Esto no demuestra que todos los errores grandes sean artefactos de la censura.
- **Datos de 1990.** No reflejan precios actuales.
- **Unidad de análisis.** Cada fila es un distrito, no una vivienda individual.

## Uso del modelo guardado

`modelo.joblib` es un `Pipeline` completo (imputador, codificador y Random Forest). Recibe un `DataFrame` **crudo** con las 9 columnas listadas en `modelo_info.json` y devuelve el precio estimado en dólares:

```python
import joblib, pandas as pd

modelo = joblib.load('fase-1/modelo.joblib')
nuevo = pd.DataFrame([{
    'longitude': -122.23, 'latitude': 37.88, 'housing_median_age': 41.0,
    'total_rooms': 880.0, 'total_bedrooms': 129.0, 'population': 322.0,
    'households': 126.0, 'median_income': 8.3252, 'ocean_proximity': 'NEAR BAY'}])
print(modelo.predict(nuevo))
```

Acepta valores faltantes en las columnas numéricas y no falla ante categorías de `ocean_proximity` no vistas. El notebook comprueba estos casos al recargar el modelo (sección 9).

`modelo_info.json` registra el modelo elegido, las columnas de entrada, el objetivo y las versiones de scikit-learn, pandas y numpy con las que se guardó.

## Versiones y entorno

`requirements.txt` fija la versión exacta de cada dependencia. El proyecto se probó con Python 3.14.5 y scikit-learn 1.9.1. Para cargar `modelo.joblib` se recomienda usar la misma versión de scikit-learn registrada en `modelo_info.json`.

## Instrucciones para ejecutar el notebook

Desde la raíz del repositorio:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Descargar `housing.csv` de Kaggle y ubicarlo en `fase-1/data/housing.csv`.

Abrir `fase-1/notebook.ipynb`, seleccionar el kernel del entorno virtual y ejecutar Restart & Run All. El notebook corre de principio a fin sin intervención manual y genera:

| Archivo | Contenido |
|---|---|
| `data/X_train_raw.csv`, `data/X_test_raw.csv`, `data/y_train.csv`, `data/y_test.csv` | Particiones crudas (no versionadas) |
| `resultados/metricas_cv.csv` | Selección: validación cruzada de los seis candidatos |
| `resultados/metricas.csv` | Evaluación final en prueba: baseline y modelo elegido |
| `modelo.joblib` | Pipeline completo entrenado |
| `modelo_info.json` | Columnas de entrada, modelo elegido y versiones |

Los archivos de métricas se recrean en cada ejecución, de modo que correr el notebook dos veces seguidas no duplica filas. Ya no se generan `X_train.csv` ni `X_test.csv`: la preparación vive dentro del pipeline.

## Estructura de la carpeta

```
fase-1/
├── data/
│   └── housing.csv          (no versionado; descargar de Kaggle)
├── resultados/
│   ├── metricas.csv
│   └── metricas_cv.csv
├── notebook.ipynb
├── modelo.joblib
├── modelo_info.json
└── README.md
```