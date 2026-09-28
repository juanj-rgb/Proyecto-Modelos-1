# Fase 1 — Modelo Predictivo: Precios de Vivienda en California

Proyecto Integrador · Modelos y Simulación de Sistemas I · Universidad de Antioquia · 2026-II

## Integrantes

| Integrante | Responsabilidad principal | Modelo |
|---|---|---|
| Oscar Plaza | Análisis exploratorio | Regresión Lineal |
| Samuel Velásquez | Preparación de datos y prevención de fuga de información | Árbol de Decisión |
| Juan Rincón | Partición, modelo base y administración del repositorio | Random Forest |

## Descripción del problema

Se busca estimar el valor mediano de las viviendas de un distrito censal de California a partir de características demográficas y geográficas del distrito: ubicación, antigüedad de las viviendas, número de habitaciones, población, hogares, ingreso mediano y cercanía al océano.

Es un problema de regresión supervisada. La variable objetivo, `median_house_value`, es numérica y continua.

Cada observación corresponde a un distrito completo, no a una vivienda individual, lo que impone un límite a la precisión alcanzable: el modelo no dispone de atributos del inmueble concreto.

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
| Valores faltantes en `total_bedrooms` | Imputación por mediana | La distribución es asimétrica a la derecha; la mediana es robusta a los valores extremos. Eliminar filas descartaría distritos completos sin razón que lo justifique. |
| Variable categórica `ocean_proximity` | One-Hot Encoding con `drop_first=True` | Es una variable nominal sin orden natural. La categoría omitida actúa como nivel de referencia y evita colinealidad perfecta. |
| Partición de los datos | 80/20, aleatoria, `random_state=42` | Fija antes de cualquier transformación, de modo que el conjunto de prueba no influya en los parámetros aprendidos. |

Todas las transformaciones que aprenden parámetros de los datos se ajustan únicamente con el conjunto de entrenamiento y se aplican al de prueba. La sección 5 del notebook detalla las cinco verificaciones de prevención de fuga de información.

## Algoritmo utilizado

Se entrenaron y compararon tres algoritmos: Regresión Lineal, Árbol de Decisión y Random Forest. Para la Regresión Lineal y el Random Forest se entrenó además una versión sin las columnas One-Hot, con el fin de medir el aporte real de `ocean_proximity`.

El modelo final almacenado en `modelo.joblib` es el Random Forest con preprocesamiento completo, por ser el de menor error absoluto medio.

## Métrica empleada

Se reportan MAE, RMSE y R², y se utiliza el MAE como criterio de selección.

El MAE está expresado en dólares y se interpreta directamente como el error promedio del modelo. Al no elevar los errores al cuadrado, no queda dominado por los distritos cuyo precio aparece censado en el valor máximo del censo, donde el error grande es un artefacto de los datos y no un fallo del modelo. El RMSE se reporta porque sí es sensible a esos casos extremos, y el R² permite comparar contra el modelo base en una escala independiente de las unidades.

## Principales resultados

| Modelo | Estrategia | Integrante | MAE | RMSE | R2 |
|---|---|---|---|---|---|
| Random Forest | Preprocesamiento completo | Juan Rincón | 31.639,71 | 49.036,98 | 0,8165 |
| Random Forest | Solo numéricas | Juan Rincón | 32.098,51 | 49.887,18 | 0,8101 |
| Árbol de Decisión | Preprocesamiento completo | Samuel | 42.866,15 | 62.817,98 | 0,6989 |
| Regresión Lineal | Preprocesamiento completo | Oscar | 50.670,49 | 70.059,19 | 0,6254 |
| Regresión Lineal | Solo numéricas | Oscar | 51.810,09 | 71.131,26 | 0,6139 |
| Baseline | DummyRegressor (media) | Juan Rincón | 90.606,85 | 114.485,64 | -0,0002 |

Ordenados de menor a mayor MAE. El modelo final es el Random Forest con preprocesamiento completo.

El modelo base, que predice siempre la media, obtiene un R² de -0,0002: como corresponde a un modelo que no usa ninguna variable predictora, no explica nada de la variabilidad del precio. Todos los modelos entrenados lo superan con amplitud.

El Random Forest con preprocesamiento completo es el mejor: reduce el error absoluto medio de 90.606,85 a 31.639,71 dólares, es decir un 65% menos que el modelo base, y explica el 81,65% de la variabilidad. La Regresión Lineal se queda en un R² de 0,6254, y el Árbol de Decisión alcanza 0,6989. Que los dos modelos basados en árboles superen claramente al lineal indica que la relación entre las predictoras y el precio no es lineal, algo que ya sugería el diagrama de dispersión de `median_income`, donde la nube se curva y se satura en el valor máximo.

La comparación con y sin One-Hot cuantifica el aporte de `ocean_proximity`: reduce el MAE un 2,20% en la Regresión Lineal y un 1,43% en el Random Forest. Es una mejora modesta pero consistente en ambos modelos, lo que confirma que la cercanía al océano aporta información que las coordenadas geográficas por sí solas no capturan del todo.

## Instrucciones para ejecutar el notebook

Desde la carpeta `fase-1/`:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r ../requirements.txt
```

Descargar `housing.csv` de Kaggle y ubicarlo en `fase-1/data/housing.csv`.

Abrir `notebook.ipynb`, seleccionar el kernel del entorno virtual y ejecutar Restart & Run All. El notebook corre de principio a fin sin intervención manual y genera:

| Archivo | Contenido |
|---|---|
| `data/X_train.csv`, `data/X_test.csv` | Conjuntos ya procesados, reutilizables sin repetir la preparación |
| `data/X_train_raw.csv`, `data/X_test_raw.csv`, `data/y_train.csv`, `data/y_test.csv` | Particiones crudas |
| `resultados/metricas.csv` | Tabla comparativa de los seis modelos evaluados |
| `modelo.joblib` | Modelo final entrenado |

El archivo de métricas se recrea en cada ejecución, de modo que correr el notebook dos veces seguidas no duplica filas.

## Estructura de la carpeta

```
fase-1/
├── data/
│   └── housing.csv          (no versionado; descargar de Kaggle)
├── resultados/
│   └── metricas.csv
├── notebook.ipynb
├── modelo.joblib
└── README.md
```
