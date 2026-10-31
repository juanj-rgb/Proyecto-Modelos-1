# 🏡 Predicción de Precios de Vivienda en California

Proyecto Integrador de **Modelos y Simulación de Sistemas I** (Universidad de Antioquia, 2026-II). Pipeline de Machine Learning para predecir el valor mediano de vivienda (`median_house_value`) en distritos de California.

- **Tipo de problema:** regresión supervisada
- **Dataset:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (Kaggle, Cam Nugent)
- **Docente:** Andrés Parra

---

## 📌 Contexto y objetivo

El objetivo es construir un modelo predictivo reproducible y bien documentado, cuyo proceso pueda ejecutarse de nuevo únicamente con el notebook entregado, evitando errores metodológicos como la fuga de información. El proyecto se desarrolla de forma iterativa por Sprints, con control de versiones mediante Git Flow simplificado (`main`, `develop`, `feature/*`) y Pull Requests revisados por otra persona del equipo.

---

## 🗂️ Estructura del repositorio

```text
Proyecto-Modelos-1/
├── fase-1/
│   ├── data/                  # housing.csv (no versionado) y particiones crudas generadas por el notebook
│   ├── resultados/
│   │   ├── metricas_cv.csv    # Selección: validación cruzada sobre entrenamiento (6 candidatos)
│   │   └── metricas.csv       # Evaluación final en prueba, una sola vez (se recrea en cada ejecución)
│   ├── notebook.ipynb         # Notebook principal: EDA, pipeline, selección, evaluación y guardado
│   ├── modelo.joblib          # Pipeline completo entrenado (imputador + codificador + Random Forest)
│   ├── modelo_info.json       # Columnas de entrada, modelo elegido y versiones de las librerías
│   └── README.md              # Documentación específica de la Fase 1
├── requirements.txt           # Dependencias con versiones fijadas
├── .gitignore
└── README.md                  # Este archivo
```

> `housing.csv` no se versiona por su tamaño. Descárgalo de Kaggle y ubícalo en `fase-1/data/housing.csv`.

---

## 📅 Índice de Sprints

| Sprint | Estado | Detalle |
|:---:|:---:|---|
| **Sprint 1** | ✅ Completado (v1.1, corregido tras la retroalimentación) | [Ver detalles](#-sprint-1--modelo-predictivo-fase-1) |
| **Sprint 2** | 🔜 Pendiente | Scripts de entrenamiento y predicción |
| **Sprint 3** | 🔜 Pendiente | Despliegue con Docker y API REST |

---

## 🚀 Sprint 1 — Modelo predictivo (Fase 1)

> **Nota de versión (v1.1).** En la entrega original, los 6 modelos se compararon sobre el conjunto de prueba; en esta versión la selección se hace con validación cruzada sobre el entrenamiento y la prueba se usa una sola vez. Por eso las cifras difieren de las de la entrega original.

### Alcance
El modelo estima el valor mediano de distritos de California **ya representados en los datos** (censo de 1990, que cubre todo el estado); **no pretende generalizar a regiones nuevas**. Ver [limitaciones](#limitaciones).

### Partición y semilla
- 80% entrenamiento / 20% prueba con `train_test_split` y `random_state=42`.
- La partición se hace **antes** de cualquier imputación o codificación.
- El entrenamiento sirve para ajustar y para **elegir** el modelo; la prueba se reserva para una única evaluación final.

### Modelo base
`DummyRegressor(strategy='mean')`: predice siempre la media de `y_train`. Es el piso contra el cual se mide cualquier mejora.

### Preparación de datos (dentro de un `Pipeline`)
| Variable | Problema | Estrategia |
|---|---|---|
| `total_bedrooms` | ~1% de valores nulos | Imputación por **mediana** con `SimpleImputer` (robusta ante la asimetría de la variable) |
| `ocean_proximity` | Categórica nominal, sin orden | **One-Hot Encoding** con `OneHotEncoder(handle_unknown='ignore')`, ajustado solo con entrenamiento |

Se reemplazó `pd.get_dummies(drop_first=True)` porque decide las columnas y la categoría de referencia con cada lote de datos: un registro aislado de `INLAND` quedaba como una fila de ceros, igual que la categoría omitida. El `OneHotEncoder` conserva las categorías aprendidas en entrenamiento.

### Prevención de fuga de información
- `train_test_split` se ejecuta sobre los datos crudos, antes de transformar nada.
- El imputador y el codificador forman parte del `Pipeline` y se ajustan **solo con datos de entrenamiento**, incluso dentro de cada fold de la validación cruzada.
- La variable objetivo nunca se usa como predictora.
- El conjunto de prueba no interviene en ninguna decisión: el modelo se elige por validación cruzada de 5 folds sobre el entrenamiento y la prueba se usa una sola vez.

### Métricas
- **MAE:** error promedio en dólares; es el criterio principal de selección.
- **RMSE:** penaliza con más fuerza los errores grandes.
- **R²:** proporción de la variabilidad del precio explicada por el modelo.

### Resultados

**Selección: validación cruzada de 5 folds sobre el entrenamiento** (ordenada de menor a mayor MAE):

| Modelo | Estrategia / Preprocesamiento | MAE_cv ($) | MAE_std ($) | RMSE_cv ($) | R²_cv |
|---|---|---:|---:|---:|---:|
| **Random Forest** | **Preprocesamiento completo** | **32,309.46** | 548.23 | 49,422.38 | 0.8172 |
| Random Forest | Solo numéricas (sin One-Hot) | 32,769.70 | 512.82 | 50,125.40 | 0.8120 |
| Árbol de Decisión | Preprocesamiento completo | 43,925.54 | 440.21 | 64,061.84 | 0.6929 |
| Regresión Lineal | Preprocesamiento completo | 49,667.16 | 1,005.30 | 68,604.16 | 0.6477 |
| Regresión Lineal | Solo numéricas (sin One-Hot) | 50,674.03 | 1,012.03 | 69,490.01 | 0.6385 |
| Baseline | DummyRegressor (media) | 91,391.47 | 1,008.24 | 115,617.19 | -0.0002 |

**Evaluación final: conjunto de prueba, una sola vez:**

| Modelo | Estrategia | MAE ($) | RMSE ($) | R² |
|---|---|---:|---:|---:|
| **Random Forest** | **Preprocesamiento completo** | **31,636.35** | **48,977.06** | **0.8169** |
| Baseline | DummyRegressor (media) | 90,606.85 | 114,485.64 | -0.0002 |

El baseline se mide en prueba solo como referencia; no participa en la selección.

### Limitaciones
- **Zonas nuevas.** Al validar separando por zonas geográficas (celdas de 1° de latitud y longitud, 55 zonas), el MAE del modelo elegido sube de 32,309.46 a 48,698.39 USD (alrededor de 51% más). El modelo rinde mejor en zonas ya representadas que en zonas que no vio.
- **Tope de 500,001.** La variable objetivo está truncada; no se conoce el valor real de esos distritos. En la prueba, 179 de 4,128 distritos (4.3%) están en el tope: el MAE del modelo es de 73,644.80 USD en ellos frente a 29,732.20 en el resto, y aportan alrededor del 21.5% del error cuadrático. Esto no demuestra que todos los errores grandes sean artefactos de la censura.
- **Datos de 1990.** No reflejan precios actuales.
- **Unidad de análisis.** Cada fila es un distrito, no una vivienda individual.

### Conclusión del Sprint
La validación cruzada eligió el Random Forest con preprocesamiento completo. En la prueba reduce el MAE de 90,606.85 a 31,636.35 USD (65% menos que el baseline) y explica el 81.69% de la variabilidad del precio. Agregar `ocean_proximity` reduce el MAE de validación cruzada un 1.99% en la Regresión Lineal y un 1.40% en el Random Forest: una mejora modesta; en el Random Forest es del mismo orden que la variación entre folds, por lo que no es concluyente. El modelo final se guarda como pipeline completo en `fase-1/modelo.joblib`.

### Uso del modelo guardado
`modelo.joblib` recibe un `DataFrame` **crudo** con las 9 columnas de `fase-1/modelo_info.json` y devuelve el precio estimado en USD. Incluye imputación, codificación y modelo:

```python
import joblib, pandas as pd

modelo = joblib.load('fase-1/modelo.joblib')
nuevo = pd.DataFrame([{
    'longitude': -122.23, 'latitude': 37.88, 'housing_median_age': 41.0,
    'total_rooms': 880.0, 'total_bedrooms': 129.0, 'population': 322.0,
    'households': 126.0, 'median_income': 8.3252, 'ocean_proximity': 'NEAR BAY'}])
print(modelo.predict(nuevo))
```

Acepta valores faltantes en las columnas numéricas y no falla ante categorías no vistas. Se guardó con la versión de scikit-learn registrada en `modelo_info.json`; para cargarlo se recomienda usar esa misma versión.

---

## 🔧 Cómo reproducir

```bash
# 1. Clonar
git clone https://github.com/juanj-rgb/Proyecto-Modelos-1.git
cd Proyecto-Modelos-1

# 2. Entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. Dependencias (versiones fijadas; probado con Python 3.14.5)
pip install -r requirements.txt

# 4. Descargar housing.csv de Kaggle y ubicarlo en fase-1/data/housing.csv

# 5. Ejecutar el notebook
cd fase-1
jupyter notebook notebook.ipynb
# Kernel > Restart & Run All
```

---

## 👥 Equipo

| Integrante | Responsabilidad |
|---|---|
| **Oscar Plaza** | Análisis exploratorio, gestión del flujo principal, control de versiones e integración |
| **Juan José Durango** | Configuración y administración del repositorio, partición de datos y modelo base |
| **Samuel Velásquez** | Lógica de imputación y codificación One-Hot, prevención de fuga de información |