# 🏡 Predicción de Precios de Vivienda en California

Proyecto Integrador de **Modelos y Simulación de Sistemas I** (Universidad de Antioquia, 2026-II). Pipeline de Machine Learning para predecir el valor mediano de vivienda (`median_house_value`) en distritos de California.

- **Tipo de problema:** regresión supervisada
- **Dataset:** [California Housing Prices](https://www.kaggle.com/datasets/camnugent/california-housing-prices) (Kaggle, Cam Nugent)
- **Docente:** Andrés Parra

---

## 📌 Contexto y objetivo

El objetivo es construir un modelo predictivo reproducible y bien documentado, cuyo proceso pueda ejecutarse de nuevo únicamente con el notebook entregado, evitando errores metodológicos como la fuga de información. El proyecto se desarrolla de forma iterativa por Sprints, con control de versiones mediante Git Flow simplificado (`main`, `develop`, `feature/*`) y Pull Requests.

---

## 🗂️ Estructura del repositorio

```text
Proyecto-Modelos-1/
├── fase-1/
│   ├── data/                  # housing.csv (no versionado) y particiones generadas por el notebook
│   ├── resultados/
│   │   └── metricas.csv       # Métricas de los 6 modelos (se recrea en cada ejecución)
│   ├── notebook.ipynb         # Notebook principal: EDA, preparación, modelos y guardado
│   ├── modelo.joblib          # Modelo final entrenado (Random Forest)
│   └── README.md              # Documentación específica de la Fase 1
├── requirements.txt           # Dependencias del proyecto
├── .gitignore
└── README.md                  # Este archivo
```

> `housing.csv` no se versiona por su tamaño. Descárgalo de Kaggle y ubícalo en `fase-1/data/housing.csv`.

---

## 📅 Índice de Sprints

| Sprint | Estado | Detalle |
|:---:|:---:|---|
| **Sprint 1** | ✅ Completado | [Ver detalles](#-sprint-1--modelo-predictivo-fase-1) |
| **Sprint 2** | 🔜 Pendiente | Scripts de entrenamiento y predicción |
| **Sprint 3** | 🔜 Pendiente | Despliegue con Docker y API REST |

---

## 🚀 Sprint 1 — Modelo predictivo (Fase 1)

### Partición y semilla
- 80% entrenamiento / 20% prueba con `train_test_split` y `random_state=42`.
- La partición se hace **antes** de cualquier imputación o codificación.

### Modelo base
`DummyRegressor(strategy='mean')`: predice siempre la media de `y_train`. Es el piso contra el cual se mide cualquier mejora.

### Preparación de datos
| Variable | Problema | Estrategia |
|---|---|---|
| `total_bedrooms` | ~1% de valores nulos | Imputación por **mediana** con `SimpleImputer` (robusta ante la asimetría de la variable) |
| `ocean_proximity` | Categórica nominal, sin orden | **One-Hot Encoding** (`drop_first=True`) |

### Prevención de fuga de información
- `train_test_split` se ejecuta sobre los datos crudos, antes de transformar nada.
- El imputador se ajusta (`fit`) **solo con `X_train`**; sobre `X_test` solo se aplica `transform`, con la mediana aprendida en entrenamiento.
- Las columnas del One-Hot de prueba se alinean a las de entrenamiento.
- La variable objetivo nunca se usa como predictora, y el conjunto de prueba no interviene en ninguna decisión de preparación.

### Métricas
- **MAE:** error promedio en dólares; es el criterio principal de selección.
- **RMSE:** penaliza con más fuerza los errores grandes.
- **R²:** proporción de la variabilidad del precio explicada por el modelo.

### Resultados
Ordenados de menor a mayor MAE:

| Modelo | Estrategia / Preprocesamiento | Autor | MAE ($) | RMSE ($) | R² |
|---|---|---|---:|---:|---:|
| **Random Forest** | **Preprocesamiento completo** | **Juan Rincón** | **31,639.71** | **49,036.98** | **0.8165** |
| Random Forest | Solo numéricas (sin One-Hot) | Juan Rincón | 32,098.51 | 49,887.18 | 0.8101 |
| Árbol de Decisión | Preprocesamiento completo | Samuel | 42,866.15 | 62,817.98 | 0.6989 |
| Regresión Lineal | Preprocesamiento completo | Oscar | 50,670.49 | 70,059.19 | 0.6254 |
| Regresión Lineal | Solo numéricas (sin One-Hot) | Oscar | 51,810.09 | 71,131.26 | 0.6139 |
| Baseline | DummyRegressor (media) | Juan Rincón | 90,606.85 | 114,485.64 | -0.0002 |

### Conclusión del Sprint
El Random Forest con preprocesamiento completo reduce el MAE de 90,606.85 a 31,639.71 USD (65% menos que el baseline) y explica el 81.65% de la variabilidad del precio. Agregar `ocean_proximity` mejora consistentemente ambos modelos: en Random Forest el MAE baja en más de 450 USD (de 32,098.51 a 31,639.71). Que los modelos basados en árboles superen al lineal indica que la relación entre las variables y el precio no es lineal. El modelo final se guarda en `fase-1/modelo.joblib`.

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

# 3. Dependencias
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
