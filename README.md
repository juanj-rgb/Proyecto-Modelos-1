# Predicción de Precios de Vivienda en California

Este repositorio contiene el desarrollo de un pipeline end-to-end de Machine Learning para
predecir el valor mediano de las viviendas en distintos distritos de California, utilizando el dataset *California Housing*.

---

## Contexto y Objetivo del Proyecto

El objetivo principal es construir y evaluar modelos analíticos capaces de estimar el precio de las propiedades
(`median_house_value`) en función de variables demográficas, geográficas y socioeconómicas. 

Este proyecto se aborda mediante una metodología iterativa dividida en sprints, 
asegurando buenas prácticas de control de versiones (Git/GitHub), limpieza de datos y validación de modelos.

---

## Estructura del Repositorio

```text
Proyecto-Modelos-1/
│
├── fase-1/                   # Sprint 1: Preparación, Baseline y Modelos Iniciales
│   ├── data/                 # Datasets crudos y particiones (Train / Test)
│   └── notebook.ipynb        # Cuaderno principal de experimentación
│
├── resultados/
│   └── metricas.csv          # Bitácora general de experimentos y métricas
│
├── .gitignore                # Archivos ignorados por Git
├── requirements.txt          # Dependencias y librerías del proyecto
└── README.md                 # Documentación e índice general del proyecto


Índice de Sprints y Avances del Proyecto
Sprint 1: Preparación de Datos, Baseline y Modelos Iniciales

Sprint 2: (En desarrollo / Próximamente)

Sprint 3: (En desarrollo / Próximamente)

Sprint 1: Preparación de Datos, Baseline y Modelos Iniciales
Durante la primera etapa se estableció el flujo base de preparación de datos y evaluación de modelos, 
fijando una semilla aleatoria (random_state=42)
para garantizar la reproducibilidad de las particiones (80% Entrenamiento / 20% Prueba).

1. Modelo Baseline (Línea Base)
Se entrenó un DummyRegressor usando la media del conjunto de entrenamiento para establecer
la métrica mínima contra la cual comparar modelos más complejos.

2. Preprocesamiento e Ingeniería de Características
Imputación de Faltantes: Se utilizó SimpleImputer(strategy='median') para gestionar los valores
nulos en la variable total_bedrooms.

Codificación Categórica: Se applied One-Hot Encoding sobre la variable categórica ocean_proximity,
permitiendo incorporar la ubicación geográfica costera como señal predictiva.

3. Comparativa de Resultados y Desempeño
Las métricas registradas en resultados/metricas.csv muestran el impacto directo de la ingeniería
de datos en la precisión de las predicciones:

Baseline_Mean,Oscar,90606.85,114485.64,-0.0002
Baseline_Mean,Oscar,90606.85,114485.64,-0.0002
Linear_Regression_Simple,Oscar,51810.09,71131.26,0.6139
Random_Forest_Simple,Oscar,32098.51,49887.18,0.8101
Baseline_Mean,Oscar,90606.85,114485.64,-0.0002
Linear_Regression_Simple,Oscar,51810.09,71131.26,0.6139
Random_Forest_Simple,Oscar,32098.51,49887.18,0.8101
Baseline_Mean,Oscar,90606.85,114485.64,-0.0002
Linear_Regression_FullPrep,Oscar_Samuel,50670.49,70059.19,0.6254
Random_Forest_FullPrep,Oscar_Samuel,31639.71,49036.98,0.8165

Conclusión del Sprint 1: La incorporación del One-Hot Encoding en ocean_proximity junto con la
imputación correcta redujo el error medio absoluto (MAE) de Random Forest en más de $450 USD
por vivienda y elevó la capacidad explicativa ($R^2$) al 81.65%.


Cómo Reproducir el Proyecto
1. Clonar el repositorio

git clone [https://github.com/juanj-rgb/Proyecto-Modelos-1.git](https://github.com/juanj-rgb/Proyecto-Modelos-1.git)
cd Proyecto-Modelos-1

2. Crear y activar un entorno virtual

python -m venv .venv

# En Windows (PowerShell):
.\.venv\Scripts\activate

# En Mac/Linux:
source .venv/bin/activate

3. Instalar dependencias

pip install -r requirements.txt

4. Ejecutar la Fase 1

Abre el archivo fase-1/notebook.ipynb en VS Code o Jupyter Lab y ejecuta todas las celdas para reproducir la carga de datos,
el entrenamiento y la generación de la tabla de métricas.

Equipo de Trabajo

Oscar Plaza— Gestión del flujo principal, estructura del pipeline de modelos, control de versiones e integración final.

Juan Rincon — Configuración y administración del repositorio de proyectos.

Samuel Velasquez— Lógica de imputación de valores nulos y codificación One-Hot para características categóricas.
