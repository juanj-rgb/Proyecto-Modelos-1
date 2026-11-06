# Fase 2 — Scripts y Docker: Predicción de precios de vivienda en California

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

El pipeline de `fase-2` es el mismo de la Fase 1 v1.1 (`SimpleImputer` por mediana en las 8 variables numéricas + `OneHotEncoder(handle_unknown='ignore')` en `ocean_proximity` + Random Forest de 100 árboles, `random_state=42`), con la misma partición 80/20 (`random_state=42`) y la misma validación cruzada (5 folds, `KFold` con mezcla y semilla 42). Por eso `v1` reproduce las cifras de la Fase 1 v1.1 (sección 9).

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
| Pull Requests sin revisión formal | En la Fase 1 v1.1 los PR #9 a #12 se fusionaron con aprobación de otro integrante. En esta fase, cada PR hacia `develop` lo aprobó un integrante distinto de su autor. |

## 3. Entorno

**Python utilizado: 3.14.5** (entorno local de Oscar, con el que se entrenaron los modelos entregados). La imagen Docker usa `python:3.14-slim`. Las mismas versiones de `requirements.txt` se verificaron además con Python 3.12 y con Python 3.14.10 (las 8 pruebas pasan en ambas).

```powershell
cd fase-2
python -m venv venv
venv\Scripts\Activate.ps1        # PowerShell. En Bash: source venv/bin/activate
pip install -r requirements.txt
