"""Constantes del proyecto: esquema del dataset California Housing."""

DEFAULT_TARGET = "median_house_value"

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
CATEGORICAL_FEATURES = ["ocean_proximity"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columnas usadas para definir "zonas" en la particion espacial opcional.
ZONE_COLUMNS = ("latitude", "longitude")

MODEL_FILENAME = "modelo.joblib"
METRICS_FILENAME = "metrics.json"
METADATA_FILENAME = "metadata.json"
PREDICTION_COLUMN = "prediccion"

ALGORITHMS = ("random_forest", "decision_tree", "linear_regression")
