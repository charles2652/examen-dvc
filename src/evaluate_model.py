#!/usr/bin/env python3


import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import logging
import sys
import json


# Gestion des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
LOGS_DIR = PROJECT_ROOT / "logs"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


X_TEST_PATH = DATA_PROCESSED / "X_test_scaled.csv"
Y_TEST_PATH = DATA_PROCESSED / "y_test.csv"
MODEL_PATH = MODELS_DIR / "model.pkl"


# Configuration des logs
LOG_FILE = LOGS_DIR / "evaluate_model.log"

logger = logging.getLogger("evaluate_model")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Script principal
def main():
    logger.info("Début de l'évaluation du modèle")

    logger.info("Chargement des données de test")
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

    X_test = X_test.select_dtypes(include=["float64", "int64"])
    logger.info(f"Nombre de features utilisées : {X_test.shape[1]}")

    logger.info("Chargement du modèle entraîné")
    model = joblib.load(MODEL_PATH)

    logger.info("Prédiction sur le jeu de test")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"MSE : {mse:.4f}")
    logger.info(f"R2  : {r2:.4f}")

    metrics = {
        "mse": mse,
        "r2": r2
    }

    metrics_path = METRICS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Métriques sauvegardées dans {metrics_path}")
    logger.info("Évaluation terminée avec succès")

if __name__ == "__main__":
    main()
