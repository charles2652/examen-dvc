#!/usr/bin/env python3


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import logging
import sys
import json
import joblib


# Gestion des chemins

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers d'entrée
X_TRAIN_PATH = DATA_PROCESSED / "X_train_scaled.csv"
Y_TRAIN_PATH = DATA_PROCESSED / "y_train.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.json"


# Configuration des logs


LOG_FILE = LOGS_DIR / "train_model.log"

logger = logging.getLogger("train_model")
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
    logger.info("Début de l'entraînement du modèle")

    logger.info("Chargement des données d'entraînement")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

    X_train = X_train.select_dtypes(include=["float64", "int64"])
    logger.info(f"Nombre de features utilisées : {X_train.shape[1]}")

    if BEST_PARAMS_PATH.exists():
        logger.info("Chargement des meilleurs paramètres depuis best_params.json")
        with open(BEST_PARAMS_PATH, "r") as f:
            best_params = json.load(f)
    else:
        logger.warning("best_params.json non trouvé, utilisation des paramètres par défaut")
        best_params = {
            "n_estimators": 100,
            "max_depth": None
        }

    logger.info(f"Paramètres du modèle : {best_params}")

    model = RandomForestRegressor(
        random_state=42,
        **best_params
    )


    logger.info("Entraînement du modèle en cours")
    model.fit(X_train, y_train)


    model_path = MODELS_DIR / "model.pkl"
    joblib.dump(model, model_path)

    logger.info(f"Modèle sauvegardé dans {model_path}")
    logger.info("Entraînement terminé avec succès")

if __name__ == "__main__":
    main()
