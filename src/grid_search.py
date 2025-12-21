#!/usr/bin/env python3


import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path
import logging
import sys
import json

# Gestion des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Les Fichiers d'entrée
X_TRAIN_PATH = DATA_PROCESSED / "X_train_scaled.csv"
Y_TRAIN_PATH = DATA_PROCESSED / "y_train.csv"


# Configuration des logs

LOG_FILE = LOGS_DIR / "grid_search.log"

logger = logging.getLogger("grid_search")
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
    logger.info("Début du Grid Search")

    logger.info("Chargement des données d'entraînement")
    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()

    X_train = X_train.select_dtypes(include=["float64", "int64"])
    logger.info(f"Nombre de features utilisées : {X_train.shape[1]}")

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20]
    }

    logger.info(f"Paramètres testés : {param_grid}")

    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=param_grid,
        cv=3,
        n_jobs=-1
    )

    logger.info("Lancement du GridSearchCV")
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    logger.info(f"Meilleurs paramètres : {best_params}")

    best_params_path = MODELS_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)

    logger.info(f"Paramètres sauvegardés dans {best_params_path}")
    logger.info("Grid search terminé avec succès")

if __name__ == "__main__":
    main()
