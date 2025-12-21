# src/split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging
import sys


# Les répertoires utilisés
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "raw.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# Création des dossiers si besoin
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration des logs

LOG_FILE = LOGS_DIR / "split_data.log"

logger = logging.getLogger("split_data")
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

# Log fichier
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

# Log écran
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Script principale


def main():
    logger.info("Début du split des données")

    logger.info(f"Lecture des données depuis : {DATA_RAW}")
    df = pd.read_csv(DATA_RAW)

    target_col = "silica_concentrate"
    logger.info(f"Variable cible : {target_col}")

    X = df.drop(columns=[target_col])
    y = df[[target_col]]

    logger.info("Séparation train / test (80% / 20%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Sauvegarde des fichiers dans data/processed/")
    X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    logger.info("Split terminé avec succès")

if __name__ == "__main__":
    main()
