#!/usr/bin/env python3
# src/scale_data.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging
import sys

# Gestion des chemins
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Fichiers input avant traitement 
X_train_path = DATA_PROCESSED / "X_train.csv"
X_test_path = DATA_PROCESSED / "X_test.csv"
y_train_path = DATA_PROCESSED / "y_train.csv"
y_test_path = DATA_PROCESSED / "y_test.csv"

# Configuration des logs

LOG_FILE = LOGS_DIR / "scale_data.log"

logger = logging.getLogger("scale_data")
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
    logger.info("Début du scaling des données")

    logger.info("Chargement des données splitées")
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    if "date" in X_train.columns:
        logger.info("Traitement de la colonne date")
        for df in [X_train, X_test]:
            df["year"] = pd.to_datetime(df["date"]).dt.year
            df["month"] = pd.to_datetime(df["date"]).dt.month
            df["day"] = pd.to_datetime(df["date"]).dt.day
            df.drop(columns=["date"], inplace=True)

    text_cols = X_train.select_dtypes(include="object").columns
    if len(text_cols) > 0:
        logger.info(f"Suppression des colonnes textuelles : {list(text_cols)}")
        X_train.drop(columns=text_cols, inplace=True)
        X_test.drop(columns=text_cols, inplace=True)

    numeric_cols = X_train.select_dtypes(include="number").columns
    logger.info(f"Standardisation des colonnes : {list(numeric_cols)}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train[numeric_cols]),
        columns=numeric_cols
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test[numeric_cols]),
        columns=numeric_cols
    )

    # Sauvegarde
    logger.info("Sauvegarde des fichiers scalés")

    X_train_scaled.to_csv(DATA_PROCESSED / "X_train_scaled.csv", index=False)
    X_test_scaled.to_csv(DATA_PROCESSED / "X_test_scaled.csv", index=False)
    y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
    y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

    logger.info("Scaling terminé avec succès")

if __name__ == "__main__":
    main()
