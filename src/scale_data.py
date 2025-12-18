#!/usr/bin/env python3
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Chemins des fichiers
X_train_path = Path("data/processed/X_train.csv")
X_test_path = Path("data/processed/X_test.csv")
y_train_path = Path("data/processed/y_train.csv")
y_test_path = Path("data/processed/y_test.csv")

# Charger les fichiers déjà splités
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

# --------------------------
# 1️⃣ Gérer les colonnes non numériques
# --------------------------
if "date" in X_train.columns:
    X_train["year"] = pd.to_datetime(X_train["date"]).dt.year
    X_train["month"] = pd.to_datetime(X_train["date"]).dt.month
    X_train["day"] = pd.to_datetime(X_train["date"]).dt.day
    X_test["year"] = pd.to_datetime(X_test["date"]).dt.year
    X_test["month"] = pd.to_datetime(X_test["date"]).dt.month
    X_test["day"] = pd.to_datetime(X_test["date"]).dt.day
    X_train = X_train.drop(columns=["date"])
    X_test = X_test.drop(columns=["date"])

# Supprimer d'autres colonnes textuelles si besoin
text_cols = X_train.select_dtypes(include="object").columns
X_train = X_train.drop(columns=text_cols)
X_test = X_test.drop(columns=text_cols)

# --------------------------
# 2️⃣ Standardiser les colonnes numériques
# --------------------------
numeric_cols = X_train.select_dtypes(include="number").columns
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols)

# --------------------------
# 3️⃣ Sauvegarder les fichiers
# --------------------------
X_train_scaled.to_csv("data/processed/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/processed/X_test_scaled.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Scaling terminé et fichiers sauvegardés dans data/processed/")
