from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

processed_path = Path("../data/processed_data")

X_train = pd.read_csv(processed_path / "X_train_scaled.csv")
y_train = pd.read_csv(processed_path / "y_train.csv")

# Charger les meilleurs paramètres
best_params = joblib.load("../models/best_params.pkl")

# Entraîner le modèle
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train.values.ravel())

# Sauvegarder le modèle entraîné
joblib.dump(model, "../models/final_model.pkl")
