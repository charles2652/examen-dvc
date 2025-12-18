from pathlib import Path
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

processed_path = Path("../data/processed_data")

X_train = pd.read_csv(processed_path / "X_train_scaled.csv")
y_train = pd.read_csv(processed_path / "y_train.csv")

# Modèle et grille de paramètres
model = RandomForestRegressor(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train.values.ravel())

# Sauvegarder les meilleurs paramètres
Path("../models").mkdir(exist_ok=True)
joblib.dump(grid.best_params_, "../models/best_params.pkl")
