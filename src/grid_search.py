# grid_search.py

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Charger X_train_scaled et y_train
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Garder uniquement les colonnes numériques
X_train = X_train.select_dtypes(include=["float64", "int64"])

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10, 20]
}

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Sauvegarder les meilleurs paramètres
import json
with open("models/best_params.json", "w") as f:
    json.dump(grid_search.best_params_, f)

print("Grid search terminé")
