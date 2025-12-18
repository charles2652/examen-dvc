import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# Garder uniquement les colonnes numériques
X_test = X_test.select_dtypes(include=["float64", "int64"])

# Charger le modèle
model = joblib.load("models/model.pkl")

# Prédire
y_pred = model.predict(X_test)

# Calculer métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Sauvegarder les métriques
import json
with open("metrics/metrics.json", "w") as f:
    json.dump({"mse": mse, "r2": r2}, f)

print(f"Évaluation terminée. MSE: {mse:.4f}, R2: {r2:.4f}")
