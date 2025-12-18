from pathlib import Path
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

processed_path = Path("../data/processed_data")

X_test = pd.read_csv(processed_path / "X_test_scaled.csv")
y_test = pd.read_csv(processed_path / "y_test.csv")

# Charger le modèle
model = joblib.load("../models/final_model.pkl")

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarder les prédictions
predictions_path = Path("../data")
predictions_path.mkdir(exist_ok=True)
pd.DataFrame({"y_true": y_test.values.ravel(), "y_pred": y_pred}).to_csv(predictions_path / "predictions.csv", index=False)

# Calculer et sauvegarder les métriques
metrics = {
    "mse": mean_squared_error(y_test, y_pred),
    "r2": r2_score(y_test, y_pred)
}

metrics_path = Path("../metrics")
metrics_path.mkdir(exist_ok=True)
with open(metrics_path / "scores.json", "w") as f:
    json.dump(metrics, f, indent=4)
