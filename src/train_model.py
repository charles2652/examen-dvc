import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Charger les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Garder uniquement les colonnes numériques
X_train = X_train.select_dtypes(include=["float64", "int64"])

# Créer le modèle avec les meilleurs paramètres si déjà trouvés
model = RandomForestRegressor(
    n_estimators=100,  # exemple, ou charger depuis best_params.json
    max_depth=None
)

# Entraîner
model.fit(X_train, y_train)

# Sauvegarder
joblib.dump(model, "models/model.pkl")
print("Entraînement terminé et modèle sauvegardé dans models/model.pkl")
