from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Chemin vers les fichiers split
processed_path = Path("../data/processed_data")

# Charger les datasets
X_train = pd.read_csv(processed_path / "X_train.csv")
X_test = pd.read_csv(processed_path / "X_test.csv")

# Sélection des colonnes numériques uniquement
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train_numeric = X_train[numeric_cols]
X_test_numeric = X_test[numeric_cols]

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

# Reconstruire les DataFrames avec les colonnes originales
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols)

# Sauvegarder les datasets normalisés
X_train_scaled_df.to_csv(processed_path / "X_train_scaled.csv", index=False)
X_test_scaled_df.to_csv(processed_path / "X_test_scaled.csv", index=False)

# Sauvegarder le scaler pour un usage futur
Path("../models").mkdir(exist_ok=True)
joblib.dump(scaler, "../models/scaler.pkl")

print("Normalisation terminée. Colonnes numériques normalisées :")
print(list(numeric_cols))
