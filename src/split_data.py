# src/split_data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Chemin vers le fichier raw
data_path = Path("raw.csv")

# Lire les données
df = pd.read_csv(data_path)

# Nom de la variable cible
target_col = "silica_concentrate"

# Séparer features et target
X = df.drop(columns=[target_col])
y = df[[target_col]]  # garder comme DataFrame pour sauvegarde

# Fractionner en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Créer le dossier processed si nécessaire
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

# Sauvegarder les fichiers
X_train.to_csv(processed_dir / "X_train.csv", index=False)
X_test.to_csv(processed_dir / "X_test.csv", index=False)
y_train.to_csv(processed_dir / "y_train.csv", index=False)
y_test.to_csv(processed_dir / "y_test.csv", index=False)

print("Split terminé et fichiers sauvegardés dans data/processed/")
