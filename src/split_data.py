from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemin relatif vers le fichier raw
data_path = Path("../raw.csv")
df = pd.read_csv(data_path)

# Séparer features et cible
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sauvegarder les datasets
processed_path = Path("../data/processed_data")
processed_path.mkdir(parents=True, exist_ok=True)

X_train.to_csv(processed_path / "X_train.csv", index=False)
X_test.to_csv(processed_path / "X_test.csv", index=False)
y_train.to_csv(processed_path / "y_train.csv", index=False)
y_test.to_csv(processed_path / "y_test.csv", index=False)
