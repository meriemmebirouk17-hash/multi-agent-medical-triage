"""Probabilites du modele clinique sur son jeu de TEST (jamais vu a l'entrainement).

On refait exactement la meme separation train/test que le membre 2 :
test_size=0.2, stratify=y, random_state=42.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "healthcare-dataset-stroke-data.csv"
MODEL_PATH = ROOT / "models" / "stroke_rf.joblib"
OUTPUT_CSV = ROOT / "data" / "stroke_test_probs.csv"

#preparation des donnees
df = pd.read_csv(DATA_CSV)
df = df.drop(columns=["id"])
df = df[df["gender"] != "Other"]

X = df.drop(columns=["stroke"])
y = df["stroke"]

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Probabilites du modele sur le jeu de test uniquement
modele = joblib.load(MODEL_PATH)
pipeline = modele["pipeline"]
seuil = modele["threshold"]
proba = pipeline.predict_proba(X_test)[:, 1]

sortie = pd.DataFrame(
    {"age": X_test["age"].values, "stroke": y_test.values, "prob_stroke": proba}
)
sortie.to_csv(OUTPUT_CSV, index=False)

# Controle : ces chiffres doivent etre ceux du notebook du membre 2
prediction = (proba >= seuil).astype(int)
print("Fichier cree :", OUTPUT_CSV)
print("Patients de test :", len(sortie), "dont AVC :", int(y_test.sum()))
print("Seuil :", round(seuil, 3))
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))