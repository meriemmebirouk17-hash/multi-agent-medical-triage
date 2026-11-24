"""Agent de consensus : priorite de triage de demonstration.

Regression logistique entrainee sur des "patients virtuels"
(voir training/train_consensus.py). Ce n'est pas un outil medical.
"""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "consensus_lr.joblib"

# Seuils (choix de conception) : milieux entre les scores moyens des groupes
# de patients virtuels : sain 0.44 / une pathologie 0.67-0.87 / les deux 0.94
SEUIL_MODEREE = 0.55
SEUIL_ELEVEE = 0.90

_modele = None

def _charger_modele():
    """Charge le modele de consensus une seule fois."""
    global _modele
    if _modele is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modele de consensus introuvable : {MODEL_PATH}")
        _modele = joblib.load(MODEL_PATH)
    return _modele


def niveau_priorite(score):
    """Transforme le score (0 a 1) en niveau de priorite."""
    if score >= SEUIL_ELEVEE:
        return "HIGH"
    if score >= SEUIL_MODEREE:
        return "MODERATE"
    return "LOW"


def predict_priority(p_pneumonia, p_stroke, age):
    """Combine les deux probabilites (et l'age) en un niveau de priorite."""
    for nom, valeur in (("p_pneumonia", p_pneumonia), ("p_stroke", p_stroke)):
        if valeur is None or not 0 <= valeur <= 1:
            raise ValueError(f"{nom} doit etre une probabilite entre 0 et 1 : {valeur!r}")
    if age is None:
        raise ValueError("L'age est necessaire pour le consensus.")

    modele = _charger_modele()
    valeurs = {
        "p_pneumonia": p_pneumonia,
        "p_stroke": p_stroke,
        "interaction": p_pneumonia * p_stroke,
        "age": float(age),
    }

    donnees = pd.DataFrame([valeurs])[modele["features"]]
    score = float(modele["pipeline"].predict_proba(donnees)[0, 1])
    niveau = niveau_priorite(score)

    return {
        "label": f"PRIORITE {niveau}",
        "level": niveau,
        "probability": score,
        "simulated": False,
    }

