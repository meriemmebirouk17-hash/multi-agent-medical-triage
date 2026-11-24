"""Adaptateur de l'agent clinique (risque d'AVC, modele du membre 2)."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "stroke_rf.joblib"

# Les 10 variables attendues par le modele du membre 2

COLONNES = [
    "gender", "age", "hypertension", "heart_disease", "ever_married",
    "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status",
    ]

# Valeurs autorisees pour les variables a choix (celles du jeu de donnees)
VALEURS_AUTORISEES = {
    "gender": ["Female", "Male"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Govt_job", "Never_worked", "Private", "Self-employed", "children"],
    "Residence_type": ["Rural", "Urban"],
    "smoking_status": ["Unknown", "formerly smoked", "never smoked", "smokes"],
    "hypertension": [0, 1],
    "heart_disease": [0, 1],
}

# Limites de plausibilite pour les variables numeriques (controle de saisie)
LIMITES = {"age": (0, 120), "avg_glucose_level": (40, 600), "bmi": (10, 100)}

_modele = None

def _charger_modele():
    """Charge le modele une seule fois, puis le garde en memoire."""
    global _modele
    if _modele is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modele clinique introuvable : {MODEL_PATH} ")
        _modele = joblib.load(MODEL_PATH)
    return _modele


def _valider(clinical_data):
    """Verifie les donnees saisies et retourne une ligne propre."""
    if not isinstance(clinical_data, dict):
        raise ValueError("Les donnees cliniques doivent etre un dictionnaire.")

    manquantes = [
        c for c in COLONNES if c != "bmi" and clinical_data.get(c) is None
    ]

    if manquantes:
        raise ValueError("Variables manquantes : " + ", ".join(manquantes))

    ligne = {}
    for colonne in COLONNES:
        valeur = clinical_data.get(colonne)

        # Le BMI peut etre absent : le modele le remplace par la mediane
        if colonne == "bmi" and pd.isna(valeur):
            ligne[colonne] = np.nan
            continue

        if colonne in VALEURS_AUTORISEES:
            if valeur not in VALEURS_AUTORISEES[colonne]:
                raise ValueError(
                    f"Valeur invalide pour {colonne} : {valeur!r}. "
                    f"Valeurs possibles : {VALEURS_AUTORISEES[colonne]}"
                )

        else: 
            try:
                valeur = float(valeur)
            except (TypeError, ValueError):
                raise ValueError(f"{colonne} doit etre un nombre : {valeur!r}")
            mini, maxi = LIMITES[colonne]
            if not mini <= valeur <= maxi:
                raise ValueError(
                    f"{colonne} = {valeur} est hors de l'intervalle {mini} a {maxi}."
                )
        
        ligne[colonne] = valeur

    return ligne


def predict_stroke(clinical_data):
    """Estime le risque d'AVC et retourne un resultat au format du projet."""
    ligne = _valider(clinical_data)
    modele = _charger_modele()
    pipeline = modele["pipeline"]
    seuil = float(modele["threshold"])

    donnees = pd.DataFrame([ligne])[COLONNES]
    probabilite = float(pipeline.predict_proba(donnees)[0,1])

    return {
        "label": "RISQUE ELEVE" if probabilite >= seuil else "RISQUE FAIBLE",
        "probability": probabilite,
        "threshold": round(seuil, 3),
        "simulated": False,
    }

