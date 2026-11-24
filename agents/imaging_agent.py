
"""Adaptateur de l'agent d'imagerie"""
from pathlib import Path

from agents import image_agent

image_agent.MODEL_PATH = str(
    Path(__file__).resolve().parent.parent/ "models" / "cnn_pneumonia.keras"
)

def predict_pneumonia(image):
    """Analyse une radiographie et retourne un resultat au format du projet."""
    result = image_agent.predict_full(image)
    return {
        "label": result["label"],
        "probability": result["probability"],
        "threshold": result["threshold"],
        "simulated": False,
    }

