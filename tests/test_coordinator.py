"""Tests de comportement du coordinateur.

Ces tests verifient le fonctionnement du LOGICIEL (statuts, erreurs, separation
des resultats). Ils ne valident pas medicalement les predictions des modeles.
"""

import pytest
from PIL import Image

from agents import clinical_agent, coordinator, image_agent

PATIENT = {
    "gender": "Male",
    "age": 67,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}

@pytest.fixture
def image():
    """Image grise synthetique : sert uniquement a tester le logiciel"""
    return Image.new("RGB", (300, 300), color=(128, 128, 128))


def statuts(resultats):
    return (
        resultats["imaging"]["status"],
        resultats["clinical"]["status"],
        resultats["consensus"]["status"],
    )

# --- Cas d'entree ---

def test_aucune_entree():
    resultats = coordinator.run_triage()
    assert statuts(resultats) == ("not_run", "not_run", "not_run")
    assert resultats["imaging"]["result"] is None
    assert resultats["clinical"]["result"] is None


def test_radiographie_seule(image):
    resultats = coordinator.run_triage(image=image)
    assert statuts(resultats) == ("ok", "not_run", "not_run")
    assert 0 <= resultats["imaging"]["result"]["probability"] <= 1
    assert resultats["clinical"]["result"] is None


def test_donnees_cliniques_seules():
    resultats = coordinator.run_triage(clinical_data=PATIENT)
    assert statuts(resultats) == ("not_run", "ok", "not_run")
    assert 0 <= resultats["clinical"]["result"]["probability"] <= 1
    assert resultats["imaging"]["result"] is None


def test_les_deux_entrees(image):
    resultats = coordinator.run_triage(image=image, clinical_data=PATIENT)
    assert statuts(resultats) == ("ok", "ok", "ok")
    assert resultats["consensus"]["result"]["level"] in {"LOW", "MODERATE", "HIGH"}
    # Les resultats des deux agents restent separes
    assert resultats["imaging"]["result"] is not resultats["clinical"]["result"]


# --- Donnees incorrectes ou incompletes ---

def test_donnees_cliniques_incompletes():
    resultats = coordinator.run_triage(clinical_data={"age":50})
    assert resultats["clinical"]["status"] == "error"
    assert "Variables manquantes" in resultats["clinical"]["message"]
    assert resultats["consensus"]["status"] == "not_run"


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("gender", "Alien"),
        ("age", -5),
        ("age", "abc"),
        ("avg_glucose_level", 10000),
        ("hypertension", 2),
    ],
)
def test_valeur_clinique_invalide(champ, valeur):
    donnees = {**PATIENT, champ: valeur}
    resultats = coordinator.run_triage(clinical_data=donnees)
    assert resultats["clinical"]["status"] == "error"
    assert resultats["consensus"]["status"] == "not_run"

def test_bmi_manquant_est_accepte():
    donnees = {**PATIENT, "bmi": None}
    resultats = coordinator.run_triage(clinical_data=donnees)
    assert resultats["clinical"]["status"] == "ok"


def test_image_illisible():
    resultats = coordinator.run_triage(image="fichier_qui_n_existe_pas.png")
    assert resultats["imaging"]["status"] == "error"
    assert resultats["consensus"]["status"] == "not_run"


# --- Echec d'un modele ---

def test_echec_agent_imagerie(monkeypatch, image):
    def agent_en_panne(_):
        raise RuntimeError("panne simulee")

    monkeypatch.setattr(coordinator, "predict_pneumonia", agent_en_panne)
    resultats = coordinator.run_triage(image=image, clinical_data=PATIENT)
    assert statuts(resultats) == ("error", "ok", "not_run")
    assert "panne simulee" in resultats["imaging"]["message"]


def test_echec_agent_clinique(monkeypatch, image):
    def agent_en_panne(_):
        raise RuntimeError("panne simulee")

    monkeypatch.setattr(coordinator, "predict_stroke", agent_en_panne)
    resultats = coordinator.run_triage(image=image, clinical_data=PATIENT)
    assert statuts(resultats) == ("ok", "error", "not_run")
    assert "panne simulee" in resultats["clinical"]["message"]


def test_fichier_modele_imagerie_introuvable(monkeypatch, image):
    monkeypatch.setattr(image_agent, "_model", None)
    monkeypatch.setattr(image_agent, "MODEL_PATH", "modele_inexistant.keras")
    resultats = coordinator.run_triage(image= image)
    assert resultats["imaging"]["status"] == "error"
    assert "FileNotFoundError" in resultats["imaging"]["message"]


def test_fichier_modele_clinique_introuvable(monkeypatch):
    monkeypatch.setattr(clinical_agent, "_modele", None)
    monkeypatch.setattr(
        clinical_agent,
        "MODEL_PATH",
        clinical_agent.MODEL_PATH.parent / "inexistant.joblib",
    )
    resultats = coordinator.run_triage(clinical_data=PATIENT)
    assert resultats["clinical"]["status"] == "error"
    assert "FileNotFoundError" in resultats["clinical"]["message"]
