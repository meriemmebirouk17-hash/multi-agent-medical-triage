
    
"""Coordinateur : appelle les agents disponibles et rassemble leurs resultats."""

from agents.clinical_agent import predict_stroke
from agents.consensus_agent import predict_priority
from agents.imaging_agent import predict_pneumonia


def _not_run(reason):
    """Resultat d'un agent qui n'a pas ete execute."""
    return {"status": "not_run", "result": None, "message": reason}


def _run_agent(prediction_function, data):
    """Execute un agent. Si une erreur survient, elle est capturee."""
    try:
        result = prediction_function(data)
        return {"status": "ok", "result": result, "message": ""}
    except Exception as error:
        return {
            "status": "error",
            "result": None,
            "message": f"{type(error).__name__}: {error}",
        }


def _run_consensus(imaging, clinical, clinical_data):
    """Lance le consensus seulement si les deux agents ont produit un resultat."""
    if imaging["status"] != "ok" or clinical["status"] != "ok":
        return _not_run(
            "Le consensus necessite que l'agent d'imagerie et l'agent clinique "
            "aient tous deux produit un resultat."
        )
    entrees = {
        "p_pneumonia": imaging["result"]["probability"],
        "p_stroke": clinical["result"]["probability"],
        "age": clinical_data["age"],
    }
    return _run_agent(
        lambda e: predict_priority(e["p_pneumonia"], e["p_stroke"], e["age"]),
        entrees,
    )


def run_triage(image=None, clinical_data=None):
    """Lance les agents pour lesquels des donnees sont fournies.

    Retourne trois resultats separes : 'imaging', 'clinical' et 'consensus'.
    Chaque resultat a un statut : 'ok', 'error' ou 'not_run'.
    """
    if image is not None:
        imaging = _run_agent(predict_pneumonia, image)
    else:
        imaging = _not_run("Aucune radiographie fournie.")

    if clinical_data:
        clinical = _run_agent(predict_stroke, clinical_data)
    else:
        clinical = _not_run("Aucune donnee clinique fournie.")

    consensus = _run_consensus(imaging, clinical, clinical_data)

    return {"imaging": imaging, "clinical": clinical, "consensus": consensus}