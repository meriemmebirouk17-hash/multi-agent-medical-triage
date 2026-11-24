import streamlit as st
from PIL import Image

from agents.coordinator import run_triage

st.set_page_config(page_title="Triage médical multi-agent", page_icon="🩺")

# Choix affichés en français -> valeurs attendues par le modèle clinique
GENRES = {"Femme": "Female", "Homme": "Male"}
OUI_NON = {"Non": 0, "Oui": 1}
MARIAGE = {"Non": "No", "Oui": "Yes"}
TRAVAIL = {
    "Secteur privé": "Private",
    "Indépendant": "Self-employed",
    "Fonction publique": "Govt_job",
    "Enfant (pas d'emploi)": "children",
    "Jamais travaillé": "Never_worked",
}
RESIDENCE = {"Urbaine": "Urban", "Rurale": "Rural"}
TABAC = {
    "Ne fume jamais": "never smoked",
    "Ancien fumeur": "formerly smoked",
    "Fume": "smokes",
    "Inconnu": "Unknown",
}


def afficher_resultat(titre, resultat, libelle_proba, note=None):
    """Affiche le résultat d'un agent selon son statut (ok, error, not_run)."""
    st.subheader(titre)
    statut = resultat["status"]

    if statut == "not_run":
        st.info(f"Agent non exécuté. {resultat['message']}")
    elif statut == "error":
        st.error(f"Erreur pendant l'exécution : {resultat['message']}")
    else:
        sortie = resultat["result"]
        if sortie.get("simulated"):
            st.warning("Résultat simulé : ce n'est pas une vraie prédiction.")
        st.write(f"Classe prédite : **{sortie['label']}**")
        probabilite = sortie.get("probability")
        if probabilite is not None:
            st.metric(libelle_proba, f"{probabilite * 100:.1f} %")
        seuil = sortie.get("threshold")
        if seuil is not None:
            st.caption(f"Seuil de décision utilisé : {seuil}")
        if note:
            st.caption(note)


NIVEAUX = {
    "LOW": ("Faible", "green"),
    "MODERATE": ("Modérée", "orange"),
    "HIGH": ("Élevée", "red"),
}


def afficher_consensus(resultat):
    """Affiche le résultat de l'agent de consensus (priorité de triage)."""
    st.subheader("Agent de consensus (priorité de triage)")
    statut = resultat["status"]

    if statut == "not_run":
        st.info(f"Agent non exécuté. {resultat['message']}")
    elif statut == "error":
        st.error(f"Erreur pendant l'exécution : {resultat['message']}")
    else:
        sortie = resultat["result"]
        niveau, couleur = NIVEAUX.get(sortie["level"], (sortie["level"], "gray"))
        st.markdown(f"### Priorité de triage : :{couleur}[{niveau}]")
        st.metric("Score du consensus", f"{sortie['probability'] * 100:.1f} %")
        st.caption(
            "Démonstration : ce niveau est calculé par une régression logistique "
            "entraînée sur des patients virtuels (radiographies et dossiers "
            "cliniques associés au hasard). Ce n'est pas une décision médicale."
        )

# --- Présentation ---
st.title("Système multi-agent de triage médical")
st.warning(
    "Prototype universitaire. Les résultats affichés ne constituent pas un "
    "diagnostic médical et ne remplacent pas l'avis d'un professionnel de santé."
)

# --- Entrées ---
st.header("1. Données d'entrée")

st.subheader("Radiographie thoracique")
fichier = st.file_uploader(
    "Importer une radiographie (PNG ou JPG)", type=["png", "jpg", "jpeg"]
)

image = None
if fichier is not None:
    try:
        image = Image.open(fichier)
        image.load()
        st.image(image, caption="Radiographie importée", width=300)
    except Exception as erreur:
        st.error(f"Impossible de lire l'image : {erreur}")
        image = None

st.subheader("Données cliniques")
saisir = st.checkbox("Saisir les données cliniques du patient")

donnees_cliniques = None
if saisir:
    st.caption(
        "Les valeurs proposées par défaut sont de simples exemples : "
        "remplacez-les par celles du patient."
    )
    col1, col2 = st.columns(2)
    with col1:
        genre = st.selectbox("Genre", list(GENRES))
        age = st.number_input(
            "Âge (années)", min_value=0.0, max_value=120.0, value=50.0, step=1.0
        )
        hypertension = st.selectbox("Hypertension", list(OUI_NON))
        maladie_cardiaque = st.selectbox("Maladie cardiaque", list(OUI_NON))
        marie = st.selectbox("Déjà marié(e)", list(MARIAGE))
    with col2:
        travail = st.selectbox("Type de travail", list(TRAVAIL))
        residence = st.selectbox("Lieu de résidence", list(RESIDENCE))
        glycemie = st.number_input(
            "Glycémie moyenne (mg/dL)",
            min_value=40.0, max_value=600.0, value=100.0, step=1.0,
        )
        imc_inconnu = st.checkbox("IMC inconnu")
        imc = st.number_input(
            "IMC (indice de masse corporelle)",
            min_value=10.0, max_value=100.0, value=25.0, step=0.1,
            disabled=imc_inconnu,
        )
        tabac = st.selectbox("Statut tabagique", list(TABAC))

    donnees_cliniques = {
        "gender": GENRES[genre],
        "age": age,
        "hypertension": OUI_NON[hypertension],
        "heart_disease": OUI_NON[maladie_cardiaque],
        "ever_married": MARIAGE[marie],
        "work_type": TRAVAIL[travail],
        "Residence_type": RESIDENCE[residence],
        "avg_glucose_level": glycemie,
        "bmi": None if imc_inconnu else imc,
        "smoking_status": TABAC[tabac],
    }

# --- Analyse ---
st.header("2. Analyse")
st.caption(
    "Pour obtenir la priorité de triage, il faut fournir une radiographie "
    "ET les données cliniques."
)
if st.button("Lancer l'analyse"):
    with st.spinner("Analyse en cours..."):
        resultats = run_triage(image=image, clinical_data=donnees_cliniques)

    afficher_resultat(
        "Agent d'imagerie (pneumonie)",
        resultats["imaging"],
        "Probabilité estimée pour la classe PNEUMONIA",
        note=(
            "Ce modèle a été entraîné sur un seul jeu de radiographies : "
            "ses résultats peuvent être peu fiables sur d'autres images."
        ),
    )
    afficher_resultat(
        "Agent clinique (risque d'AVC)",
        resultats["clinical"],
        "Score du modèle",
        note=(
            "Ce score n'est pas un risque réel en pourcentage : le modèle est "
            "réglé avec un seuil bas pour repérer les cas rares."
        ),
    )
    afficher_consensus(resultats["consensus"])
    