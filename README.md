# Système multi-agent de triage médical\

Prototype universitaire qui coordonne trois agents :

- **Agent d'imagerie** : CNN (apprentissage par transfert, TesnsorFlow/Keras) qui estime la probabilité de pneumonie à partir d'une radiographie thoraciques.
- **Agent clinique** : Radnom Forest (scikit-learn) qui estime un score de risque d'AVC à partir de données cliniques.
-**Agent de consensus** : Régression logistique qui combine les deux scores en un niveau de priorité de triage (LOW, MODERATE, HIGH). Elle est entraînée sur des << patients virtuels >> (radiographies et dossiers cliniques associés au hasard).

> **Avertissement.** Ce projet est un prototype pédagogique. Les résultats ne constituent pas un diagnostic médical et ne remplacent pas l'avis d'un professionnel de santé.

## Prérequis 

-Windows, Powershell
-Python 3.13

## Installation

Depuis le dossier du projet :

```powershell
py -3.13 -m venv env_ia
.\env_ia\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ne renommez pas et ne déplacer pas le dossier du projet aprés avoir créé 'env_ia' : l'environnemnet virtuel doit être recréé.

## Fichiers nécessaires 

Les modèles entraînés doivent se trouver dans 'models/' :

| Fichier | Origine |
|---|---|
| `cnn_pneumonia.keras` | modèle d'imagerie (membre 1) |
| `stroke_rf.joblib` | modèle clinique (membre 2) |
| `consensus_lr.joblib` | modèle de consensus, généré par `training/train_consensus.py` |

## Lancer l'application

```powershell
streamlit run app.py
```

L'application s'ouvre dans le navigateur (adresse 'http://localhost:8501'). Il est possible de fournir une radiographie, des données cliniques, ou les deux. Les résultats de chaque agent sont affichés séparément. La priorité de triage n'est calculée que si les deux analyses sont disponibles.

## Lancer les tests

```powershell
python -m pytest tests -v
```

Les tests vérifient le comportement du logiciel (entrées absentes, données invalides, pannes de modèle). Ils ne valident pas médicalement les prédictions.

## Régénérer le modèle de consensus

Nécessite `data/healthcare-dataset-stroke-data.csv` et `data/image_agent_test_probs.csv` :

```powershell
python training\make_stroke_test_probs.py
python training\train_consensus.py
```

## Structure du projet

```
app.py                  interface Streamlit
agents/
    coordinator.py      orchestration des agents
    imaging_agent.py    adaptateur du modèle d'imagerie
    image_agent.py      code de prédiction fourni par le membre 1
    clinical_agent.py   adaptateur du modèle clinique (validation des entrées)
    consensus_agent.py  priorité de triage
models/                 modèles entraînés
data/                   jeux de données et fichiers de probabilités
training/               scripts d'entraînement du consensus
tests/                  tests de comportement
notebooks/              notebook d'entraînement de l'agent clinique
reports/                documents du rapport
```

## Limites

- Les modèles sont entraînés sur un seul jeu de données chacun : leurs résultats peuvent être peu fiables sur d'autres données.
- Le score clinique n'est pas un risque réel en pourcentage.
- Le consensus est entraîné sur des patients virtuels, en supposant l'indépendance entre pneumonie et AVC : c'est une démonstration du mécanisme, pas un outil médical.

