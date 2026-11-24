"""Entrainement de l'agent de consensus (regression logistique).

Principe : on fabrique des "patients virtuels" en associant une radiographie
et un dossier clinique issus des jeux de TEST des deux modeles de base.
La cible est 1 si le patient virtuel a une pneumonie OU un AVC.
"""

import warnings
warnings.filterwarnings("ignore", message="Unknown solver options")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
IMG_CSV = ROOT / "data" / "image_agent_test_probs.csv"
STROKE_CSV = ROOT / "data" / "stroke_test_probs.csv"
OUTPUT_MODEL = ROOT / "models" / "consensus_lr.joblib"

SEED = 42
N_PAR_GROUPE_TRAIN = 1500 # patients virtuels par groupe pour entrainer
N_PAR_GROUPE_TEST = 500 # patients virtuels par groupe pour evaluer

VARIANTES = {
    "A : 2 probabilites": ["p_pneumonia", "p_stroke"],
    "B : + interaction": ["p_pneumonia", "p_stroke", "interaction"],
    "C : + interaction + age": ["p_pneumonia", "p_stroke", "interaction", "age"],
}

VARIANTE_RETENUE = "C : + interaction + age"

NOMS_GROUPES = {
    (0, 0): "sain",
    (1, 0): "pneumonie seule",
    (0, 1): "AVC seul",
    (1, 1): "les deux",
}

def fabriquer_patients(img_df, stroke_df, n_par_groupe, rng):
    """ Associe au hasard des radios et des dossiers, par groupe (stratifie). """
    groupes = []
    for (pneu, avc), nom in NOMS_GROUPES.items():
        radios = img_df[img_df["pneumonia"] == pneu].sample(
            n_par_groupe, replace=True, random_state=int(rng.integers(0, 10**9))
        )
        dossiers = stroke_df[stroke_df["avc"] == avc].sample(
            n_par_groupe, replace=True, random_state=int(rng.integers(0, 10**9))
        )
        groupes.append(
            pd.DataFrame(
                {
                    "p_pneumonia": radios["p_pneumonia"].values,
                    "p_stroke": dossiers["p_stroke"].values,
                    "age": dossiers["age"].values,
                    "pneumonia": pneu,
                    "avc": avc,
                    "groupe": nom,
                }
            )
        )

    patients = pd.concat(groupes, ignore_index=True)
    patients["interaction"] = patients["p_pneumonia"] * patients["p_stroke"]
    patients["cible"] = ((patients["pneumonia"] == 1) | (patients["avc"] == 1)).astype(int)
    return patients


#1. lecture des deux fichiers de probabilites (jeux de test des modeles  de base)

img = pd.read_csv(IMG_CSV).rename(
    columns={"true": "pneumonia", "prob_pneumonia": "p_pneumonia"}
)

stroke = pd.read_csv(STROKE_CSV).rename(
    columns={"stroke": "avc", "prob_stroke": "p_stroke"}
)

# 2. Chaque source est coupee en deux moities : une pour entrainer la regression,
#    une pour l'evaluer. Un meme patient reel n'est donc jamais des deux cotes.

img_train, img_test = train_test_split(
    img, test_size=0.5, stratify=img["pneumonia"], random_state=SEED
)

stroke_train, stroke_test = train_test_split(
    stroke, test_size=0.5, stratify=stroke["avc"], random_state=SEED
)

# 3. Fabrication des patients virtuels (4 groupes de taille egale)
rng = np.random.default_rng(SEED)
train = fabriquer_patients(img_train, stroke_train, N_PAR_GROUPE_TRAIN, rng)
test = fabriquer_patients(img_test, stroke_test, N_PAR_GROUPE_TEST, rng)

print(f"Patients virtuels :{len(train)} pour entrainer, {len(test)} pour evaluer")

#4. Entrainement et comparaison des trois variantes
modeles = {}
print("\n Resultats sur les patients virtuels de test (seuil 0.5)")
for nom, colonnes in VARIANTES.items():
    pipeline = Pipeline(
        [("echelle", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    )
    pipeline.fit(train[colonnes], train["cible"])
    proba = pipeline.predict_proba(test[colonnes])[:, 1]
    prediction = (proba >= 0.5).astype(int)
    print(
        f" {nom:26s} AUC={roc_auc_score(test['cible'], proba):.3f}  "
        f"rappel={recall_score(test['cible'], prediction):.3f}  "
        f"precision={precision_score(test['cible'], prediction):.3f}"
    )
    modeles[nom] = pipeline

# 5. Detail pour la variante retenue
retenue = modeles[VARIANTE_RETENUE]
colonnes = VARIANTES[VARIANTE_RETENUE]
test = test.assign(score=retenue.predict_proba(test[colonnes])[:, 1])
print(f"\nVariante retenue : {VARIANTE_RETENUE}")
print("Score moyen du consensus par type de patient virtuel :")
print(test.groupby("groupe")["score"].mean().round(3).to_string())
print("\nCoefficients (variables standardisees) :")
for nom , coef in zip(colonnes, retenue.named_steps["logreg"].coef_[0]):
    print(f"  {nom:14s} {coef:+.3f}")


# 6. Sauvegarde
joblib.dump(
    {"pipeline": retenue, "features": colonnes, "variante": VARIANTE_RETENUE},
    OUTPUT_MODEL,
)
print("\nModele sauvegarde :", OUTPUT_MODEL)

