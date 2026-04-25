"""
scoring.py — Comparaison de configurations (dataset × méthode × modèle)

Ce script compare 4 combinaisons :
  1) Dataset base + modèles baseline
  2) Dataset enrichi + modèles baseline
  3) Dataset base + tuning (GridSearch)
  4) Dataset enrichi + tuning (GridSearch)

Référence: le pipeline `main` (dataset `.xlsx`, colonne `TypeErreur1`).
"""

from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

from feature_engineering import enrichir_dataset
from pipeline import pretraiter


DATASET_BASE_XLSX = "dataset_erreurs_reprises.xlsx"
DATASET_ENRICHI_XLSX = "dataset_enrichi.xlsx"


MAPPING_TARGET = {
    "E grammaticale": "Problème_Grammatical",
    "E antecedent": "Problème_Antecedent",
    "E reprise": "Problème_Reprise",
}


FEATURES_BASE_NUM = [
    "Distance_phrases",
    "Distance_mots",
    "Distance_caracteres",
    "GN_concurrents",
    "GN_concurrents_compatibles",
    "Similarite_reprise_antecedent",
]

FEATURES_BASE_CAT = [
    "Type_pronom",
    "Definitude_GN",
    "Fonction_reprise",
    "Fonction_antecedent",
]

FEATURES_ENRICHIES_NUM = FEATURES_BASE_NUM + [
    "Longueur_reprise",
    "Longueur_antecedent",
    "Match_genre",
    "Match_nombre",
    "Est_pronom",
]

FEATURES_ENRICHIES_CAT = FEATURES_BASE_CAT + [
    "Genre_reprise",
    "Nombre_reprise",
    "Genre_antecedent",
    "Nombre_antecedent",
    "Type_pronom_detaille",
]


def charger_xy_xlsx(path_xlsx: str, features_num: list[str], features_cat: list[str]):
    df = pd.read_excel(path_xlsx)
    df.columns = [c.strip().replace("\xa0", "") for c in df.columns]

    df["Classe_erreur"] = df["TypeErreur1"].map(MAPPING_TARGET)
    df = df.dropna(subset=["Classe_erreur"])

    features = features_num + features_cat
    X = df[features]
    y = df["Classe_erreur"]
    return X, y


def entrainer_et_scorer_baselines(X_train_prep, X_test_prep, y_train, y_test):
    modeles = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    scores = {}
    for nom, modele in modeles.items():
        modele.fit(X_train_prep, y_train)
        y_pred = modele.predict(X_test_prep)
        scores[nom] = (
            f1_score(y_test, y_pred, average="macro"),
            f1_score(y_test, y_pred, average="weighted"),
            balanced_accuracy_score(y_test, y_pred),
        )
    return scores


def optimiser_lr_rf_gb_gridsearch(X_train_prep, y_train):
    param_grid_lr = {
        "C": [0.01, 0.1, 1, 10],
        # 'liblinear' ne supporte pas le multi-classes (n_classes >= 3) en multinomial
        "solver": ["lbfgs", "newton-cholesky"],
        "class_weight": ["balanced", None],
        "max_iter": [2000],
    }
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid_lr,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_lr.fit(X_train_prep, y_train)

    param_grid_rf = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_rf.fit(X_train_prep, y_train)
    param_grid_gb = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0]
    }
    
    grid_gb = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid_gb,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1
    )
    grid_gb.fit(X_train_prep, y_train)

    return grid_lr.best_estimator_, grid_rf.best_estimator_, grid_gb.best_estimator_


def scorer_modele(modele, X_test_prep, y_test):
    y_pred = modele.predict(X_test_prep)
    return (
        f1_score(y_test, y_pred, average="macro"),
        f1_score(y_test, y_pred, average="weighted"),
        balanced_accuracy_score(y_test, y_pred),
    )


def main():
    print("[1/4] Dataset BASE + Modèles BASELINE")
    X_base, y_base = charger_xy_xlsx(DATASET_BASE_XLSX, FEATURES_BASE_NUM, FEATURES_BASE_CAT)
    Xb_train, Xb_test, yb_train, yb_test, _ = pretraiter(
        X_base, y_base, FEATURES_BASE_NUM, FEATURES_BASE_CAT
    )
    scores_base = entrainer_et_scorer_baselines(Xb_train, Xb_test, yb_train, yb_test)

    print("[2/4] Dataset ENRICHI + Modèles BASELINE")
    enrichir_dataset(DATASET_BASE_XLSX, DATASET_ENRICHI_XLSX)
    X_enr, y_enr = charger_xy_xlsx(DATASET_ENRICHI_XLSX, FEATURES_ENRICHIES_NUM, FEATURES_ENRICHIES_CAT)
    Xe_train, Xe_test, ye_train, ye_test, _ = pretraiter(
        X_enr, y_enr, FEATURES_ENRICHIES_NUM, FEATURES_ENRICHIES_CAT
    )
    scores_enr = entrainer_et_scorer_baselines(Xe_train, Xe_test, ye_train, ye_test)

    # [3/4] Dataset BASE + Tuning
    lr_base, rf_base, gb_base = optimiser_lr_rf_gb_gridsearch(Xb_train, yb_train)
    scores_base_tun = {
        "LogisticRegression": scorer_modele(lr_base, Xb_test, yb_test),
        "RandomForest": scorer_modele(rf_base, Xb_test, yb_test),
        "GradientBoosting": scorer_modele(gb_base, Xb_test, yb_test), # New
    }

    # [4/4] Dataset ENRICHI + Tuning
    lr_enr, rf_enr, gb_enr = optimiser_lr_rf_gb_gridsearch(Xe_train, ye_train)
    scores_enr_tun = {
        "LogisticRegression": scorer_modele(lr_enr, Xe_test, ye_test),
        "RandomForest": scorer_modele(rf_enr, Xe_test, ye_test),
        "GradientBoosting": scorer_modele(gb_enr, Xe_test, ye_test), # New
    }

    rows = []
    for modele, (f1m, f1w, bacc) in scores_base.items():
        rows.append(
            {
                "Dataset": "Base",
                "Methode": "Baseline",
                "Modele": modele,
                "F1_macro": f1m,
                "F1_weighted": f1w,
                "Balanced_accuracy": bacc,
            }
        )
    for modele, (f1m, f1w, bacc) in scores_enr.items():
        rows.append(
            {
                "Dataset": "Enrichi",
                "Methode": "Baseline",
                "Modele": modele,
                "F1_macro": f1m,
                "F1_weighted": f1w,
                "Balanced_accuracy": bacc,
            }
        )
    for modele, (f1m, f1w, bacc) in scores_base_tun.items():
        rows.append(
            {
                "Dataset": "Base",
                "Methode": "GridSearch",
                "Modele": modele,
                "F1_macro": f1m,
                "F1_weighted": f1w,
                "Balanced_accuracy": bacc,
            }
        )
    for modele, (f1m, f1w, bacc) in scores_enr_tun.items():
        rows.append(
            {
                "Dataset": "Enrichi",
                "Methode": "GridSearch",
                "Modele": modele,
                "F1_macro": f1m,
                "F1_weighted": f1w,
                "Balanced_accuracy": bacc,
            }
        )

    df_resultats = (
        pd.DataFrame(rows)
        .round(4)
        .sort_values(["Dataset", "Methode", "Modele"])
        .reset_index(drop=True)
    )

    print(df_resultats.to_string(index=False))
    df_resultats.to_csv("resultats_scoring2.csv", index=False)


if __name__ == "__main__":
    main()
