"""
amelioration.py - Script d'optimisation et d'analyse des erreurs

Ce script vient compléter le script de base (pipeline.py) pour accomplir trois tâches avancées :
1. Optimisation (Tuning) : Tester de multiples réglages (hyperparamètres) pour trouver la meilleure version possible des modèles.
2. Interprétabilité : Identifier les variables linguistiques (features) ayant le plus influencé la décision du modèle.
3. Analyse qualitative : Exporter un diagnostic visuel (HTML) des erreurs de classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Masquage des avertissements techniques pour garder une sortie console propre
warnings.filterwarnings("ignore", category=UserWarning)

# Import des fonctions de préparation de données depuis le script principal
from pipeline import construire_matrice, pretraiter

# Import de la fonction d'interface visuelle depuis le module dédié
from visualisation import exporter_erreurs_html

SCORING_METRICS = ["f1_macro", "f1_weighted", "balanced_accuracy"]


def optimiser_modeles(X_train, y_train, scoring):
    """
    Au lieu de se contenter des réglages par défaut, cette fonction utilise
    une grille de recherche (GridSearchCV). Elle va créer, entraîner et évaluer
    des dizaines de variantes de chaque algorithme pour retenir le plus performant.
    """

    # RÉGRESSION LOGISTIQUE
    print("=== Optimisation de la Régression Logistique ===")
    # param_grid_lr définit le catalogue des réglages à essayer pour la régression logistique.
    # Ex: 'C' contrôle la force de la pénalité (afin de limiter le surapprentissage).
    param_grid_lr = {
        # Différents niveaux de régularisation
        "C": [0.01, 0.1, 1, 10],
        # Différentes méthodes mathématiques de résolution
        "solver": ["newton-cholesky", "lbfgs"],
        # Avec ou sans rééquilibrage automatique des classes
        "class_weight": ["balanced", None],
        # Assez de temps pour converger
        "max_iter": [2000],
    }

    # GridSearchCV s'occupe de croiser tous ces paramètres.
    # cv=5 : chaque combinaison est évaluée avec une validation croisée en 5 parties (5-fold)
    # scoring='f1_macro' : l'évaluation privilégie la capacité du modèle à être bon sur toutes les classes d'erreurs, même les plus rares.
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid_lr,
        cv=5,
         scoring=scoring,
        n_jobs=-1,
    )
    grid_lr.fit(X_train, y_train)
    print(f"Meilleurs paramètres LR : {grid_lr.best_params_}")
    print(f"Meilleur score F1 (CV)  : {grid_lr.best_score_:.3f}")

    # RANDOM FOREST
    print("\n=== Optimisation du Random Forest ===")
    # param_grid_rf définit le catalogue des réglages pour l'algorithme Random Forest.
    param_grid_rf = {
        # Nombre d'arbres décisionnels dans la forêt
        "n_estimators": [50, 100, 200],
        # Profondeur maximale d'un arbre (= sa complexité)
        "max_depth": [
            None,
            10,
            20,
        ],
        # Nombre minimal d'exemples requis pour créer une nouvelle règle
        "min_samples_split": [
            2,
            5,
            10,
        ],
        "class_weight": ["balanced", "balanced_subsample"],
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid_rf,
        cv=5,
        scoring=scoring,
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    print(f"Meilleurs paramètres RF : {grid_rf.best_params_}")
    print(f"Meilleur score F1 (CV)  : {grid_rf.best_score_:.3f}")

    return grid_lr.best_estimator_, grid_rf.best_estimator_

def afficher_feature_importance(modele, preprocesseur, f_num, f_cat):
    """
    Extrait le classement des variables. Cela répond à la question :
    "Parmi les distances, les GN concurrents ou les types de pronoms, quelle
    information a le plus aidé l'algorithme à détecter la bonne classe d'erreur ?"
    """

    # L'attribut 'feature_importances_' n'est disponible que pour certains algorithmes (comme Random Forest).
    if hasattr(modele, "feature_importances_"):

        # Récupération de l'encodeur catégoriel (OneHotEncoder) utilisé lors du prétraitement.
        # Cela permet de retrouver le nom exact des colonnes catégorielles éclatées (ex: 'Type_pronom_personnel')
        ohe = preprocesseur.named_transformers_["cat"].named_steps["onehot"]
        cat_features_encoded = ohe.get_feature_names_out(f_cat)

        # Fusion des noms des variables numériques et catégorielles respectives
        all_features = f_num + list(cat_features_encoded)

        # Tri des variables de la plus influente à la moins influente
        importances = modele.feature_importances_
        indices = np.argsort(importances)[::-1]

        print(
            "\n=== Top 15 des variables les plus discriminantes (Feature Importance) ==="
        )
        for i in range(min(15, len(importances))):
            print(
                f"{i+1:2d}. {all_features[indices[i]]:40s} ({importances[indices[i]]:.4f})"
            )
    # Si le modèle gagnant est une Régression Logistique, cette méthode d'interprétabilité ne s'applique pas telle quelle.
    elif hasattr(modele, "coef_"):
        print(
            "\nNote: Le modèle choisi (Régression logistique) utilise des coefficients linéaires (coef_), son extraction d'importance nécessite une approche différente non affichée ici."
        )


# ============================================================
# EXÉCUTION DE L'AMÉLIORATION
# ============================================================

if __name__ == "__main__":
    print("--- ÉTAPE 1 : MATRICE ET PRÉTRAITEMENT ---")
    # L'appel direct à construire_matrice() et pretraiter() économise l'écriture de code
    # en s'appuyant sur les modules fiables du premier fichier.
    X, y, features_num, features_cat = construire_matrice(
        "dataset_erreurs_reprises.xlsx"
    )
    X_train, X_test, y_train, y_test, preprocesseur = pretraiter(
        X, y, features_num, features_cat
    )

    print("\n\n--- ÉTAPE 2 : TUNING DES HYPERPARAMÈTRES (GRID SEARCH) ---")
    best_model, best_name = optimiser_modeles(X_train, y_train)

    print(f"\n\n--- ÉTAPE 3 : ÉVALUATION FINALE ({best_name}) ---")
    # Une fois les paramètres optimaux trouvés, la prédiction finale est affichée
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy sur jeu de Test : {acc:.3f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))

    print("\n--- ÉTAPE 4 : INTERPRÉTABILITÉ (FEATURE IMPORTANCE) ---")
    afficher_feature_importance(best_model, preprocesseur, features_num, features_cat)

    print("\n--- ÉTAPE 5 : EXPORT DES ERREURS POUR ANALYSE QUALITATIVE ---")
    # L'identifiant (index) d'origine des phrases de test est conservé et transmis
    # afin que le script sache précisément quelles phrases lier aux mathématiques du modèle.
    exporter_erreurs_html(y_test, y_pred, y_test.index, "analyse_erreurs.html")
