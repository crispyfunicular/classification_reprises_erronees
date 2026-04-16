import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings

# Masquage des avertissements
warnings.filterwarnings("ignore", category=UserWarning)

# Import des fonctions locales (assure-toi que pipeline.py et visualisation.py sont dans le même dossier)
from pipeline import construire_matrice, pretraiter
from visualisation import exporter_erreurs_html

def optimiser_modeles(X_train, y_train):
    """
    Entraîne et optimise la Régression Logistique ET le Random Forest.
    Renvoie un dictionnaire contenant les deux meilleurs modèles trouvés.
    """
    modeles_optimises = {}

    # 1. RÉGRESSION LOGISTIQUE
    print("=== Optimisation de la Régression Logistique ===")
    param_grid_lr = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs", "saga"],
        "class_weight": ["balanced", None],
        "max_iter": [5000],
    }
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid_lr,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )
    grid_lr.fit(X_train, y_train)
    print(f"Meilleurs paramètres LR : {grid_lr.best_params_}")
    modeles_optimises["Logistic_Regression"] = grid_lr.best_estimator_

    # 2. RANDOM FOREST
    print("\n=== Optimisation du Random Forest ===")
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
    grid_rf.fit(X_train, y_train)
    print(f"Meilleurs paramètres RF : {grid_rf.best_params_}")
    modeles_optimises["Random_Forest"] = grid_rf.best_estimator_

    return modeles_optimises

def afficher_importance_detaillee(nom_modele, modele, preprocesseur, f_num, f_cat):
    """
    Affiche l'importance des variables selon le type de modèle.
    """
    print(f"\n--- INTERPRÉTABILITÉ : {nom_modele} ---")
    
    # Récupération des noms de colonnes après OneHotEncoding
    ohe = preprocesseur.named_transformers_["cat"].named_steps["onehot"]
    cat_features_encoded = list(ohe.get_feature_names_out(f_cat))
    all_features = f_num + cat_features_encoded

    if hasattr(modele, "feature_importances_"):
        # Cas Random Forest
        importances = modele.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("Top 10 des variables (Feature Importance) :")
        for i in range(min(10, len(importances))):
            print(f"{i+1:2d}. {all_features[indices[i]]:40s} ({importances[indices[i]]:.4f})")

    elif hasattr(modele, "coef_"):
        # Cas Régression Logistique (Coefficients)
        # On prend la moyenne des coefficients absolus sur les 3 classes pour avoir une idée globale
        abs_coefs = np.mean(np.abs(modele.coef_), axis=0)
        indices = np.argsort(abs_coefs)[::-1]
        print("Top 10 des variables (Coefficients moyens absolus) :")
        for i in range(min(10, len(abs_coefs))):
            print(f"{i+1:2d}. {all_features[indices[i]]:40s} ({abs_coefs[indices[i]]:.4f})")

# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    # 1. Chargement et préparation
    X, y, features_num, features_cat = construire_matrice(
        "/Users/dariatupikina/dataset_erreurs_reprises.xlsx"
    )
    X_train, X_test, y_train, y_test, preprocesseur = pretraiter(
        X, y, features_num, features_cat
    )

    # 2. Entraînement des deux modèles
    print("\n--- ÉTAPE 2 : TUNING DES HYPERPARAMÈTRES ---")
    dictionnaire_modeles = optimiser_modeles(X_train, y_train)

    # 3. Évaluation croisée
    for nom, modele in dictionnaire_modeles.items():
        print(f"\n" + "="*50)
        print(f"ANALYSE DU MODÈLE : {nom}")
        print("="*50)

        # Prédictions
        y_pred = modele.predict(X_test)
        
        # Rapport de performance
        print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.3f}")
        print("\nRapport de classification :")
        print(classification_report(y_test, y_pred))

        # Importance des variables
        afficher_importance_detaillee(nom, modele, preprocesseur, features_num, features_cat)

        # Export HTML (un fichier par modèle)
        nom_fichier = f"analyse_erreurs_{nom}.html"
        exporter_erreurs_html(y_test, y_pred, y_test.index, nom_fichier)
        print(f"\nFichier diagnostic généré : {nom_fichier}")

    print("\n--- TRAITEMENT TERMINÉ ---")