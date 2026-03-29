"""
2_preprocessing.py — Prétraitement et split train/test

Ce script prépare les données avant la modélisation :
  - Chargement du dataset et création de la target (= la variable à prédire, construite
    en regroupant la colonne TypeErreur1 en 3 classes : Problème_Grammatical,
    Problème_Antecedent, Problème_Reprise), chargée depuis le fichier matrice.pkl produit par 1_matrice.py
  - Construction d'un ColumnTransformer (outil scikit-learn qui applique des
    transformations différentes selon le type de colonne) pour traiter séparément :
      * les features numériques (distances entre reprise et antécédent, nombre de
        GN concurrents, similarité sémantique) -> imputation par la médiane + normalisation
      * les features catégorielles (type de pronom, définitude du GN, fonctions
        syntaxiques de la reprise et de l'antécédent) -> imputation par "absent" + One-Hot Encoding.
           - Imputation : On remplace les cases vides (NaN) par le mot "absent", car
             l'absence d'un pronom ou d'un GN est en soi une information utile pour notre modèle et
             jeter ces lignes nous ferait perdre de précieuses données.
           - One-Hot Encoding : Les algorithmes ne lisent que des nombres. Cette étape
             transforme nos colonnes texte en de multiples colonnes de 0 et 1.
             Ex : la colonne "Type_pronom" devient 3 colonnes :
             "est-un-pronom-personnel", "est-un-pronom-demonstratif", "est-un-pronom-absent".)
  - Split train/test stratifié (80/20)
  - Sauvegarde des données préparées dans data_preparees.pkl, qui sera ensuite utilisé
  par les scripts suivants pour éviter de re-traiter le dataset à chaque étape.
"""

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# ============================================================
# 1. CHARGEMENT DE LA MATRICE (produite par 1_matrice.py)
# ============================================================
# On charge X et y depuis le fichier matrice.pkl produit par 1_matrice.py.
# Prérequis : avoir exécuté 1_matrice.py au préalable.
matrice = joblib.load("matrice.pkl")

X = matrice["X"]
y = matrice["y"]
features_numeriques = matrice["features_numeriques"]
features_categorielles = matrice["features_categorielles"]

print(
    f"Dataset chargé depuis matrice.pkl : {X.shape[0]} exemples, {X.shape[1]} features"
)
print(f"Distribution de la target :")
print(y.value_counts().to_string())
print()


# ============================================================
# 2. PRÉTRAITEMENT — ColumnTransformer
# ============================================================
# Les features numériques reçoivent une imputation par la médiane
# (robuste aux valeurs aberrantes) suivie d'une normalisation.
# Rappel : Les features catégorielles reçoivent :
#   - Une imputation par "absent" : On remplace les cases vides (NaN) par le mot "absent",
#     car l'absence d'un pronom ou d'un GN est en soi une information utile pour notre modèle et
#     jeter ces lignes nous ferait perdre de précieuses données.
#   - Un One-Hot Encoding : Les algorithmes ne lisent que des nombres. Cette étape
#     transforme nos colonnes texte en de multiples colonnes de 0 et 1 (True/False).
#     Ex : la colonne "Type_pronom" devient 3 colonnes : "est-un-pronom-personnel",
#     "est-un-pronom-demonstratif", "est-un-pronom-absent".

preprocesseur = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            features_numeriques,
        ),
        (
            "cat",
            Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="absent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            ),
            features_categorielles,
        ),
    ]
)


# ============================================================
# 3. SPLIT TRAIN/TEST (80/20, stratifié)
# ============================================================
# stratify=y garantit que la proportion de chaque classe est
# conservée dans le train et le test.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Split train/test :")
print(f"  Train : {X_train.shape[0]} exemples")
print(f"  Test  : {X_test.shape[0]} exemples")
print()


# ============================================================
# 4. FIT DU PRÉPROCESSEUR ET TRANSFORMATION
# ============================================================
# - "fit" = phase d'ajustement du préprocesseur,
# qui analyse les données pour calculer les statistiques nécessaires
# aux transformations (ex. calcul de la médiane pour l'imputation, identification
# des catégories existantes pour l'encodage).
# /!\ effectué exclusivement sur le jeu d'entraînement.
#
# - "transform" -> appliqué ensuite au train et au test en utilisant
# les statistiques communes.
# Cela garantit que le modèle n'a accès à aucune information
# provenant du jeu de test lors de son apprentissage (prévention du "data leakage").

X_train_prep = preprocesseur.fit_transform(X_train)
X_test_prep = preprocesseur.transform(X_test)

print(f"Dimensions après prétraitement :")
print(f"  X_train : {X_train_prep.shape}")
print(f"  X_test  : {X_test_prep.shape}")
print()


# ============================================================
# 5. SAUVEGARDE
# ============================================================
# On sauvegarde les données prétraitées dans un fichier binaire (.pkl) pour
# éviter d'avoir à relancer le pipeline de nettoyage à chaque entraînement de modèle (étape 3).
# On conserve également  le "preprocesseur" ajusté  sur le jeu d'entraînement.

joblib.dump(
    {
        "X_train": X_train_prep,
        "X_test": X_test_prep,
        "y_train": y_train,
        "y_test": y_test,
        "preprocesseur": preprocesseur,
        "noms_classes": [
            "Problème_Grammatical",
            "Problème_Antecedent",
            "Problème_Reprise",
        ],
    },
    "data_preparees.pkl",
)

print("Données sauvegardées dans data_preparees.pkl")
print("Lancer 3_baseline.py pour entraîner les modèles.")
