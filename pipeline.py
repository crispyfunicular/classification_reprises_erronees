"""
pipeline.py — Pipeline de classification des reprises erronées

Ce script enchaîne les trois étapes de la pipeline :
  1. Construction de la matrice de features à partir du dataset Excel
  2. Prétraitement (imputation, encodage, normalisation) et split train/test
  3. Entraînement et évaluation de modèles baseline (Régression Logistique, Random Forest)

Usage :
    python pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ============================================================
# 1. CONSTRUCTION DE LA MATRICE DE FEATURES
# ============================================================

def construire_matrice(chemin_excel):
    """
    Charge le dataset brut (Excel) des erreurs de reprises anaphoriques,
    crée la variable cible (Target) à 3 classes à partir de TypeErreur1, et
    construit la matrice X (features numériques + catégorielles) ainsi que le
    vecteur y (classes d'erreurs). Affiche un aperçu des dimensions, de la
    distribution de la target et des valeurs manquantes.
    """
    df = pd.read_excel(chemin_excel)

    # Création de la Target (= la variable que l'on cherche à prédire)
    # Les valeurs de TypeErreur1 sont renommées en 3 classes plus lisibles :
    mapping_target = {
        # E grammaticale : erreur accord, redoublement référence, mauvais usage pronom relatif...
        'E grammaticale': 'Problème_Grammatical',
        # E antecedent : ambiguïté, antécédent non inférable, antécédent trop distant...
        'E antecedent':   'Problème_Antecedent',
        # E reprise : mauvais choix lexical, flou, répétition et redondance...
        'E reprise':      'Problème_Reprise'
    }

    # .map() remplace chaque valeur de TypeErreur1 par la classe correspondante
    # Si la valeur n'est pas dans le dictionnaire, .map() renvoie NaN
    df['Classe_erreur'] = df['TypeErreur1'].map(mapping_target)

    # Suppression des lignes où TypeErreur1 est vide (NaN)
    df = df.dropna(subset=['Classe_erreur'])

    # -- Définition des features --

    features_numeriques = [
        'Distance_phrases',
        'Distance_mots',
        'Distance_caracteres',
        'GN_concurrents',
        'GN_concurrents_compatibles',
        'Similarite_reprise_antecedent'
    ]

    features_categorielles = [
        'Type_pronom',
        'Definitude_GN',
        'Fonction_reprise',
        'Fonction_antecedent'
    ]

    colonnes_utiles = features_numeriques + features_categorielles

    # X = le tableau des features = la matrice
    # y = la colonne des réponses attendues = ce que le modèle doit prédire
    X = df[colonnes_utiles]
    y = df['Classe_erreur']

    # APERÇU
    print(f"Dimensions de la matrice X : {X.shape}")
    print(f"Dimensions de y : {y.shape}")
    print()
    print("Distribution de la Target")
    print(y.value_counts())
    print()
    print("Valeurs manquantes par feature")
    print(X.isna().sum())
    print()
    print("Aperçu de la matrice (5 premières lignes)")
    print(X.head().to_string())
    print()

    return X, y, features_numeriques, features_categorielles


# ============================================================
# 2. PRÉTRAITEMENT ET SPLIT TRAIN/TEST
# ============================================================

def pretraiter(X, y, features_numeriques, features_categorielles):
    """
    Prépare les données avant la modélisation :
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
    """

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

    # -- Split train/test (80/20, stratifié) --
    # stratify=y garantit que la proportion de chaque classe est
    # conservée dans le train et le test.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Split train/test :")
    print(f"  Train : {X_train.shape[0]} exemples")
    print(f"  Test  : {X_test.shape[0]} exemples")
    print()

    # -- Fit du préprocesseur et transformation --
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

    return X_train_prep, X_test_prep, y_train, y_test, preprocesseur


# ============================================================
# 3. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES BASELINE
# ============================================================

def evaluer_baselines(X_train, X_test, y_train, y_test):
    """
    Entraîne deux modèles baseline pour établir une première référence de performance :
      - Régression Logistique (modèle linéaire, interprétable)
      - Random Forest (modèle ensembliste, non-linéaire)

    Pour chaque modèle, affiche :
      - L'accuracy sur le jeu de test
      - Le rapport de classification (précision, rappel, F1-score par classe)
      - La matrice de confusion

    Inclut également une cross-validation 5-fold pour une estimation
    plus robuste des performances sur l'ensemble du dataset.
    """

    noms_classes = [
        'Problème_Grammatical',
        'Problème_Antecedent',
        'Problème_Reprise',
    ]

    # -- Définition des modèles --
    # class_weight='balanced' compense le déséquilibre entre les 3 classes
    # (133 Antecedent / 92 Reprise / 74 Grammatical)

    # Dictionnaire contenant les deux algorithmes de référence :
    # régression logistique & random forest
    # Chacun apprend à repérer les classes d'erreurs.
    modeles = {
        # La régression logistique trace des frontières "droites" entre les différentes classes d'erreurs (plus simple).
        'Régression Logistique': LogisticRegression(
            max_iter=1000,           # Assez de temps (itérations) est laissé pour trouver la meilleure solution
            class_weight='balanced', # L'importance des classes est rééquilibrée pour ne pas ignorer les erreurs les plus rares
            random_state=42          # Le hasard est figé pour obtenir les mêmes résultats à chaque exécution
        ),

        # Le Random Forest crée des centaines de petits arbres de décision et garde le vote majoritaire (plus complexe).
        'Random Forest': RandomForestClassifier(
            n_estimators=100,        # 100 arbres de décision différents sont utilisés pour former la forêt
            class_weight='balanced', # Une plus grande attention est accordée aux erreurs peu fréquentes
            random_state=42          # Le hasard est également figé
        )
    }

    # -- Entraînement et évaluation sur le split test --

    # Dictionnaire pour mémoriser le score global de chaque algorithme
    resultats = {}

    # Chaque modèle (Régression puis Random Forest) est évalué un par un
    for nom, classifieur in modeles.items():
        # 1. Phase d'apprentissage (l'entraînement) : le modèle cherche les régularités
        # reliant les caractéristiques (X_train) aux bonnes classes d'erreurs (y_train)
        classifieur.fit(X_train, y_train)

        # 2. Phase de test (l'examen) : des données jamais vues par le modèle lui sont fournies (X_test)
        # afin qu'il devine la classe d'erreur pour chacun des exemples
        y_pred = classifieur.predict(X_test)

        # 3. Évaluation globale : le pourcentage de bonnes réponses (accuracy) est calculé
        # en comparant ses prédictions (y_pred) avec la réalité (y_test)
        accuracy = accuracy_score(y_test, y_pred)
        resultats[nom] = accuracy

        # 4. Affichage des performances détaillées
        # "Accuracy" = réussite globale
        print("=" * 60)
        print(f"Modèle : {nom}")
        print(f"Accuracy : {accuracy:.3f}")
        print()

        # Le rapport de classification donne le détail de réussite pour chaque type d'erreur.
        # Utile pour savoir si le modèle est bon en grammaire mais mauvais sur les antécédents (par exemple).
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred, target_names=noms_classes))

        # La matrice de confusion montre exactement les erreurs d'inattention du modèle.
        # Ex: combien de fois il a répondu "Problème_Reprise" alors que c'était un "Problème_Grammatical".
        print("Matrice de confusion :")
        print(f"  (lignes = réel, colonnes = prédit)")
        print(f"  Classes : {noms_classes}")
        print(confusion_matrix(y_test, y_pred, labels=noms_classes))
        print()

    # -- Cross-validation 5-fold --
    # Pour éviter le risque d'instabilité d'un split 80/20 unique
    # sur un petit dataset, la cross-validation entraîne 5 fois le modèle
    # sur des partitions différentes et moyenne les scores pour une estimation plus robuste.

    # Les données entraînement et test sont réunies temporairement
    # pour avoir le plus grand nombre d'exemples possible sous la main.
    X_all = np.vstack([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    print("=" * 60)
    print("CROSS-VALIDATION 5-FOLD (sur l'ensemble du dataset)")
    print("=" * 60)
    print()

    # Les données sont mélangées (shuffle=True) pour éviter les biais
    # puis réparties en 5 parts égales ("5-folds").
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Le modèle apprend avec 4 parts et est évalué sur la 5e.
    # L'opération est répétée 5 fois.
    # -> chaque part sert de test une fois à tour de rôle.
    for nom, classifieur in modeles.items():
        # 'f1_macro' = métrique d'évaluation
        # -> moyenne stricte qui oblige le modèle à être bon partout,
        # y compris sur les classes d'erreurs les plus rares.
        scores = cross_val_score(classifieur, X_all, y_all, cv=cv, scoring='f1_macro')

        print(f"{nom} :")
        # Les notes des 5 évaluations, suivies de la moyenne (avec sa marge d'erreur).
        print(f"  F1-macro par fold : {[f'{s:.3f}' for s in scores]}")
        print(f"  F1-macro moyen    : {scores.mean():.3f} (± {scores.std():.3f})")
        print()

    # -- Résumé --
    print("=" * 60)
    print("Résumé : Comparaison des modèles (accuracy sur le split test)")
    print("=" * 60)
    for nom, acc in resultats.items():
        print(f"  {nom:30s} → Accuracy = {acc:.3f}")
    print()


# ============================================================
# EXÉCUTION
# ============================================================

if __name__ == "__main__":
    X, y, features_num, features_cat = construire_matrice("dataset_erreurs_reprises.xlsx")
    X_train, X_test, y_train, y_test, preprocesseur = pretraiter(X, y, features_num, features_cat)
    evaluer_baselines(X_train, X_test, y_train, y_test)
