"""
3_baseline.py — Entraînement et évaluation des modèles baseline

Ce script charge les données préparées par 2_preprocessing.py et entraîne
deux modèles baseline pour établir une première référence de performance :
  - Régression Logistique (modèle linéaire, interprétable)
  - Random Forest (modèle ensembliste, non-linéaire)

Pour chaque modèle, il affiche :
  - L'accuracy sur le jeu de test
  - Le rapport de classification (précision, rappel, F1-score par classe)
  - La matrice de confusion

Il inclut également une cross-validation 5-fold pour une estimation
plus robuste des performances sur l'ensemble du dataset.

Prérequis : avoir exécuté 2_preprocessing.py au préalable.
"""

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np


# ============================================================
# 1. CHARGEMENT DES DONNÉES PRÉPARÉES
# ============================================================

# Chargement des données préparées à l'étape précédente
data = joblib.load('data_preparees.pkl')

# Les données sont assignées à différentes variables
# X représente les caractéristiques (les indices) de nos textes :
# - X_train : utilisées par le modèle pour s'entraîner
# - X_test : gardées de côté pour évaluer le modèle à la fin
X_train       = data['X_train']
X_test        = data['X_test']

# y représente ce que le modèle doit deviner (les 3 classes d'erreurs) :
# - y_train : les "réponses" fournies au modèle pendant son entraînement
# - y_test : les vraies réponses, qui serviront à calculer la note finale du modèle
y_train       = data['y_train']
y_test        = data['y_test']

# Le nom explicite des 3 classes est récupéré pour l'affichage final
noms_classes  = data['noms_classes']

# Affichage récapitulatif
print(f"Données chargées depuis data_preparees.pkl")
print(f"  Train : {X_train.shape[0]} exemples, {X_train.shape[1]} features")
print(f"  Test  : {X_test.shape[0]} exemples")
print()


# ============================================================
# 2. DÉFINITION DES MODÈLES
# ============================================================
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


# ============================================================
# 3. ENTRAÎNEMENT ET ÉVALUATION SUR LE SPLIT TEST
# ============================================================

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


# ============================================================
# 4. CROSS-VALIDATION 5-FOLD
# ============================================================
# Pour éviter le risque d'instabilité d'un split 80/20 unique
# sur un petit dataset, la cross-validation entraîne 5 fois le modèle
# sur des partitions différentes et moyenne les scores pour une estimation plus robuste.

# Pour cette étape, les données entraînement et test sont réunies temporairement
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


# ============================================================
# RÉSUMÉ
# ============================================================
print("=" * 60)
print("Résumé : Comparaison des modèles (accuracy sur le split test)")
print("=" * 60)
for nom, acc in resultats.items():
    print(f"  {nom:30s} → Accuracy = {acc:.3f}")
print()
print("Lancer 4_tuning.py pour optimiser les hyperparamètres.")
