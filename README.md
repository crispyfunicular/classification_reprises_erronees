# Classification des reprises anaphoriques erronées

## Objectif
Ce projet est réalisé dans le cadre d'un travail de groupe (4 personnes) en Traitement du Langage Naturel (NLP) / Machine Learning. 

L'objectif principal est de **détecter et classer automatiquement des erreurs de reprises anaphoriques** dans des textes en français. À partir d'un jeu de données annoté, nous devons construire un pipeline de Machine Learning capable de classer ces erreurs en **3 grandes catégories (Target)** :
1. **Problèmes grammaticaux** (ex: erreurs d'accord en genre/nombre).
2. **Problèmes avec l’antécédent** (ex: antécédent flou, absent, ou trop de GN concurrents).
3. **Problèmes avec la reprise** (ex: mauvais choix du type de pronom).


## Plan d'action

### Préparation des données
- **Création de la Target :** Mapper les colonnes `TypeErreur` / `SousTypeErreur` existantes pour créer l'unique colonne cible contenant nos 3 classes.
- **Nettoyage :** Gestion des valeurs manquantes.
- **Encodage :** Transformation des variables textuelles/catégorielles en valeurs numériques (ex: `Type_pronom`, `Fonction_reprise`).
- **Analyse (EDA) :** Vérification de l'équilibre des classes.

### Architecture
- **Split Train/Test :** Division du corpus en données d'entraînement et de test (en utilisant `stratify` pour conserver la proportion des 3 classes).
- **Baseline :** Création d'un premier modèle simple (ex: Régression Logistique ou Arbre de Décision) pour valider le bon fonctionnement du pipeline.

### Modélisation
- **Sélection des features :** Identifier quelles caractéristiques linguistiques sont les plus pertinentes pour différencier les 3 types d'erreurs.
- **Entraînement :** Tester et optimiser des classifieurs plus performants (Random Forest, XGBoost, SVM).
- **Tuning :** Optimisation des hyperparamètres pour maximiser les performances.

### Évaluation
- **Évaluation fine :** Génération des métriques de classification (Précision, Rappel, F1-Score par classe).
- **Analyse des erreurs :** Création de la matrice de confusion et analyse qualitative (Pourquoi le modèle se trompe-t-il sur certaines phrases ?).
- *(Bonus)* **Modèle de langage :** Tester un modèle de Deep Learning (type CamemBERT) utilisant uniquement le texte brut (`Contexte`).
