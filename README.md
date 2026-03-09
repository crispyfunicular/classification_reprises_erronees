# Classification des reprises anaphoriques erronées

## Objectif
Ce projet est réalisé dans le cadre d'un travail de groupe (4 personnes) en Traitement du Langage Naturel (NLP) / Machine Learning. 

L'objectif principal est de **détecter et classer automatiquement des erreurs de reprises anaphoriques** dans des textes en français. À partir d'un jeu de données annoté, nous devons construire un pipeline de Machine Learning capable de classer ces erreurs en **3 grandes catégories (Target)** :
1. **Problèmes grammaticaux** (ex: erreurs d'accord en genre/nombre).
2. **Problèmes avec l’antécédent** (ex: antécédent flou, absent, ou trop de GN concurrents).
3. **Problèmes avec la reprise** (ex: mauvais choix du type de pronom).


## Plan d'action

### 1. Préparation des données
**Postulat** : chaque type d'erreur est corrélé à des variables spécifiques

- **Problèmes d'antécédent :** Le modèle se concentrera sur l'ambiguïté et l'éloignement (`GN_concurrents`, `GN_concurrents_compatibles`, `Distance_phrases`).
- **Problèmes de reprise :** Le modèle se concentrera sur la nature du mot et le sens (`Type_pronom`, `Definitude_GN`, `Similarite_reprise_antecedent`).
- **Problèmes grammaticaux :** Le modèle se concentrera sur la syntaxe (`Fonction_reprise`, `Fonction_antecedent`, `Distance_mots`).

**Action technique :** Nous utiliserons un `ColumnTransformer` (via Scikit-Learn) pour appliquer le One-Hot Encoding uniquement sur les variables catégorielles pertinentes, normaliser les distances, et exclure le texte brut de nos modèles de Machine Learning classiques.

Par conséquent, avant la modélisation, le jeu de données brut nécessite une mise en forme :
- **Création de la Target (Variable Cible) :** Regroupement (mapping) des annotations détaillées (`TypeErreur`, `SousTypeErreur`) pour former notre variable cible unique à 3 classes (Grammaire, Antécédent, Reprise).
- **Nettoyage :** Traitement des valeurs manquantes et exclusion des métadonnées inutiles.
- **Analyse Exploratoire (EDA) :** Étude de la distribution de la cible (vérification de l'équilibre des 3 classes) et observation des corrélations initiales.


### 2. Architecture
- **Split Train/Test :** Division du corpus en données d'entraînement et de test (en utilisant `stratify` pour conserver la proportion des 3 classes).
- **Baseline :** Création d'un premier modèle simple (ex: Régression Logistique ou Arbre de Décision) pour valider le bon fonctionnement du pipeline.

### 3. Modélisation
- **Sélection des features :** Identifier quelles caractéristiques linguistiques sont les plus pertinentes pour différencier les 3 types d'erreurs.
- **Entraînement :** Tester et optimiser des classifieurs plus performants (Random Forest, XGBoost, SVM).
- **Tuning :** Optimisation des hyperparamètres pour maximiser les performances.

### 4. Évaluation
- **Évaluation fine :** Génération des métriques de classification (Précision, Rappel, F1-Score par classe).
- **Analyse des erreurs :** Création de la matrice de confusion et analyse qualitative (Pourquoi le modèle se trompe-t-il sur certaines phrases ?).
- *(Bonus)* **Modèle de langage :** Tester un modèle de Deep Learning (type CamemBERT) utilisant uniquement le texte brut (`Contexte`).
