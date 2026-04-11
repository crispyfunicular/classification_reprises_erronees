# Projet 7 : Classification automatique de reprises erronées
> Morgane Bona-Pellissier, Salma Chatoui, Daria Tupikina et Sarah Yaya (Master 1 pluriTAL)

## Élaboration de la pipeline

### Étape 1 : création de la matrice (`construire_matrice()`)

> **Consigne :** *Récupérer un fichier csv auprès de Vanessa Gaudray Bouju. Le fichier contient : la reprise, son antécédent, le contexte antérieur, le type de reprise, et d'autres infos. L'objectif est de classer les reprises erronées en 3 classes (problèmes grammaticaux, problèmes avec l'antécédent, problèmes avec la reprise).*  
**Postulat** : chaque type d'erreur est corrélé à des variables spécifiques

- **Problèmes d'antécédent :** Le modèle se concentrera sur l'ambiguïté et l'éloignement (`GN_concurrents`, `GN_concurrents_compatibles`, `Distance_phrases`).
- **Problèmes de reprise :** Le modèle se concentrera sur la nature du mot et le sens (`Type_pronom`, `Definitude_GN`, `Similarite_reprise_antecedent`).
- **Problèmes grammaticaux :** Le modèle se concentrera sur la syntaxe (`Fonction_reprise`, `Fonction_antecedent`, `Distance_mots`).

Lors de cette première étape, la fonction `construire_matrice()` effectue la construction de la matrice initiale :

- **Chargement des données** brutes depuis le format Excel originel.
- **Création de la variable cible (Target)** en regroupant les valeurs initiales (`TypeErreur1`) en 3 classes de problèmes distinctes (`Problème_Grammatical`, `Problème_Antecedent`, `Problème_Reprise`) pour simplifier et cibler la prédiction. 
- **Suppression des valeurs** manquantes au niveau de la variable cible.
- **Sélection des features** à exploiter en distinguant explicitement les features numériques (distances, similarité sémantique, concurrences, etc.) et les features catégorielles (fonctions syntaxiques, types de pronoms, définitude, etc.).
- **Séparation** de la matrice des features `X` et du vecteur attendu `y`, transmis directement à l'étape suivante.

### Étape 2 : préparation des données (`pretraiter()`)

> **Consigne :** *Diviser le corpus en train/test.*

**Action technique :** Nous avons utilisé un `ColumnTransformer` (via Scikit-Learn) pour appliquer le One-Hot Encoding uniquement sur les variables catégorielles pertinentes, normaliser les distances, et exclure le texte brut de nos modèles de Machine Learning classiques.

À partir de la matrice générée, la fonction `pretraiter()` prépare les données pour la modélisation à venir :
- **Construction d'un `ColumnTransformer` (Pipeline de Pré-traitement)** permettant des traitements spécifiques selon le type de feature :
  - *Features numériques* : Traitement par imputation de la médiane (complète les données manquantes de manière robuste aux valeurs aberrantes), puis normalisation (`StandardScaler`).
  - *Features catégorielles* : Imputation des données manquantes par une valeur dédiée ("absent" — puisque cette information peut jouer un rôle utile pour le modèle) suivie de leur encodage en variables muettes (`OneHotEncoder`).
- **Séparation stratifiée du jeu de données (Train/Test Split)** (80%/20%), ce qui conserve la même proportion des classes cible entre les exemples dédiés à l'apprentissage (Train) et ceux pour l'évaluation finale (Test).
- **Application stricte du préprocesseur** (ajustement ou `fit`) exclusivement sur le corpus d'entraînement afin d'éviter toute fuite d'informations vers le corpus de test (Data Leakage). Les deux ensembles sont ensuite passés en transformation (`transform`).
- Les données prétraitées sont transmises directement à l'étape suivante.

### Étape 3 : entraînement et évaluation des modèles baseline (`evaluer_baselines()`)

La fonction `evaluer_baselines()` établit les premières performances de référence en entraînant deux modèles classiques sur les données préparées à l'étape 2 :
- **Définition de deux modèles de référence (Baseline)**, en appliquant un poids proportionnel aux classes (`class_weight='balanced'`) pour compenser leur déséquilibre :
  - *Régression Logistique* : modèle linéaire et interprétable.
  - *Random Forest* : modèle ensembliste et non-linéaire basé sur des arbres de décision.
- **Entraînement et évaluation sur le jeu de test** : chaque modèle est entraîné sur les données Train (`fit`), puis évalué sur le Test, produisant l'accuracy, un rapport détaillé de classification (précision, rappel, F1-score par classe) ainsi qu'une matrice de confusion.
- **Validation croisée (Cross-validation 5-fold)** : comme le jeu de données est restreint (299 exemples), le script effectue une validation croisée sur l'ensemble du jeu de données (5 partitions) afin d'obtenir une estimation plus robuste et stabilisée des performances, en mesurant le F1-score macro.

### Étape 4 : Optimisation et Analyse Qualitative des Erreurs (`amelioration.py`)

Afin de maximiser les performances de nos algorithmes et rendre le modèle intelligible pour l'analyse linguistique, nous avons créé un script dédié à l'optimisation, `amelioration.py`. Ce script procède aux étapes suivantes :

1. **Optimisation des hyperparamètres (Tuning)** : Utilisation de `GridSearchCV` (Validation Croisée sur 5 Folds) pour tester méthodiquement une grille exhaustive de paramètres sur la *Régression Logistique* (paramètres de régularisation `C`, `solver`) et le *Random Forest* (profondeur maximale, nombre d'arbres `n_estimators`, critère de séparation). Le but est de trouver la configuration maximisant le F1-score au-delà de la baseline existante.
2. **Interprétabilité (Feature Importance)** : Une fois le meilleur modèle non-linéaire (Random Forest) entraîné, nous extrayons l'importance relative de chaque caractéristique (Feature Importance). Cette étape permet de justifier notre **postulat initial** : le modèle se base effectivement majoritairement sur les distances (en caractères, mots, phrases) et sur la similarité sémantique plutôt que de deviner au hasard.
3. **Analyse Qualitative des Erreurs** : L'évaluation mathématique (Accuracy, F1-score) est indispensable mais insuffisante en NLP structuré. Pour répondre au besoin qualitatif, le script exporte un document de diagnostic (`analyse_erreurs.html`). Ce rapport confronte dynamiquement la classe attendue ("Vraie Classe") et l'erreur du classifieur ("Classe Prédite"). Il intègre le contexte textuel brut avec surlignage interactif de l'antécédent et de la reprise, facilitant l'exploration et la formulation de nouvelles hypothèses linguistiques.## Discussion des résultats


## Conclusion


## Bibliographie

