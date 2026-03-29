"""
1_matrice.py — Construction de la matrice de features

Ce script charge le dataset brut (Excel) des erreurs de reprises anaphoriques,
crée la variable cible (Target) à 3 classes à partir de TypeErreur1, et
construit la matrice X (features numériques + catégorielles) ainsi que le
vecteur y (classes d'erreurs). Il affiche un aperçu des dimensions, de la
distribution de la target et des valeurs manquantes.

Produit : matrice.pkl (X et y bruts, avant prétraitement) utilisé par
2_preprocessing.py pour éviter de relire et re-traiter le fichier Excel.
"""

import pandas as pd
import joblib

# CHARGEMENT DES DONNÉES (à partir du fichier Excel)
df = pd.read_excel('dataset_erreurs_reprises.xlsx')

# Création de la Target (= la variable que l'on cherche à prédire)
# On renomme les valeurs de TypeErreur1 en 3 classes plus lisibles :
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

# DÉFINITION DES FEATURES
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

# CONSTRUCTION DE LA MATRICE
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

# SAUVEGARDE
# On exporte X et y bruts (avant tout prétraitement) dans un fichier
# intermédiaire, pour que 2_preprocessing.py n'ait pas à relire le
# fichier Excel ni à recréer la target depuis zéro.
joblib.dump({
    'X': X,
    'y': y,
    'features_numeriques':    features_numeriques,
    'features_categorielles': features_categorielles
}, 'matrice.pkl')

print()
print("✅ Matrice sauvegardée dans matrice.pkl")
print("   → Lancer 2_preprocessing.py pour prétraiter et splitter les données.")