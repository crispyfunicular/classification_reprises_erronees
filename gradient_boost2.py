import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from visualisation import exporter_erreurs_html

def entrainer_classifieur(chemin_fichier):
    # 1. Chargement du dataset
    df = pd.read_excel(chemin_fichier)
    
    # 2. Application de ton mapping de classes
    mapping_target = {
        'E grammaticale': 'Problème_Grammatical',
        'E antecedent':   'Problème_Antecedent',
        'E reprise':      'Problème_Reprise'
    }
    
    # Création de la colonne cible propre et suppression des lignes vides ou non mappées
    df['Target_Class'] = df['TypeErreur1'].map(mapping_target)
    df = df.dropna(subset=['Target_Class'])

    # 3. Sélection des features selon ta liste exacte
    features_numeriques = [
        'Distance_caracteres', 
        'Distance_mots', 
        'Distance_phrases',
        'GN_concurrents', 
        'GN_concurrents_compatibles',
        'Similarite_reprise_antecedent'
    ]
    
    features_categorielles = [
        'TypeReprise', 
        'Type_pronom', 
        'Definitude_GN',
        'Fonction_reprise', 
        'Fonction_antecedent',
        'Antecedent_annote'
    ]

    # --- NETTOYAGE DES DONNÉES ---
    # Conversion forcée en numérique pour éviter l'erreur "Input contains NaN"
    for col in features_numeriques:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remplissage des cases vides (0 pour les chiffres, 'ABSENT' pour le texte)
    df[features_numeriques] = df[features_numeriques].fillna(0)
    df[features_categorielles] = df[features_categorielles].fillna('ABSENT')

    # 4. Construction du Pipeline de traitement
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), features_numeriques),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_categorielles)
        ])

    # 5. Définition du modèle (Gradient Boosting)
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3, 
            random_state=42
        ))
    ])

    # 6. Split Train/Test (80/20)
    X = df[features_numeriques + features_categorielles]
    y = df['Target_Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Entraînement
    print(f"Lancement de l'entraînement sur {len(X_train)} exemples...")
    model_pipeline.fit(X_train, y_train)

    # 8. Évaluation
    y_pred = model_pipeline.predict(X_test)
    
    print("\n" + "="*40)
    print("RÉSULTATS DU CLASSIFIEUR (Gradient Boosting)")
    print("="*40)
    print(classification_report(y_test, y_pred))
    
    print("\nMatrice de Confusion :")
    print(confusion_matrix(y_test, y_pred))
    
    return model_pipeline, y_test, y_pred

def analyser_importance_features(pipeline, features_numeriques, features_categorielles):
    # 1. Extraire le modèle et le préprocesseur du pipeline
    modele = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']

    # 2. Récupérer les noms des colonnes après transformation
    # Pour les numériques, les noms ne changent pas
    noms_num = features_numeriques
    
    # Pour les catégorielles, on récupère les noms générés par le OneHotEncoder
    noms_cat = preprocessor.named_transformers_['cat'].get_feature_names_out(features_categorielles)
    
    # Fusionner tous les noms
    tous_les_noms = list(noms_num) + list(noms_cat)

    # 3. Créer un DataFrame pour visualiser les résultats
    importances = modele.feature_importances_
    df_importance = pd.DataFrame({
        'Feature': tous_les_noms,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 4. Affichage
    print("\n" + "="*40)
    print("IMPORTANCE DES FEATURES (Top 15)")
    print("="*40)
    print(df_importance.head(15).to_string(index=False))
    
    return df_importance



# --- EXÉCUTION ---
chemin = '/Users/dariatupikina/dataset_erreurs_reprises.xlsx'

try:
    # On récupère les 3 objets renvoyés par la fonction
    mon_modele, y_test_final, y_pred_final = entrainer_classifieur(chemin)
    
    # Listes pour l'importance
    features_num = ['Distance_caracteres', 'Distance_mots', 'Distance_phrases', 
                    'GN_concurrents', 'GN_concurrents_compatibles', 'Similarite_reprise_antecedent']
    features_cat = ['TypeReprise', 'Type_pronom', 'Definitude_GN', 
                    'Fonction_reprise', 'Fonction_antecedent', 'Antecedent_annote']
    
    # 1. Analyse de l'importance
    analyser_importance_features(mon_modele, features_num, features_cat)
    
    # 2. Export HTML (on utilise les variables récupérées)
    nom_fichier = "analyse_erreurs_gradient_boost.html"
    
    # IMPORTANT : on utilise y_test_final.index pour faire le lien avec l'Excel d'origine
    exporter_erreurs_html(y_test_final, y_pred_final, y_test_final.index, nom_fichier)
    
    print(f"\n Fichier diagnostic généré : {nom_fichier}")
    print("Processus terminé avec succès !")

except Exception as e:
    print(f"\n Une erreur est survenue : {e}")
    # Optionnel : pour voir exactement où ça plante
    import traceback
    traceback.print_exc()