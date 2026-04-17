from pipeline import construire_matrice , evaluer_baselines

from amelioration import pretraiter,optimiser_modeles, accuracy_score, best_model
from feature_engineering import enrichir_dataset
import pandas as pd


'''On compare 4 combinaisons dataset × modèle et centralise tous les résultats
dans un DataFrame pandas.'''
features_base_num = [
                 "DistanceCarac",
                 "DistanceMot",

                 ]
features_base_cat = [
                "FonctionRep",
                "FonctionAnte",
                "AnteAnnote",
    ]
features_base = features_base_num + features_base_cat
# -- Définition des features --
features_numeriques = [
        "Longueur_reprise",
        "Longueur_antecedent",
        "Match_nombre",
        "DistanceCarac",
        "DistanceMot",
    ]

features_categorielles = [
        "Nombre_reprise",
        "Genre_reprise",
        "Nombre_antecedent",
        "Match_genre",
         "Genre_antecedent",
         "Est_pronom",
         "Type_pronom_detaille",
         "TypeReprise",
         "AnteAnnote",
         "FonctionRep",
         "FonctionAnte"
    ]
features = features_numeriques + features_categorielles

##1 score texte base
X_base, y_base = construire_matrice("dataset_erreurs_reprises.csv", features_base)
X_base_train_prep, X_base_test_prep, y_base_train, y_base_test, preprocesseur =pretraiter(X_base, y_base, features_base_num, features_base_cat)
score_base = evaluer_baselines(X_base_train_prep, X_base_test_prep, y_base_train, y_base_test)


##2 score texte enrichi par des features
dff_enrichi= enrichir_dataset("dataset_erreurs_reprises.csv","dataset_enrichi.csv")
X_enrichi, y_enrichi= construire_matrice("dataset_enrichi.csv",features)
X_enr_train_prep, X_enr_test_prep, y_enr_train, y_enr_test, preprocesseur = pretraiter(X_enrichi, y_enrichi,features_numeriques, features_categorielles)
score_enrichi = evaluer_baselines(X_enr_train_prep, X_enr_test_prep, y_enr_train, y_enr_test)

##3 base + tuning
best_opt_base1, best_opt_base2 = optimiser_modeles(X_base_train_prep, y_base_train,'f1_macro' )
y_base_pred = best_model.predict(X_base_test)
score_tuning= accuracy_score(y_base_test,y_base_pred)

##4 enrichi + tunning

best_opt_enr1, best_opt_enr2 = optimiser_modeles(X_enr_train_prep, y_enr_train, 'f1_macro')
y_enr_pred = best_model.predict(X_enr_test)
score_mix = accuracy_score(y_enr_test,y_enr_pred)


