from pipeline import construire_matrice , evaluer_baselines
from amelioration    import pretraiter, optimiser_modeles
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
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
print("[1/4]  Dataset BASE  +  Modèles BASELINE")
X_base, y_base = construire_matrice("dataset_erreurs_reprises.csv", features_base)
X_base_train_prep, X_base_test_prep, y_base_train, y_base_test, preprocesseur =pretraiter(X_base, y_base, features_base_num, features_base_cat)
score_base = evaluer_baselines(X_base_train_prep, X_base_test_prep, y_base_train, y_base_test)


##2 score texte enrichi par des features
print("[2/4]  Dataset Enrichi  +  Modèles BASELINE")
dff_enrichi= enrichir_dataset("dataset_erreurs_reprises.csv","dataset_enrichi.csv")
X_enrichi, y_enrichi= construire_matrice("dataset_enrichi.csv",features)
X_enr_train_prep, X_enr_test_prep, y_enr_train, y_enr_test, preprocesseur = pretraiter(X_enrichi, y_enrichi,features_numeriques, features_categorielles)
score_enrichi = evaluer_baselines(X_enr_train_prep, X_enr_test_prep, y_enr_train, y_enr_test)



##3 base + tuning
print("[3/4]  Dataset BASE  +  Tuning (GridSearch)")
base_lr, base_rf = optimiser_modeles(X_base_train_prep, y_base_train,'f1_macro' )
y_base_pred_lr = base_lr.predict(X_base_test_prep)
#f1 macro
score_tun_lr_f1_macro= f1_score(y_base_test,y_base_pred_lr, average='macro')
#f1 weighted
score_tun_lr_f1_weighted= f1_score(y_base_test,y_base_pred_lr,average='weighted')
# balanced_accuracy_
score_tune_lr_acc= balanced_accuracy_score(y_base_test,y_base_pred_lr)


y_base_pred_rf = base_rf.predict(X_base_test_prep)
#f1 macro
score_tun_rf_f1_macro= f1_score(y_base_test, y_base_pred_rf,average='macro')
#f1 weighted
score_tun_rf_f1_weighted= f1_score(y_base_test, y_base_pred_rf,average='weighted')
# balanced_accuracy_
score_tune_rf_acc= balanced_accuracy_score(y_base_test, y_base_pred_rf)


##4 enrichi + tunning
print("  [4/4]  Dataset Enrichi  +  Tuning (GridSearch)")
mix_lr, mix_rf = optimiser_modeles(X_enr_train_prep, y_enr_train, 'f1_macro')
y_enr_pred_lr = mix_lr.predict(X_enr_test_prep)

#f1 macro
score_mixe_lr_f1_macro= f1_score(y_enr_test,y_enr_pred_lr, average='macro')
#f1 weighted
score_mixe_lr_f1_weighted= f1_score(y_enr_test,y_enr_pred_lr, average='weighted')
# balanced_accuracy_
score_mixe_lr_lr_acc= balanced_accuracy_score(y_enr_test,y_enr_pred_lr)

y_enr_pred_rf = mix_rf.predict(X_enr_test_prep)
#f1 macro
score_mixe_rf_f1_macro= f1_score(y_enr_test, y_enr_pred_rf, average='macro')
#f1 weighted
score_mixe_rf_f1_weighted= f1_score(y_enr_test, y_enr_pred_rf, average='weighted')
# balanced_accuracy_
score_mixe_rf_lr_acc= balanced_accuracy_score(y_enr_test, y_enr_pred_rf)



##Creation du dataframe lignes = differentes types de combinaisons // colonnes= scores
#Base
df_base    = pd.DataFrame([
    {
        "Modele"            : "LogisticRegression",
        "F1_macro"          : score_base['Régression Logistique'][0],
        "F1_weighted"       : score_base['Régression Logistique'][1],
        "Balanced_accuracy" : score_base['Régression Logistique'][2],
        "Dataset"           : "Base",
        "Methode"           :"Baseline",
        },
        {
        "Modele"            : "RandomForest",
        "F1_macro"          : score_base['Random Forest'][0],
        "F1_weighted"       : score_base['Random Forest'][1],
        "Balanced_accuracy" : score_base['Random Forest'][2],
        "Dataset"           : "Base",
        "Methode"           :"Baseline",
            }
                           ])
df_base["Dataset"]  = "Base"
df_base["Methode"]  = "Baseline"
#Enrichi
df_enrichi = pd.DataFrame([
    {
        "Modele"            : "LogisticRegression",
        "F1_macro"          : score_enrichi['Régression Logistique'][0],
        "F1_weighted"       : score_enrichi['Régression Logistique'][1],
        "Balanced_accuracy" : score_enrichi['Régression Logistique'][2],
        "Dataset"           : "Enrichi",
        "Methode"           : "Baseline",
        },
        {
        "Modele"            : "RandomForest",
        "F1_macro"          : score_enrichi['Random Forest'][0],
        "F1_weighted"       : score_enrichi['Random Forest'][1],
        "Balanced_accuracy" : score_enrichi['Random Forest'][2],
        "Dataset"           : "Base",
        "Methode"           :"Baseline",
            }
                            ])
df_enrichi["Dataset"] = "Enrichi"
df_enrichi["Methode"] = "Baseline"

#Tuning – Dataset Base
df_tuning_base = pd.DataFrame([
    {
        "Modele"            : "LogisticRegression",
        "F1_macro"          : score_tun_lr_f1_macro,
        "F1_weighted"       : score_tun_lr_f1_weighted,
        "Balanced_accuracy" : score_tune_lr_acc,
        "Dataset"           : "Base",
        "Methode"           : "GridSearch",
    },
    {
        "Modele"            : "RandomForest",
        "F1_macro"          : score_tun_rf_f1_macro,
        "F1_weighted"       : score_tun_rf_f1_weighted,
        "Balanced_accuracy" : score_tune_rf_acc,
        "Dataset"           : "Base",
        "Methode"           : "GridSearch",
    },
])

#Tuning – Dataset Enrichi
df_tuning_enrichi = pd.DataFrame([
    {
        "Modele"            : "LogisticRegression",
        "F1_macro"          : score_mixe_lr_f1_macro,
        "F1_weighted"       : score_mixe_lr_f1_weighted,
        "Balanced_accuracy" : score_mixe_lr_lr_acc,
        "Dataset"           : "Enrichi",
        "Methode"           : "GridSearch",
    },
    {
        "Modele"            : "RandomForest",
        "F1_macro"          : score_mixe_rf_f1_macro,
        "F1_weighted"       : score_mixe_rf_f1_weighted,
        "Balanced_accuracy" : score_mixe_rf_lr_acc,
        "Dataset"           : "Enrichi",
        "Methode"           : "GridSearch",
    },
])

df_resultats = (
    pd.concat([df_base, df_enrichi, df_tuning_base, df_tuning_enrichi],
              ignore_index=True)
    .round(4)
    [["Dataset", "Methode", "Modele", "F1_macro", "F1_weighted", "Balanced_accuracy"]]
    .sort_values(["Dataset", "Methode", "Modele"])
    .reset_index(drop=True)
)

print(df_resultats.to_string(index=False))
#enregistrement des scores dans un CSV
df_resultats.to_csv('resultats_scoring.csv')
