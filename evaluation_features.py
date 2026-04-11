"""
Comparaison des scores F1 avec et sans les nouvelles features de Sarah
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Charger le dataset enrichi (généré par sarah_features.py)
df = pd.read_excel('dataset_enrichi.xlsx')
df.columns = [c.strip().replace('\xa0', '') for c in df.columns]
df = df.dropna(subset=['TypeErreur'])

y = df['TypeErreur']

# --- VERSION 1 : features de base (comme Salma) ---
features_base = ['TypeReprise', 'AnteAnnote', 'DistanceMot', 'FonctionRep', 'FonctionAnte']
X_base = df[features_base].fillna('inconnu')
for col in X_base.columns:
    X_base[col] = LabelEncoder().fit_transform(X_base[col].astype(str))

# --- VERSION 2 : features de base + nouvelles features Sarah ---
features_sarah = ['Match_genre', 'Match_nombre', 'Est_pronom', 'Longueur_reprise', 'Longueur_antecedent']
features_enrichi = features_base + features_sarah
X_enrichi = df[features_enrichi].fillna('inconnu')
for col in features_base:
    X_enrichi[col] = LabelEncoder().fit_transform(X_enrichi[col].astype(str))

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split identique pour les deux
X_base_train, X_base_test, X_enr_train, X_enr_test, y_train, y_test = train_test_split(
    X_base, X_enrichi, y_encoded, random_state=11, test_size=0.3, stratify=y_encoded
)

# Modèle 1 : baseline
clf1 = LogisticRegression(max_iter=10000).fit(X_base_train, y_train)
y_pred1 = clf1.predict(X_base_test)
f1_base = f1_score(y_test, y_pred1, average='weighted')

# Modèle 2 : avec features Sarah
clf2 = LogisticRegression(max_iter=10000).fit(X_enr_train, y_train)
y_pred2 = clf2.predict(X_enr_test)
f1_enrichi = f1_score(y_test, y_pred2, average='weighted')

print("="*50)
print(f"F1 score SANS nouvelles features : {f1_base:.3f}")
print(f"F1 score AVEC nouvelles features : {f1_enrichi:.3f}")
print(f"Amélioration : {(f1_enrichi - f1_base)*100:.1f}%")
print("="*50)
print("\nDétail AVEC nouvelles features :")
print(classification_report(y_test, y_pred2, target_names=le.classes_))
