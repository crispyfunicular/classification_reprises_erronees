import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.preprocessing import LabelEncoder

with open('dataset_erreurs_reprises.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    dataset=[]
    for row in reader:
        dataset.append(row)
df= pd.DataFrame(dataset)
X= df.drop(columns=['NumTexte', 'TexteErreur','SousTypeErreur','TexteAnte','Contexte', 'TypeErreur'])
y=df['TypeErreur']
X['DistanceMot']= pd.to_numeric(X['DistanceMot'].replace('', '0'))

X
X= X.select_dtypes(exclude=['number']) \
                .apply(LabelEncoder().fit_transform) \
                .join(X.select_dtypes(include=['number']))

le= LabelEncoder()
le.fit(y)
print(list(le.classes_))
y =le.transform(y)

y

'''s_AnteAnnote= df['AnteAnnote']
def AnteAnnote_to_int(x:str)->int:
    if x == 'oui':
        return 1
    elif x == 'non':
        return 0
    else:
        return 2
S_intanteannot = s_AnteAnnote.apply(AnteAnnote_to_int)
X['AnteAnnote'] = S_intanteannot

s_TypeErreur= df['TypeErreur']
def TypeErreur_to_int(x:str)->int:
    if x == 'E antécédent':
        return 0
    elif x == 'E reprise':
        return 1
    elif x == 'E grammaticale':
        return 2
    elif x =='':
        return 3
S_intTypeErreur= s_TypeErreur.apply(TypeErreur_to_int)'''

'''s_TypeReprise= df['TypeReprise']
def TypeReprise_to_int(x:str)->int:
    if x== 'R totale pronominale':
        return 0
    elif x== 'R résomptive':
        return 1
    elif x== 'R possessive':
        return 2
    elif x=='R groupe':
        return 3
    elif x== 'R partielle':
        return 4
    elif x== 'R totale fidèle identique':
        return 5
    elif x=='R autre':
        return 6
    elif x=='R totale infidèle':
        return 7
    elif x=='':
        return 8
    elif x=='R totale fidèle non identique':
        return 9
    else :
        raise Exception(x)


S_intTypeReprise = s_TypeReprise.apply(TypeReprise_to_int)
X['TypeReprise']=S_intTypeReprise
'''

#y #'SousTypeErreur',
#,('Contexte'),'FonctionRep','FonctionAnte

#X['FonctionRep']= pd.DataFrame(X['FonctionRep'])
#FonctionRep_encoded = OneHotEncoder(sparse_output=False).fit_transform( X[['FonctionRep']])
#X['FonctionRep']=FonctionRep_encoded


#X['FonctionAnte']= pd.DataFrame(X['FonctionAnte'])
#FonctionRep_encoded = OneHotEncoder(sparse_output=False).fit_transform( X[['FonctionAnte']])
#X['FonctionAnte']=FonctionRep_encoded



X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=11, test_size=0.3)



clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)



y_pred = clf.predict(X_test)
print(y_pred)



clf.score(X_test,y_test)



y_pred=le.inverse_transform(y_pred)
print(y_pred)


y_test=le.inverse_transform(y_test)


comparaison = pd.DataFrame({'Etiquette initiale': y_test,'Prédiction': y_pred})
print(comparaison)icale


f1_score(y_test, y_pred, average='weighted')
