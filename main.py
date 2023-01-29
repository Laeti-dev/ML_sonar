import pandas as pd


# observations = pd.read_csv('./datas/sonar.all-data.csv')
# Decouvrir le DF
# print(observations.head())
# print(observations.columns.values)
# print(observations.shape)
# add columns name
names = [f'F{x}' for x in range(1,61)]
names.append('OBJET')
print(names)

observations = pd.read_csv('./datas/sonar.all-data.csv', names=names)
print(observations.head())
# pour enlever la limite d'affichage de colonnes de pandas
pd.set_option('display.max_columns',None)
# print(observations.head(5))

# changer la colonne OBJET en valeurs rocher = 0, mine = 1
observations['OBJET'] = (observations['OBJET'] == 'M').astype(int)

# Manque t'il des infos?
print(observations.info())

# column objet
print(observations.groupby(by='OBJET').size())

print(observations.describe())

# importer le fichier JMPStatistiques pour utiliser les fonctions stats
import JMPStatistiques as jmp
stats = jmp.JMPStatistiques(observations['F1'])
# stats.analyseFeature()

# Utilisation de matplotlib pour visualiser toutes nos mesures
from matplotlib import pyplot as plt
# observations.plot.box(figsize=(10,10), xticks=[]) #pour ne pas afficher les x

# plt.title('Détection des valeurs extrêmes')
# plt.xlabel('Les 60 fréquences')
# plt.ylabel('Puissance du signal')
# plt.show()

# Nous remarquons que de nombreuses mesures présentent des valeurs abérrentes
# Pouvant entrainer de mauvaises prédictions il faut les traiter sans nécessairement les retirer

# D'abord, choisissons le modèle de prédiction
# 1 -scinder nos observations en données d'apprentissage et données de test avec la fonction train_test_split
# de skl
from sklearn.model_selection import train_test_split
array = observations.values

# S'assurer que toutes les données sont sous forme décimale
X = array[:,0:-1].astype(float)

# Définir la dernière colonne comme la feature de prédiction
Y = array[:,-1]

# Créer les jeux d'apprentissage (80%) et de test (20%)
percentage_donnees_test = 0.2
X_APPRENTISSAGE, X_VALIDATION, \
    Y_APPRENTISSAGE, Y_VALIDAITON = train_test_split(X, Y, test_size=percentage_donnees_test,
                                                                                random_state=42)

# Tester les algorithmes avec la fonction fit (qui réalise l'apprentissage)
# S'agissant d'algorithmes de classification, il faut utiliser la fonction acuracy_score pour
# calculer la précision d'apprentissage (VS r2_score pour les algorithmes de régression)
from sklearn.metrics import accuracy_score

# 1- REGRESSION LOGISTIQUE
from sklearn.linear_model import LogisticRegression
regression_logistique = LogisticRegression()
regression_logistique.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = regression_logistique.predict(X_VALIDATION)
print(f'logistic regression:{str(accuracy_score(predictions, Y_VALIDAITON))}')

#  2- ARBRE DE DECISION
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = decision_tree.predict(X_VALIDATION)
print(f'Decision tree:{str(accuracy_score(predictions, Y_VALIDAITON))}')

# 3- RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = random_forest.predict(X_VALIDATION)
print(f'Random Forest:{str(accuracy_score(predictions, Y_VALIDAITON))}')

# 4- K-NEAREST NEIGHBOR
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_APPRENTISSAGE,Y_APPRENTISSAGE)
predictions = knn.predict(X_VALIDATION)
print(f'KNN:{str(accuracy_score(predictions, Y_VALIDAITON))}')

# 5- Support Vector Machine
from sklearn.svm import SVC
svm = SVC(gamma='auto')
svm.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = svm.predict(X_VALIDATION)
print(f'SVM:{str(accuracy_score(predictions, Y_VALIDAITON))}')
# Trouver l'hyperparametre C
from sklearn.model_selection import GridSearchCV
# Défini une plage de valeurs à tester