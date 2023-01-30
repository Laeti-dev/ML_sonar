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
# Définir une plage de valeurs à tester allant de 1 à 100 (1 étant la valeur par défaut de l'hyperparamètre
# C utilisé par l'algorithme SVC
penalite = [{'C': range(1,100)}]

# Tests avec 5 échantillons de Validations Croisées
# L'algorithme GridSearchCV prend les paramètres suivantd :
# - l'algorithme à tester
# - les hyperparamètres à optimiser
# - le nombre de validations croisées = définir en combien de groupe séparer nos données,
# déterminer une optimisation sur les premiers groupes et les tester sur le dernier
recherche_optimisation = GridSearchCV(SVC(), penalite, cv=5)
recherche_optimisation.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
print(f'Le meilleur paramètre est {recherche_optimisation.best_params_}')
# Optimisation du SVM
svm = SVC(C=35, gamma='auto')
svm.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = svm.predict(X_VALIDATION)
print(f'SVM:{str(accuracy_score(predictions, Y_VALIDAITON))}')

# 6- GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingClassifier
gradientBoosting = GradientBoostingClassifier()
gradientBoosting.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = gradientBoosting.predict(X_VALIDATION)
print(f'Gradient Boosting:{str(accuracy_score(predictions, Y_VALIDAITON))}')

# Gestion des valeurs extrêmes
# en supprimant les ligne > ou < 1,5 fois la valeur de leur interquartile
# 1- Pour chaque caractéristique, on cherche les numéros de ligne correspondant à une donnée exttrême
import numpy as np
# Créer une liste chargée de contenir ces numéros de lignes
num_lignes = []
# Itérer sur les 60 features
for feature in observations.columns.tolist():
    # Calcul des percentiles
    Q1 = np.percentile(observations[feature], 25)
    Q3 = np.percentile(observations[feature], 75)
    # Définition de la borne
    extremes = 1.5*(Q3-Q1)
    # Créer une liste des index de lignes de données extrêmes
    liste_donnees_extremes = observations[(observations[feature]<Q1-extremes) |
                                          (observations[feature]>Q3+extremes)].index
    # Les ajouter à notre liste
    num_lignes.extend(liste_donnees_extremes)

    # Trie cette liste par ordre croissant
    num_lignes.sort()

    # Créer une liste contenant les numéros de lignes à supprimer
    to_delete = []

    # Itérer sur la liste de numéros de lignes
    for ligne in num_lignes:
        # Récupérer son numéro
        numero = ligne
        # Compter le nombre de fois que ce numéro de ligne apparait dans l'ensemble des numéros de ligne
        nb_val_extremes = num_lignes.count(numero)
        # Si ce compte est > 7 alors on ajoute ce numéro à la liste à supprimer
        if (nb_val_extremes > 7):
            to_delete.append(numero)
    # Supprimer les doublons
    to_delete = list(set(to_delete))

# suprimer ces lignes du DF

print(to_delete)
print(f'À supprimer: {len(to_delete)}')
observations.drop(to_delete, axis=0)

# Suppression des erreurs de type warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 1- REGRESSION LOGISTIQUE
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
# Définir une plage de valeurs à tester allant de 1 à 100 (1 étant la valeur par défaut de l'hyperparamètre
# C utilisé par l'algorithme SVC
penalite = [{'C': range(1,100)}]

# Tests avec 5 échantillons de Validations Croisées
# L'algorithme GridSearchCV prend les paramètres suivantd :
# - l'algorithme à tester
# - les hyperparamètres à optimiser
# - le nombre de validations croisées = définir en combien de groupe séparer nos données,
# déterminer une optimisation sur les premiers groupes et les tester sur le dernier
recherche_optimisation = GridSearchCV(SVC(), penalite, cv=5)
recherche_optimisation.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
print(f'Le meilleur paramètre est {recherche_optimisation.best_params_}')
# Optimisation du SVM
svm = SVC(C=35, gamma='auto')
svm.fit(X_APPRENTISSAGE, Y_APPRENTISSAGE)
predictions = svm.predict(X_VALIDATION)
print(f'SVM:{str(accuracy_score(predictions, Y_VALIDAITON))}')