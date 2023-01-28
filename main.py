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
stats.analyseFeature()

# Utilisation de matplotlib pour visualiser toutes nos mesures
from matplotlib import pyplot as plt
observations.plot.box(figsize=(10,10), xticks=[]) #pour ne pas afficher les x

plt.title('Détection des valeurs extrêmes')
plt.xlabel('Les 60 fréquences')
plt.ylabel('Puissance du signal')
plt.show()

# Nous remarquons que de nombreuses mesures présentent des valeurs abérrentes
# Pouvant entrainer de mauvaises prédictions il faut les traiter sans nécessairement les retirer

# D'abord, choisissons le modèle
