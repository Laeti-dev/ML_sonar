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