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
