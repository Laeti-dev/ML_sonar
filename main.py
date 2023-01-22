import pandas as pd


observations = pd.read_csv('./datas/sonar.all-data.csv')

print(observations.head())
print(observations.columns.values)
print(observations.shape)
