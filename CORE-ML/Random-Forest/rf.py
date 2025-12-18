import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('hiring_data.csv')

le = LabelEncoder()

df['Degree_Level'] = le.fit_transform(df['Degree_Level'])

X = df.drop('Hired', axis=1).values
y = df['Hired'].values

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

prediction = clf.predict([[10, 100, 9, 1]])

print("Prediction: ", prediction)