import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

data = pd.read_csv('data.csv')

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

data = pd.get_dummies(data, columns=['City'] , drop_first=True)

x = data.drop('Grade', axis=1)
y = data['Grade']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

sonuc = pd.DataFrame({
    'Actual_Grade': y_test.values,   # actual grade
    'Predicted_Grade': y_pred           # predicted grade
})

print(sonuc)

r2 = r2_score(y_test, y_pred)
print("R^2 score:", r2)

plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Actual vs Predicted Grade')
plt.show()