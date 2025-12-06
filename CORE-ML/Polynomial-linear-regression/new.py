import pandas as pd  # We import the data processing library with the alias 'pd'.
import matplotlib.pyplot as plt  # Library for plotting graphs.
from sklearn.linear_model import LinearRegression  # Linear modeling tool.
from sklearn.preprocessing import PolynomialFeatures  # Polynomial (curved) transformation tool.

# NOTE: Here, you read the dataset and assign it to the variable 'pd'.
pd = pd.read_csv('data.csv')

# x: Independent variable (Position Level).
# With .values, we convert this to a numpy array.
# We use double brackets [[]] so that it becomes a 2-dimensional matrix (machine learning models require this).
x = pd[['Position_Level']].values

# y: Dependent variable (Salary). The target we want to predict.
y = pd[['Salary']].values

# We create polynomial features.
# degree=4: Prepares the data to be transformed into a fourth degree equation (x^4).
# This allows us to draw a curved line instead of a straight line.
poly = PolynomialFeatures(degree=4)

# We take the x values (1, 2, 3...) and transform them into (x^0, x^1, x^2, x^3, x^4).
# It is necessary to transform the data like this so that the model can learn "curves".
x_poly = poly.fit_transform(x)

# We call the regression model.
lr = LinearRegression()

# We fit (train) the model.
# NOTE: We train the model not with the original 'x', but with the transformed 'x_poly'.
lr.fit(x_poly, y)

# We plot the actual data (x and y) as red scatter points.
plt.scatter(x, y, color='red')

# We plot the model's predicted curve.
# We provide the x values and plot the y values that the model (through x_poly) predicts, in blue.
plt.plot(x, lr.predict(x_poly), color='blue')

# We display the graph.
plt.show()

# Predicting for the level 6.5:
# The machine doesn't understand the raw value '6.5', since we trained the model with polynomials.
# First, we transform 6.5 to the format that the model understands using 'poly.fit_transform'.
predict = lr.predict(poly.fit_transform([[6.5]]))

# We print the result.
print(predict)