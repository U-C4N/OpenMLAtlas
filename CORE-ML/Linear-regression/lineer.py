# 1. Importing the required libraries
# -----------------------------------

# NumPy: A core library for numerical computing (arrays, vectors, matrices).
# 'as np': We'll use the shorter alias 'np'.
import numpy as np

# Matplotlib.pyplot: Used for plotting graphs and visualizing data.
# 'plt': Short alias we will use when drawing plots.
import matplotlib.pyplot as plt

# Pandas: Lets us read data files and work with tabular data (like an Excel sheet).
# 'pd': Standard alias for pandas.
import pandas as pd

# Scikit-learn (sklearn): A popular library that contains many machine learning algorithms.
# LinearRegression: The simple linear regression model we will use.
from sklearn.linear_model import LinearRegression

# r2_score: A metric to evaluate how good the predictions are.
# (It typically ranges from 0 to 1; closer to 1 means a better fit.)
from sklearn.metrics import r2_score

# train_test_split: Splits the dataset into "train" and "test" parts.
# This lets us evaluate the model on data it has never seen before.
from sklearn.model_selection import train_test_split


# 2. Loading and preparing the data
# ---------------------------------

# pd.read_csv: Reads the 'salary.csv' file from the current folder into a DataFrame (table).
# Each row represents an employee. The columns store the input feature(s) and the target.
df = pd.read_csv('salary.csv')

# Print the first 5 rows to quickly verify the data loaded correctly.
print(df.head())

# X: The input feature(s) (independent variable).
# We use double brackets [[...]] to keep X as a 2D table (DataFrame), which sklearn expects.
X = df[["YearsExperience"]]

# y: The target we want to predict (dependent variable).
y = df["Salary"]



# 3. Splitting into training and test sets
# ---------------------------------------

# train_test_split:
# - test_size=0.3: Use 30% of the data for testing and 70% for training.
# - random_state=10: Ensures the split is reproducible across runs.
# X_train, y_train: Data the model learns from.
# X_test, y_test: Unseen data used only to evaluate performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# 4. Building and training the model
# ----------------------------------

# LinearRegression(): Creates a linear regression model.
model = LinearRegression()

# model.fit: The training step.
# The model learns the relationship between X_train (years) and y_train (salary).
model.fit(X_train, y_train)


# 5. Making predictions
# ---------------------

# Use the trained model to predict salaries for the X_test samples.
y_pred = model.predict(X_test)


# 6. Evaluating the model
# -----------------------

# r2_score compares the true values (y_test) with the predictions (y_pred).
# Values closer to 1 indicate a better fit.
r2 = r2_score(y_test, y_pred)
print(f"R2 score: {r2}")

# Print the predicted salaries to inspect the results.
print(y_pred)