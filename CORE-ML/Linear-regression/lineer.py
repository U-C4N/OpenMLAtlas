# 1. Importing Libraries (Toolkits)
# ----------------------------------

# NumPy: The fundamental library for numerical operations and mathematical calculations.
# 'as np': We will refer to it as 'np' for brevity instead of typing 'numpy' every time.
import numpy as np 

# Matplotlib.pyplot: Used for visualizing (plotting) data as graphs.
# 'plt': We use this short alias for matplotlib's plotting module. Our painter friend!
import matplotlib.pyplot as plt 

# Pandas: Used for data analysis and table manipulations (think of it like Excel).
# 'pd': The universal shortcut for the pandas library.
import pandas as pd 

# Scikit-learn (sklearn): The giant library that contains machine learning algorithms.
# LinearRegression: We are calling the model class that builds linear relationships.
from sklearn.linear_model import LinearRegression 

# metrics: Tools for measuring how successful our model is.
# mean_squared_error: Calculates the error (difference between real and predicted values).
# r2_score: The model's success score (the closer to 1, the better).
from sklearn.metrics import mean_squared_error, r2_score 

# train_test_split: Function for splitting data into "Train" and "Test" sets.
# (Explained in detail below why we do this).
from sklearn.model_selection import train_test_split


# 2. Loading and Preparing the Data
# ---------------------------------

# pd.read_csv: Reads the file named 'salary.csv' and turns it into a table.
# This file contains comma-separated values.
# NOTE: In your code, you had 'pd = pd.read...', which is incorrect. The variable should be named 'df' (dataframe).
df = pd.read_csv('salary.csv') 

# df.head(): Prints only the first 5 rows of the table.
# Purpose: To quickly check if data loaded correctly.
print(df.head()) 

# X (Uppercase): Independent variable (Features). The clue we'll use for prediction.
# [['YearsExperience']]: We use double brackets because Scikit-learn models
# always expect the input as a 2D table (DataFrame). A single bracket would give a list (Series).
X = df[['YearsExperience']] 

# y (Lowercase): Dependent variable (Target). The target we are trying to predict.
# Here, our target is 'Salary'. Single bracket is sufficient (Series).
y = df['Salary'] 


# 3. Splitting the Data (Training and Testing)
# --------------------------------------------

# train_test_split: Splits our data into 4 parts.
# test_size=0.3: Sets aside 30% of data for "Test", uses 70% for "Training".
# random_state=1: Always shuffles data in the same order (so results don't change each run).
# X_train: Training set (Years of Experience). The model will learn from these.
# X_test:  Test set (Years of Experience). The data we save to test our model.
# y_train: Training set answers (Salaries). Model will try to learn y_train from X_train.
# y_test:  Test set answers (Real Salaries). We'll compare the model's predictions to these.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


# 4. Building and Training the Model
# -----------------------------------

# Creating the model (like an empty brain, knows nothing yet).
model = LinearRegression() 

# model.fit: This is where the learning (training) happens!
# The model finds the mathematical relationship between X_train (years) and y_train (salaries).
# In the background, it creates an equation of a line (y = ax + b).
model.fit(X_train, y_train) 


# 5. Making Predictions
# ---------------------

# model.predict: We tell the model; "Here are the X_test data (years) I saved,
# use the formula you learned to predict their salaries."
# y_pred: The salary values predicted by the model.
y_pred = model.predict(X_test) 


# 6. Measuring Success
# --------------------

# r2_score: Compares the real salaries (y_test) to the model's predictions (y_pred).
# Results are between 0 and 1. 1 means perfect fit.
r2 = r2_score(y_test, y_pred) 
print(f"R2 Score: {r2}") 

print(y_pred)