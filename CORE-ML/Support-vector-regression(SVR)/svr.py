import pandas as pd # For data analysis and DataFrame manipulation.
import numpy as np # For numerical operations and array/matrix management.
import matplotlib.pyplot as plt # Standard library for data visualization (not used in this block).
from sklearn.preprocessing import StandardScaler # Used to scale features to the same range.
from sklearn.svm import SVR # Support Vector Regression class.

# 1. Data Loading
# Read the 'data.csv' file into a Pandas DataFrame.
data = pd.read_csv('data.csv')

# 2. Data Preprocessing
# Independent Variable (Feature): Position Level.
# Sklearn models generally expect 2D matrices as input, thus double brackets [[]].
X = data[["Position_Level"]].values

# Dependent Variable (Target): Salary.
# This is the target column we want to predict.
y = data[["Salary"]].values

# 3. Feature Scaling
# PROBLEM: The huge difference between Position (1-10) and Salary (50k+) can cause the model (SVR) to miscalculate the error.
# SOLUTION: Standardize the data. We transform the data so the mean is 0 and variance is 1 (usually squeezed between -3 and +3).

sc_X = StandardScaler() # Tool to scale input (Position) data.
sc_y = StandardScaler() # Tool to scale output (Salary) data.

# fit_transform logic:
# 1. fit: Calculate/learn the statistical structure (mean and std) of the data.
# 2. transform: Standardize the data using the learned values (Z-score transformation).
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y)

# 4. Model Creation and Training
# kernel='rbf': Radial Basis Function. When data is non-linear, this kernel draws the curve with the best fit.
regressor = SVR(kernel='rbf')

# Training the Model (Fitting):
# y_scaled.ravel(): Flattens the target variable from (n,1) shape matrix to (n,) flat vector (avoids model warning).
regressor.fit(X_scaled, y_scaled.ravel())

# 5. Prediction Process

# Step A: Prepare the Query (Scenario: What is the salary of an employee at level 6.5?)
# Since the model was trained with 2D matrices, input should be given as [[ ]].
example = [[6.5]]

# Step B: Transform the Query
# The model does not understand "Real Numbers" (6.5); it only recognizes the "Scaled Numbers" it was trained with.
# So, we scale 6.5 according to the rules learned by sc_X.
# NOTE: Never use 'fit' here (we're not learning something new), just apply the existing rule (transform).
example_scaled = sc_X.transform(example)

# Step C: Get the Prediction
# The model returns a scaled (normalized) salary value.
# e.g. It might return 0.8, but we don't know its real-life correspondence yet.
example_prediction = regressor.predict(example_scaled)

# Step D: Transform and Format the Answer (Inverse Transform)
# We convert the model's nonsensical scaled value back to real currency (TL/USD).
# reshape(-1, 1): Since sc_y learned training in 2D format (vertical column),
# we reshape the one-dimensional prediction back into vertical form before transforming.
example_prediction_scaled = sc_y.inverse_transform(example_prediction.reshape(-1, 1))

# Print the result as the real salary value.
print(example_prediction_scaled)