# Multiple Linear Regression Example
# ----------------------------------
# In this script, we try to predict a student's grade (Grade) using multiple features
# such as gender, age, city, etc.

# Required libraries
import pandas as pd                      # Reading and manipulating tabular data
import matplotlib.pyplot as plt          # Visualizing results
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# 1. Read the dataset
# -------------------
# Read data.csv from the current folder.
data = pd.read_csv('data.csv')

# 2. Convert categorical columns to numeric
# ----------------------------------------
# Machine learning models work with numbers, not raw text.
# Convert 'Gender' (e.g., "Male" / "Female") into numeric labels.
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# One-hot encode the 'City' column.
# get_dummies creates a separate 0/1 column for each city.
# drop_first=True avoids an unnecessary extra column (dummy variable trap).
data = pd.get_dummies(data, columns=['City'], drop_first=True)

# 3. Split into features (X) and target (y)
# -----------------------------------------
# Grade is the value we want to predict.
X = data.drop('Grade', axis=1)
y = data['Grade']

# 4. Train/test split
# -------------------
# test_size=0.2: 20% test, 80% training.
# random_state=42: makes the split reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build and train the model
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predict on the test set
# --------------------------
y_pred = model.predict(X_test)

# 7. Compare actual vs predicted values
# -------------------------------------
results = pd.DataFrame({
    'Actual_Grade': y_test.values,
    'Predicted_Grade': y_pred,
})

print(results)

# 8. Evaluate performance (R^2 score)
# -----------------------------------
# R^2 closer to 1 means better fit.
r2 = r2_score(y_test, y_pred)
print("R^2 score:", r2)

# 9. Visualize: actual vs predicted scatter plot
# ----------------------------------------------
# If predictions are good, points should be close to a straight diagonal line.
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Actual Grade')
plt.ylabel('Predicted Grade')
plt.title('Actual vs Predicted Grade')
plt.show()
