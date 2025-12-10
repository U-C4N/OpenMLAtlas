import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import LabelEncoder


# --- PUTTING THE DATA ON THE TABLE ---
# We read the csv file and convert it into a table called 'data'.
data = pd.read_csv('salary_data.csv')

# --- HIRING TRANSLATORS ---
# We assigned a translator for the Education column (e.g. Bachelor's, Master's) to translate them.
le_education = LabelEncoder()

# Assigned another translator for the Management Role column (e.g. Yes, No) to translate them.
le_management_role = LabelEncoder()

# --- ENCODING PROCESS ---
# We tell the translator: "Look at the Education column, learn the words, and turn them into numbers (0, 1, 2)."
data['Education'] = le_education.fit_transform(data['Education'])

# We do the same for the Management Role. Now the table contains numbers, not words. The machine loves that.
data['Management_Role'] = le_management_role.fit_transform(data['Management_Role'])


# --- SEPARATING THE INGREDIENTS ---
# X = CLUES. What will we use to predict the salary? (Experience, Education, Management)
# Double square brackets mean "Give me these as a table."
X = data[['Experience_Years', 'Education', 'Management_Role']]

# y = TARGET. What are we trying to find? (Salary/Money)
y = data['Salary_USD']


# --- CREATING THE ROBOT ---
# We created a model (robot) that knows nothing yet and has an empty memory.
model = DecisionTreeRegressor()

# --- TRAINING TIME (FIT) ---
# We tell the robot: "Take these clues (X), and here is the answer key (y). Find the relationship!"
# The robot learns here who earns how much.
model.fit(X, y)


# --- GRAPH PREPARATION ---
# If the data comes in a messy order, the line graph will look tangled.
# So, we sort the data by experience years in ascending order (x_p).
x_p = X.sort_values(by='Experience_Years')

# We ask the robot: "What do you think the salary of these sorted people should be?"
# We save the robot's predictions in 'y_p'. (This will be the red line.)
y_p = model.predict(x_p)


# --- DRAWING THE PICTURE ---
# BLUE DOTS: The real data. (Actual salaries in the table)
plt.scatter(X['Experience_Years'], y, color='blue')

# RED LINE: Our robot's prediction. (The path the model learned)
plt.plot(x_p['Experience_Years'], y_p, color='red')

# Let's name the bottom (X-axis) so it's clear what it is.
plt.xlabel('Experience Years')

# Let's name the side (Y-axis).
plt.ylabel('Salary USD')

# Put a cool title on top of the graph.
plt.title('Salary vs Experience Years')

# And action! We display the graph on the screen.
plt.show()