# Random Forest Classifier - The Power of Many Trees!
# ---------------------------------------------------
# Imagine you need advice. Would you ask 1 person or 10 people?
# Random Forest = Asking 10 (or 100!) decision trees and taking a vote!

# 1. Importing Libraries (Our Toolbox)
# ------------------------------------

# Pandas: Our data manipulation friend. Think of it as Excel on steroids.
# 'pd' is the universal nickname everyone uses.
import pandas as pd

# Matplotlib: The artist that draws our graphs and charts.
# 'plt' is the short name we use for plotting.
import matplotlib.pyplot as plt

# RandomForestClassifier: A team of decision trees working together.
# Each tree votes, and the majority wins! Democracy in machine learning.
from sklearn.ensemble import RandomForestClassifier

# LabelEncoder: The translator that converts text to numbers.
# Machines don't understand "Bachelor's" or "Master's", but they love 0, 1, 2!
from sklearn.preprocessing import LabelEncoder


# 2. Loading the Data (Opening the File)
# --------------------------------------

# We read our CSV file and store it in a DataFrame called 'df'.
# This file contains information about job candidates.
df = pd.read_csv('hiring_data.csv')

# Let's see what our data looks like (optional but helpful)
# print(df.head())


# 3. Data Preprocessing (Preparing the Ingredients)
# -------------------------------------------------

# --- HIRING A TRANSLATOR ---
# The 'Degree_Level' column has text values like "Bachelor's", "Master's", "PhD".
# We need to convert these to numbers so our model can understand.

le = LabelEncoder()

# fit_transform does two things:
# 1. fit: Learn all unique values in the column (Bachelor's=0, Master's=1, PhD=2)
# 2. transform: Replace the text with the learned numbers
df['Degree_Level'] = le.fit_transform(df['Degree_Level'])


# 4. Separating Features and Target (X and y)
# -------------------------------------------

# X = FEATURES (The clues we use to make predictions)
# These are the columns that help us predict if someone gets hired.
# We DROP the 'Hired' column because that's what we're trying to predict!
# .values converts the DataFrame to a numpy array (the format sklearn loves)
X = df.drop('Hired', axis=1).values

# y = TARGET (What we're trying to predict)
# Will this person be hired? Yes (1) or No (0)?
y = df['Hired'].values


# 5. Creating the Forest (Building Our Team of Trees)
# ---------------------------------------------------

# --- ASSEMBLING THE COUNCIL OF TREES ---
# n_estimators = How many trees in our forest?
# More trees = Usually better predictions, but slower
# 10 trees is good for learning. In real projects, try 100-500!

clf = RandomForestClassifier(n_estimators=10)

# Fun fact: 'clf' stands for 'classifier' - a common naming convention!


# 6. Training the Model (Teaching the Forest)
# -------------------------------------------

# --- TRAINING TIME ---
# We show all our data (X) and answers (y) to every tree in the forest.
# Each tree learns slightly different patterns (that's the magic!).
# When combined, they're smarter than any single tree.

clf.fit(X, y)

# Behind the scenes:
# - Tree 1 might focus on "years of experience"
# - Tree 2 might focus on "degree level"
# - Tree 3 might find a pattern with "interview score"
# Together, they catch patterns that one tree might miss!


# 7. Making Predictions (Asking the Forest)
# -----------------------------------------

# --- JUDGMENT DAY ---
# We have a new candidate with these features:
# [10, 100, 9, 1] could mean:
# - 10 years experience
# - 100 projects completed
# - 9/10 interview score
# - 1 = Master's degree (after encoding)

# We ask all 10 trees: "Should we hire this person?"
# Each tree votes, and the majority wins!

prediction = clf.predict([[10, 100, 9, 1]])

# Note: We use [[...]] (double brackets) because sklearn expects a 2D array.
# Even for one person, we need to wrap it in a list.


# 8. Showing the Result
# --------------------

print("Prediction:", prediction)
# Output: [1] means "Hire!" / [0] means "Don't hire"


# ============================================
# WHY RANDOM FOREST IS AWESOME
# ============================================
#
# 1. WISDOM OF THE CROWD
#    - One tree can make mistakes
#    - 10 trees voting together? Much more reliable!
#
# 2. HANDLES MESSY DATA
#    - Missing values? No problem!
#    - Outliers? Trees can handle them!
#
# 3. TELLS YOU WHAT'S IMPORTANT
#    - "Experience matters most!" (feature importance)
#    - We can see which features the forest relies on
#
# 4. HARD TO OVERFIT
#    - Each tree sees different data (bootstrap sampling)
#    - Each tree uses different features (random subsets)
#    - The randomness makes the forest robust!
#
# ============================================
# REAL WORLD EXAMPLES
# ============================================
#
# - Netflix: "Will this user like this movie?"
# - Banks: "Will this person repay the loan?"
# - Hospitals: "Does this patient have diabetes?"
# - HR: "Should we hire this candidate?" (like our example!)
#
