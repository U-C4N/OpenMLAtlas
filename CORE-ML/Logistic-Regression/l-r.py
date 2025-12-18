import pandas as pd  # Import the pandas library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Import the function to split data into train and test sets
from sklearn.linear_model import LogisticRegression  # Import the LogisticRegression model for classification
from sklearn.preprocessing import StandardScaler  # Import StandardScaler to standardize features
from sklearn.metrics import confusion_matrix, classification_report  # Import metrics for model evaluation

df = pd.read_csv('data.csv')  # Read the dataset from a CSV file named 'data.csv' into a pandas DataFrame

X = df.drop(['Country', 'Gender'], axis=1)  # Drop the columns 'Country' and 'Gender' from the data to use as features (input variables)
y = df['Gender']  # Assign the 'Gender' column as the target variable

print(X.head(), "Features")  # Display the first 5 rows of the features with a label for clarification
print(y.head(), "Targets")  # Display the first 5 rows of the target variable with a label for clarification

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,           # Features and target variable to split
    test_size=0.33, # Specify that 33% of the dataset should be used as the test set
    random_state=0, # Set a random state for reproducibility
    stratify=y      # Ensure the class distribution is preserved in both the training and test sets
)

scaler = StandardScaler()  # Create a StandardScaler object to standardize features

X_train = scaler.fit_transform(X_train)  # Fit the scaler on the training data and transform the training features
X_test = scaler.transform(X_test)        # Use the already-fitted scaler to transform the test features

logr = LogisticRegression(random_state=0)  # Initialize the Logistic Regression model with a fixed random state for reproducibility
logr.fit(X_train, y_train.values.ravel())  # Train the logistic regression model on the standardized training data

y_pred = logr.predict(X_test)  # Use the trained model to make predictions on the test data

print("Actual target values:", y_test.values.ravel())  # Print the true labels for the test set
print("Predictions:", y_pred)                          # Print the predicted labels for the test set

cm = confusion_matrix(y_test, y_pred)  # Compute the confusion matrix to evaluate the classification results

print("Confusion Matrix:")  # Print a label for the confusion matrix output
print(cm)                   # Print the confusion matrix values