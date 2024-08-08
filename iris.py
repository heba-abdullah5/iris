from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import joblib

# Load the Iris dataset from CSV file
data = pd.read_csv('iris.csv')

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Compute Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model to a file
joblib.dump(logistic_model, 'logistic_regression_model.pkl')

# Load the model from the file (optional)
loaded_model = joblib.load('logistic_regression_model.pkl')

# Verify the loaded model by making predictions again
loaded_y_pred = loaded_model.predict(X_test)

# Compute Accuracy for the loaded model
loaded_accuracy = accuracy_score(y_test, loaded_y_pred)

print(f"Loaded Model Accuracy: {loaded_accuracy * 100:.2f}%")
