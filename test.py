import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset (replace with your ECG data when available)
iris = load_iris()
print(type(iris))
print(iris)
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target variable (iris species)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
svm_model = SVC(kernel="rbf")  # Experiment with different kernels (linear, polynomial)
svm_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = svm_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Example prediction (assuming you have a new data point with the same features)
new_data = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your new data point
new_prediction = svm_model.predict(new_data)
print("Predicted iris species:", new_prediction[0])
