import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For compatibility with older scikit-learn versions

# Model filename (replace with your desired filename)
model_filename = "saved_ecg_trained_data_2.pkl"

try:
  # Try to load the trained model if it exists
  svm_model = joblib.load(model_filename)
  print("Loaded saved SVM model.")
except FileNotFoundError:
  # If model is not found, proceed with training

  # Load your ECG data from CSV
  data = pd.read_csv(r"C:\Users\gaura\Documents\Coding\ECG\ECG_Disease_Prediction\ptbdb_combined.csv")

  # Separate features (ECG signals) and target variable (disease class)
  X = data.drop("diagnosis", axis=1)  # Assuming "disease_class" is the target column
  y = data["diagnosis"]
  print(type(y))
  # Feature engineering (replace with your specific domain knowledge)
  # Example: Calculate basic statistical features from ECG signals
  from scipy import stats
  X_new = pd.DataFrame()
 
      # Add more features as needed based on your data analysis

  # Split data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Create and train the SVM model
  svm_model = SVC(kernel="rbf")  # Experiment with different kernels (linear, polynomial)
  svm_model.fit(X_train, y_train)

  # Save the trained model for future use
  joblib.dump(svm_model, model_filename)
  print("Saved trained SVM model.")
  print(svm_model.score(X_test,y_test))
# Make predictions on the testing set (or new data)
# ... (rest of your code for prediction and evaluation)
