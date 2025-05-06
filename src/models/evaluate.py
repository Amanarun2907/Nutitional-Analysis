# src/models/evaluate.py

import pandas as pd
import joblib
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessing import preprocess_data  # Import for consistent preprocessing

# Load the trained XGBoost model
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'xgboost_model.joblib')
try:
    model = joblib.load(model_path)
    print(f"Trained XGBoost model loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"Error: Trained XGBoost model not found.")
    sys.exit(1)

# Load the fitted preprocessor
preprocessor_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'preprocessor.joblib')
try:
    preprocessor = joblib.load(preprocessor_path)
    print(f"Preprocessor loaded successfully from: {preprocessor_path}")
except FileNotFoundError:
    print(f"Error: Preprocessor not found. Make sure you saved it during training.")
    sys.exit(1)

# Load the original cleaned data
file_path = os.path.join(project_root, 'data', 'processed', 'cleaned_data.csv')
try:
    cleaned_df = pd.read_csv(file_path)
    print(f"Successfully loaded data from: {file_path}\n")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please ensure the path is correct.")
    sys.exit(1)

# Separate features (X) and target (y)
X = cleaned_df.drop('nutrition_score', axis=1)
y = cleaned_df['nutrition_score']

# Preprocess the data using the loaded preprocessor
try:
    X_processed = preprocessor.transform(X)
except Exception as e:
    print(f"Error during preprocessing: {e}")
    sys.exit(1)

# Split data into training and testing sets (same split as before)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Make predictions on the test set
try:
    y_pred = model.predict(X_test)
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

# Evaluate the model on the test set
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation on Test Set:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Display a few examples of actual vs. predicted values
print("\nActual vs. Predicted (First 10 Samples from Test Set):")
results_df = pd.DataFrame({'Actual': y_test.head(10), 'Predicted': y_pred[:10]})
print(results_df)