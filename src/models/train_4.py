# src/models/train_4.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the project root to the Python path if it's not already there
if project_root not in sys.path:
    sys.path.append(project_root)

# Now you should be able to import from 'src'
from src.data.preprocessing import preprocess_data

# Define the path to your cleaned data file relative to the project root
file_path = os.path.join(project_root, 'data', 'processed', 'cleaned_data.csv')

try:
    # Load the cleaned data
    cleaned_df = pd.read_csv(file_path)
    print(f"Successfully loaded data from: {file_path}\n")

    # Preprocess the data
    X_processed, y, preprocessor = preprocess_data(cleaned_df.copy())

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Initialize and train the K-Nearest Neighbors Regressor model
    # You can experiment with the 'n_neighbors' parameter
    model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("K-Nearest Neighbors Regressor Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Define the path to save the trained model relative to the src/models directory
    model_save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'knn_regressor_model.joblib')

    # Save the trained model
    joblib.dump(model, model_save_path)
    print(f"\nTrained K-Nearest Neighbors Regressor model saved to: {model_save_path}")

except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Please ensure the path is correct.")
except ImportError as e:
    print(f"Error: Could not import necessary libraries: {e}")
    print("Please make sure you have pandas, scikit-learn, and joblib installed.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")