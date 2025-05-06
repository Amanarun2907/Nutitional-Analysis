# src/models/train_2.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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

    print("Shape of X_processed:", X_processed.shape)
    print("Shape of y:", y.shape)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.99, random_state=42) # Use only 1% for training

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    # Initialize and train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    print("RandomForestRegressor initialized.")
    model.fit(X_train, y_train)
    print("RandomForestRegressor fitted.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    print("Predictions made.")

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Random Forest Regressor Model Evaluation:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")

    # Define the path to save the trained model relative to the src/models directory
    model_save_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'random_forest_model.joblib')

    # Save the trained model
    joblib.dump(model, model_save_path)
    print(f"\nTrained Random Forest Regressor model saved to: {model_save_path}")

except FileNotFoundError as e:
    print(f"File Not Found Error: {e}")
except ImportError as e:
    print(f"Import Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred during training/evaluation: {e}")