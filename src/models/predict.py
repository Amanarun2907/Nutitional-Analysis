# src/models/predict.py

import pandas as pd
import joblib
import os
import sys

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessing import preprocess_data  # Import for reference (not direct use)

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
    print(f"Error: Preprocessor not found. Make sure you saved it during training (e.g., by running train_3.py).")
    sys.exit(1)

# *** ADJUST THIS SECTION BASED ON HOW YOU GET NEW DATA ***
new_data = pd.DataFrame({
    'serving_size_g': [150, 75],
    'calories': [250, 120],
    'protein_g': [12, 6],
    'fat_total_g': [18, 9],
    'carbs_g': [8, 4],
    'fiber_g': [3, 1.5],
    'sugar_g': [4, 2],
    'calcium_mg': [120, 60],
    'iron_mg': [2.5, 1.2],
    'potassium_mg': [250, 125],
    'sodium_mg': [350, 175],
    'vitamin_c_mg': [6, 3],
    'vitamin_a_iu': [600, 300],
    'vitamin_d_iu': [120, 60],
    'cholesterol_mg': [25, 12],
    'saturated_fat_g': [6, 3],
    'trans_fat_g': [0.2, 0],
    'category': ['Breakfast', 'Beverages'],
    'brand': ['BrandX', 'GenericY']
    # Add all other relevant columns here
})
# *** END OF DATA INPUT SECTION ***

# Perform feature engineering (same as in preprocess_data)
new_data['calories_per_100g'] = new_data['calories'] / new_data['serving_size_g'] * 100
new_data['protein_per_100g'] = new_data['protein_g'] / new_data['serving_size_g'] * 100
new_data['fat_per_100g'] = new_data['fat_total_g'] / new_data['serving_size_g'] * 100
new_data['carbs_per_100g'] = new_data['carbs_g'] / new_data['serving_size_g'] * 100

total_macros = new_data['protein_g'] + new_data['fat_total_g'] + new_data['carbs_g'] + 1e-6 # Adding a small epsilon to avoid division by zero
new_data['protein_ratio'] = new_data['protein_g'] / total_macros
new_data['fat_ratio'] = new_data['fat_total_g'] / total_macros
new_data['carb_ratio'] = new_data['carbs_g'] / total_macros

# Preprocess the new data using the loaded preprocessor
try:
    numerical_features = ['serving_size_g', 'calories', 'protein_g', 'fat_total_g', 'carbs_g', 'fiber_g', 'sugar_g', 'calcium_mg', 'iron_mg', 'potassium_mg', 'sodium_mg', 'vitamin_c_mg', 'vitamin_a_iu', 'vitamin_d_iu', 'cholesterol_mg', 'saturated_fat_g', 'trans_fat_g', 'calories_per_100g', 'protein_per_100g', 'fat_per_100g', 'carbs_per_100g', 'protein_ratio', 'fat_ratio', 'carb_ratio']
    categorical_features = ['category', 'brand']
    X_new = new_data[numerical_features + categorical_features].copy()
    X_new_processed = preprocessor.transform(X_new)
except Exception as e:
    print(f"Error during preprocessing of new data: {e}")
    sys.exit(1)

# Make predictions
try:
    predictions = model.predict(X_new_processed)
    print("\nPredictions:")
    for i, prediction in enumerate(predictions):
        print(f"Sample {i+1}: Predicted Nutrition Score = {prediction:.2f}")
except Exception as e:
    print(f"Error during prediction: {e}")
    sys.exit(1)

if __name__ == '__main__':
    pass