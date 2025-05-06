# src/data/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    """
    Preprocesses the nutritional data for machine learning, handling missing values.
    """
    print("Handling missing values...")
    # Impute missing numerical values with the mean
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    numerical_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    print("Missing numerical values imputed.")

    # Fill missing categorical values with 'Missing'
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    df[categorical_cols] = df[categorical_cols].fillna('Missing')
    print("Missing categorical values filled.")

    # Define Features (X) and Target (y)
    if 'nutrition_score' in df.columns:
        X = df.drop(['name', 'nutrition_score'], axis=1, errors='ignore')
        y = df['nutrition_score']
    else:
        X = df.drop('name', axis=1, errors='ignore')
        y = None
        print("Note: 'nutrition_score' column not found, proceeding without a target.")

    # Identify numerical and categorical columns again after imputation
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(include='object').columns.tolist()

    # Remove potential identifier columns
    if 'search_term' in numerical_cols:
        numerical_cols.remove('search_term')
    if 'search_term' in categorical_cols:
        categorical_cols.remove('search_term')

    print("\nNumerical Columns for Preprocessing:", numerical_cols)
    print("Categorical Columns for Preprocessing:", categorical_cols)

    # Create Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    # Apply Preprocessing
    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor

if __name__ == "__main__":
    file_path = "C:\\Users\\aman2\\OneDrive\\Desktop\\4th semester\\Nutritional Analysis\\nutritional-analysis\\data\\processed\\cleaned_data.csv"
    try:
        cleaned_df = pd.read_csv(file_path)
        print(f"Successfully loaded data from: {file_path}\n")

        X_processed, y, preprocessor = preprocess_data(cleaned_df.copy())

        print("\nShape of Processed Features (X_processed):", X_processed.shape)
        if y is not None:
            print("Shape of Target (y):", y.shape)
        print("\nPreprocessing function executed successfully.")

        # You can save the preprocessor here if needed for later use
        # import joblib
        # joblib.dump(preprocessor, 'src/models/preprocessor.joblib')

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the path is correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # src/data/preprocessing.py

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline

# def preprocess_data(df):
#     """
#     Preprocesses the nutritional data for machine learning.

#     Args:
#         df (pd.DataFrame): The cleaned nutritional data.

#     Returns:
#         tuple: A tuple containing the preprocessed features (X_processed),
#                the target variable (y) if defined (None for now),
#                and the preprocessor object for later use.
#     """
#     # 1. Handle Missing Values (Example: Filling 'brand')
#     df['brand'].fillna('Generic', inplace=True)
#     print("Missing values in 'brand' filled with 'Generic'.")

#     # 2. Define Features (X) and Target (y)
#     # For a general nutritional analysis, we might not have a specific target yet.
#     # Let's consider 'nutrition_score' as a potential target for a regression task later,
#     # or we can proceed without a target for unsupervised learning initially.
#     if 'nutrition_score' in df.columns:
#         X = df.drop(['name', 'nutrition_score'], axis=1, errors='ignore')
#         y = df['nutrition_score']
#     else:
#         X = df.drop('name', axis=1, errors='ignore')
#         y = None
#         print("Note: 'nutrition_score' column not found, proceeding without a target.")

#     # 3. Identify Numerical and Categorical Columns
#     numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
#     categorical_cols = X.select_dtypes(include='object').columns.tolist()

#     # Remove potential identifier columns
#     if 'search_term' in numerical_cols:
#         numerical_cols.remove('search_term')
#     if 'search_term' in categorical_cols:
#         categorical_cols.remove('search_term')

#     print("\nNumerical Columns for Preprocessing:", numerical_cols)
#     print("Categorical Columns for Preprocessing:", categorical_cols)

#     # 4. Create Preprocessing Pipeline
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numerical_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#         ])

#     # 5. Apply Preprocessing
#     X_processed = preprocessor.fit_transform(X)

#     return X_processed, y, preprocessor

# if __name__ == "__main__":
#     file_path = "C:\\Users\\aman2\\OneDrive\\Desktop\\4th semester\\Nutritional Analysis\\nutritional-analysis\\data\\processed\\cleaned_data.csv"
#     try:
#         cleaned_df = pd.read_csv(file_path)
#         print(f"Successfully loaded data from: {file_path}\n")

#         X_processed, y, preprocessor = preprocess_data(cleaned_df)

#         print("\nShape of Processed Features (X_processed):", X_processed.shape)
#         if y is not None:
#             print("Shape of Target (y):", y.shape)
#         print("\nPreprocessing function executed successfully.")

#         # You can save the preprocessor here if needed for later use
#         # import joblib
#         # joblib.dump(preprocessor, 'src/models/preprocessor.joblib')

#     except FileNotFoundError:
#         print(f"Error: File not found at {file_path}. Please ensure the path is correct.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
        
        
# # print("srk")