def load_csv(file_path):
    import pandas as pd
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def save_csv(dataframe, file_path):
    """Save a DataFrame to a CSV file."""
    dataframe.to_csv(file_path, index=False)

def clean_column_names(dataframe):
    """Clean column names by stripping whitespace and converting to lowercase."""
    dataframe.columns = [col.strip().lower() for col in dataframe.columns]
    return dataframe

def check_missing_values(dataframe):
    """Check for missing values in the DataFrame."""
    return dataframe.isnull().sum()

def normalize_column(dataframe, column_name):
    """Normalize a specified column in the DataFrame."""
    max_value = dataframe[column_name].max()
    min_value = dataframe[column_name].min()
    dataframe[column_name] = (dataframe[column_name] - min_value) / (max_value - min_value)
    return dataframe