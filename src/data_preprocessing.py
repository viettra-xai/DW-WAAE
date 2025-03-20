import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from src.utils import load_data  # Import from utils.py

# Data Preprocessing Functions
def preprocess_data(data_path, filename):
    """Loads data, removes constant columns, and converts labels into binary format."""
    data = load_data(data_path, filename)
    constant_cols = data.columns[(data.nunique() == 1) | (data.nunique() == 2)]
    data = data.drop(columns=constant_cols)
    X, y = data.iloc[:, :-1], data.iloc[:, -1]  # Split features and labels
    y_bi = np.where(y == 0, 0, 1)  # Convert labels to binary
    return X, y_bi

def split_and_sample_data(X, y, abnormal_sample_ratio):
    """
    Splits data into train-test sets, samples a fraction of abnormal data, and shuffles the dataset.

    Parameters:
    - X: Features
    - y: Labels
    - abnormal_sample_ratio: Fraction of abnormal data to keep in training

    Returns:
    - X_train_shuffled, y_train_shuffled, X_test_full, y_test_full
    """
    # Split into training and testing
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.50, random_state=42)

    # Separate normal and abnormal samples
    X_train_normal = X_train_full[y_train_full == 0]
    y_train_normal = y_train_full[y_train_full == 0]
    X_train_abnormal = X_train_full[y_train_full != 0]
    y_train_abnormal = y_train_full[y_train_full != 0]

    # Convert y_train_abnormal to a pandas Series for sampling
    y_train_normal_series = pd.Series(y_train_normal, name="label")
    y_train_abnormal_series = pd.Series(y_train_abnormal, name="label")

    # Sample a fraction of abnormal data
    sample_size_abnormal = int(abnormal_sample_ratio * len(X_train_abnormal))
    X_sampled_abnormal = X_train_abnormal.sample(n=sample_size_abnormal, random_state=42)
    y_sampled_abnormal = y_train_abnormal_series.sample(n=sample_size_abnormal, random_state=42)

    # Combine normal and sampled abnormal data
    X_train_combined = pd.concat([X_train_normal, X_sampled_abnormal])
    y_train_combined = pd.concat([y_train_normal_series, y_sampled_abnormal])

    # Shuffle the dataset
    shuffled_indices = np.random.permutation(len(X_train_combined))
    X_train_shuffled = X_train_combined.iloc[shuffled_indices]
    y_train_shuffled = y_train_combined.iloc[shuffled_indices]

    return X_train_shuffled, y_train_shuffled, X_test_full, y_test_full


def normalize_data(X_train, X_test, norm_type="minmax"):    
    if norm_type == "minmax":
        scaler = MinMaxScaler()
    elif norm_type == "standard":
        scaler = StandardScaler()
    elif norm_type == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid norm_type. Choose from 'minmax', 'standard', or 'robust'.")

    X_train_normalized = scaler.fit_transform(X_train.astype("float32"))
    X_test_normalized = scaler.transform(X_test.astype("float32"))

    return X_train_normalized, X_test_normalized

