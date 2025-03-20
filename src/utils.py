import os
import pandas as pd
import pickle
import sys

# Utility Functions (utils.py)
def load_data(path, filename):
    """Loads a CSV dataset into a Pandas DataFrame."""
    csv_load_path = os.path.join(path, filename)
    return pd.read_csv(csv_load_path, index_col=0)

def save_data(df, path, filename):
    """Saves a Pandas DataFrame as a CSV file."""
    csv_save_path = os.path.join(path, filename)
    df.to_csv(csv_save_path)

def load_file(file_name):
    """Loads a pickled file."""
    with open(file_name, 'rb') as f:
        return pickle.load(f)

class HiddenPrints:
    """Utility class to suppress print outputs."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
