# Utility functions (File loading, Helpers)

import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")
