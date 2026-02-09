# Utility functions (File loading, Helpers)
import pandas as pd
import os
import random  # Added for seeding
import numpy as np  # Added for seeding

def load_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV file into a Pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")

# --- Add your new code below ---

def set_seed(seed: int):
    """
    Sets the seed for reproducibility across python, numpy, and environment.
    """
    random.seed(seed)
    np.random.seed(seed)
    # This ensures that certain hash-based operations stay consistent
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    print(f"Global seed set to: {seed}")