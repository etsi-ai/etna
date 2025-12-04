# Preprocessing (Scaling, Encoding)
# Scaling : Min-Max,RobustScaler and so on.
# Encoding : One-Hot,Categorical..

import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, task_type="classification"):
        self.task_type = task_type
        self.numeric_means = {}
        self.numeric_stds = {}
        self.cat_mappings = {} 
        self.target_mapping = {}
        self.target_mean = 0.0
        self.target_std = 1.0
        self.output_dim = 1 

    def fit_transform(self, df: pd.DataFrame, target_col: str):
        X_df = df.drop(columns=[target_col])
        y_series = df[target_col]
        
        self.numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        X_processed = []
        
        # Numeric: Standard Scaling
        for col in self.numeric_cols:
            vals = X_df[col].fillna(X_df[col].mean()).values
            mean = np.mean(vals)
            std = np.std(vals) + 1e-8
            self.numeric_means[col] = mean
            self.numeric_stds[col] = std
            X_processed.append((vals - mean) / std)
            
        # Categorical: Simple Encoding
        for col in self.cat_cols:
            vals = X_df[col].fillna("Unknown").astype(str).values
            unique_vals = np.unique(vals)
            mapping = {v: i for i, v in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping
            encoded = np.array([mapping[v] for v in vals])
            X_processed.append(encoded / (len(mapping) + 1e-8))

        # Join features
        X_final = np.column_stack(X_processed).tolist() if X_processed else []

        # Target Processing
        if self.task_type == "classification":
            unique_targets = y_series.unique()
            self.output_dim = len(unique_targets)
            self.target_mapping = {v: i for i, v in enumerate(unique_targets)}
            y_indices = y_series.map(self.target_mapping).values
            
            # One-Hot Encoding
            y_final = np.zeros((len(y_indices), self.output_dim))
            y_final[np.arange(len(y_indices)), y_indices] = 1.0
            y_final = y_final.tolist()
            
        else: # Regression
            self.output_dim = 1
            y_vals = y_series.fillna(y_series.mean()).values
            self.target_mean = np.mean(y_vals)
            self.target_std = np.std(y_vals) + 1e-8
            y_scaled = (y_vals - self.target_mean) / self.target_std
            y_final = [[y] for y in y_scaled]

        return X_final, y_final

    def transform(self, df: pd.DataFrame):
        X_processed = []
        for col in self.numeric_cols:
            vals = df[col].fillna(self.numeric_means[col]).values
            X_processed.append((vals - self.numeric_means[col]) / self.numeric_stds[col])
        for col in self.cat_cols:
            vals = df[col].fillna("Unknown").astype(str).values
            mapping = self.cat_mappings[col]
            encoded = np.array([mapping.get(v, 0) for v in vals])
            X_processed.append(encoded / (len(mapping) + 1e-8))
        return np.column_stack(X_processed).tolist() if X_processed else []
