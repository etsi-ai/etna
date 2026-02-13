# Preprocessing (Scaling, Encoding)
# Scaling : StandardScaler
# Encoding : One-Hot

import numpy as np
import pandas as pd


class Preprocessor:
    def __init__(self, task_type="classification"):
        self.task_type = task_type

        self.numeric_means = {}
        self.numeric_stds = {}
        
        self.cat_mappings = {}
        self.cat_modes = {}          # store mode per categorical column
        
        self.target_mapping = {}
        self.target_mean = 0.0
        self.target_std = 1.0

        self.input_dim = 0
        self.output_dim = 1

    # -------------------------------------------------
    # FIT + TRANSFORM
    # -------------------------------------------------
    

    def fit_transform(self, df: pd.DataFrame, target_col: str):
        X_df = df.drop(columns=[target_col])
        y_series = df[target_col]

        """
        identify columns with object or category dtypes ( First point of the issue covered here)
        Doc : https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
        """
        
        self.numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

        X_processed = []
        

        # ---------- Numeric features ----------
        for col in self.numeric_cols:
            vals = X_df[col].fillna(X_df[col].mean()).values
            mean = np.mean(vals)
            std = np.std(vals) + 1e-8

            self.numeric_means[col] = mean
            self.numeric_stds[col] = std

            X_processed.append(((vals - mean) / std).reshape(-1, 1))
            

        # ---------- Categorical features (Mode + One-Hot) ----------
        for col in self.cat_cols:
            series = X_df[col]

            #  MODE IMPUTATION (instead of "Unknown") 
            mode = series.dropna().mode()
            mode_val = mode.iloc[0] if not mode.empty else None
            self.cat_modes[col] = mode_val



            vals = series.fillna(mode_val).astype(str).values
            unique_vals = pd.Series(vals).unique().tolist()

            if "__UNK__" not in unique_vals:
                unique_vals.append("__UNK__")




            mapping = {v: i for i, v in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping
            
            # Implemented One-Hot Encoding for these columns - fit phase( Second point of the issue covered here)
            '''
            Doc: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html
                 https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
            '''                 
            one_hot = np.zeros((len(vals), len(mapping)))
            


            unk_index = mapping["__UNK__"]

            for i, v in enumerate(vals):
                index = mapping.get(v, unk_index)
                one_hot[i, index] = 1.0




            X_processed.append(one_hot)

        # ---------- Final feature matrix ----------
        """
        Doc: https://www.tensorflow.org/guide/keras/sequential_model
             Ensure the model’s first layer receives the correct input feature dimension (Third point of the issue covered here)
        """
        X_final = np.hstack(X_processed) if X_processed else np.empty((len(df), 0))
        self.input_dim = X_final.shape[1]

        # ---------- Target processing ----------
        if self.task_type == "classification":
            unique_targets = y_series.unique()
            self.output_dim = len(unique_targets)

            self.target_mapping = {v: i for i, v in enumerate(unique_targets)}
            y_idx = y_series.map(self.target_mapping).values

            y_final = np.zeros((len(y_idx), self.output_dim))
            y_final[np.arange(len(y_idx)), y_idx] = 1.0

        else:  # regression
            self.output_dim = 1
            y_vals = y_series.fillna(y_series.mean()).values

            self.target_mean = np.mean(y_vals)
            self.target_std = np.std(y_vals) + 1e-8

            y_scaled = (y_vals - self.target_mean) / self.target_std
            y_final = y_scaled.reshape(-1, 1)

        return X_final, y_final
    
        
    # -------------------------------------------------
    # TRANSFORM ONLY (INFERENCE)
    # -------------------------------------------------

    def transform(self, df: pd.DataFrame):
        X_processed = []

        for col in self.numeric_cols:
            vals = df[col].fillna(self.numeric_means[col]).values
            scaled = (vals - self.numeric_means[col]) / self.numeric_stds[col]
            X_processed.append(scaled.reshape(-1, 1))

        for col in self.cat_cols:
            mode_val = self.cat_modes[col]
            mapping = self.cat_mappings[col]

            vals = df[col].fillna(mode_val).astype(str).values

            # Implemented One-Hot Encoding for these columns - transform phase( Second point of the issue covered here)
            one_hot = np.zeros((len(vals), len(mapping)))
            unk_index = mapping["__UNK__"]

            for i, v in enumerate(vals):
                index = mapping.get(v, unk_index)
                one_hot[i, index] = 1.0


            X_processed.append(one_hot)

        X_final = np.hstack(X_processed) if X_processed else np.empty((len(df), 0))
        return X_final

    def get_state(self):
        return {
            "task_type": self.task_type,
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "cat_mappings": self.cat_mappings,
            "cat_modes": self.cat_modes,      # persisted
            "target_mapping": self.target_mapping,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "numeric_cols": self.numeric_cols,
            "cat_cols": self.cat_cols,
        }

    def set_state(self, state: dict):
        self.task_type = state["task_type"]
        self.numeric_means = state["numeric_means"]
        self.numeric_stds = state["numeric_stds"]
        self.cat_mappings = state["cat_mappings"]
        self.cat_modes = state["cat_modes"]   # restored
        self.target_mapping = state["target_mapping"]
        self.target_mean = state["target_mean"]
        self.target_std = state["target_std"]
        self.input_dim = state["input_dim"]
        self.output_dim = state["output_dim"]
        self.numeric_cols = state["numeric_cols"]
        self.cat_cols = state["cat_cols"]
