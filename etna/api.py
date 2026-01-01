#  User-facing API (Classifier, Regression)

import mlflow
import os
from .utils import load_data
from .preprocessing import Preprocessor
from . import _etna_rust 
import pandas as pd
import numpy as np

class Model:
    def __init__(self, file_path: str, target: str, task_type: str = None):
        """
        Initializes the ETNA model.
        Args:
            file_path: Path to the .csv dataset
            target: Name of the target column
            task_type: 'classification', 'regression', or None (auto-detect)
        """
        self.file_path = file_path
        self.target = target
        self.df = load_data(file_path)
        self.loss_history = [] # Store loss history
        
        # 1. Determine Task Type
        if task_type:
            # User override
            self.task_type = task_type.lower()
            self.task_code = 1 if self.task_type == "regression" else 0
            print(f"[*] User Task: {self.task_type.capitalize()} (Target '{target}')")
        else:
            # Auto-detect
            target_data = self.df[target]
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            num_unique = target_data.nunique()
            
            # Heuristic: If it's string OR (it's numeric BUT has few unique values relative to size), assume classification
            if not is_numeric or (num_unique < 20 and num_unique < len(self.df) * 0.5):
                self.task_type = "classification"
                self.task_code = 0
                print(f"[*] Auto-Detected Task: Classification (Target '{target}')")
            else:
                self.task_type = "regression"
                self.task_code = 1
                print(f"[*] Auto-Detected Task: Regression (Target '{target}')")
            
        self.preprocessor = Preprocessor(self.task_type)
        self.rust_model = None

    def train(self, epochs=100, lr=0.01, optimizer='sgd'):
        """
        Train the model.
        Args:
            epochs: Number of training epochs
            lr: Learning rate
            optimizer: 'sgd' or 'adam' (default: 'sgd')
        """
        print("[*] Preprocessing data...")
        X, y = self.preprocessor.fit_transform(self.df, self.target)
        
        input_dim = len(X[0])
        hidden_dim = 16 
        output_dim = self.preprocessor.output_dim
        
        print(f"[*] Initializing Rust Core [In: {input_dim}, Out: {output_dim}]...")
        self.rust_model = _etna_rust.EtnaModel(input_dim, hidden_dim, output_dim, self.task_code)
        
        # Normalize optimizer name
        optimizer_lower = optimizer.lower()
        if optimizer_lower not in ['sgd', 'adam']:
            print(f"[!] Unknown optimizer '{optimizer}', defaulting to 'sgd'")
            optimizer_lower = 'sgd'
        
        print(f"[*] Training started with {optimizer_lower.upper()} optimizer...")
        self.loss_history = self.rust_model.train(X, y, epochs, lr, optimizer_lower)
        print("[*] Training complete!")

    def predict(self, data_path=None):
        if self.rust_model is None:
            raise Exception("Model not trained yet! Call .train() first.")
            
        if data_path:
            df = load_data(data_path)
        else:
            df = self.df.drop(columns=[self.target])
            
        print("[*] Transforming input data...")
        X_new = self.preprocessor.transform(df)
        preds = self.rust_model.predict(X_new)
        
        if self.task_type == "classification":
            inv_map = {v: k for k, v in self.preprocessor.target_mapping.items()}
            # Convert to standard Python types
            return [inv_map.get(int(p), "Unknown") for p in preds]
        else:
            # Reverse scaling for regression and return Python floats
            results = [(p * self.preprocessor.target_std) + self.preprocessor.target_mean for p in preds]
            return [float(r) for r in results]

    def save_model(self, path="model_checkpoint.json", run_name="ETNA_Run"):
        """
        Saves the model using Rust backend AND tracks it with MLflow.
        """
        if self.rust_model is None:
            raise Exception("Model not trained yet!")

        # 1. Save locally using Rust
        print(f"Saving model to {path}...")
        self.rust_model.save(path)

        # 2. Log to MLflow (The "Unified" part) - optional, don't fail if MLflow is unavailable
        try:
            print("Logging to MLflow...")
            
            # Point to local storage for simplicity (as he requested)
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("ETNA_Experiments")

            with mlflow.start_run(run_name=run_name):
                # Log Parameters
                mlflow.log_param("task_type", self.task_type)
                mlflow.log_param("target_column", self.target)
                
                print(f"[*] Logging {len(self.loss_history)} metrics points...")
                for epoch, loss in enumerate(self.loss_history):
                    mlflow.log_metric("loss", loss, step=epoch)

                # Log the Model File (Artifact)
                mlflow.log_artifact(path)

                print("Model saved & tracked!")
                print("View at: http://localhost:5000")
        except Exception as e:
            # MLflow is optional - model is already saved locally
            print(f"[!] MLflow tracking unavailable (model still saved locally): {e}")
            print("[!] To enable MLflow tracking, start MLflow server: mlflow ui")

    @classmethod
    def load(cls, path: str):
        """
        Loads a saved model checkpoint.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        print(f"[*] Loading model from {path}...")

        # 1. Create a raw instance (bypassing __init__)
        self = cls.__new__(cls)
        
        # 2. Load the Rust backend
        try:
            self.rust_model = _etna_rust.EtnaModel.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Rust backend: {e}")

        # 3. Set placeholder state (since we didn't save preprocessors in this MVP)
        self.file_path = None
        self.target = "Unknown (Loaded)"
        self.loss_history = []
        
        # Default assumption to prevent crashes
        self.task_type = "classification" 
        self.preprocessor = Preprocessor(self.task_type)

        print("[*] Model loaded successfully!")
        return self