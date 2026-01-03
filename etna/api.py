# User-facing API (Classifier, Regression)

import os
import json
##import mlflow
import pandas as pd
import numpy as np

from .utils import load_data
from .preprocessing import Preprocessor

# Safe Rust import
try:
    from . import _etna_rust
except ImportError:
    _etna_rust = None


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
        self.loss_history = []

        # Determine task type
        if task_type:
            self.task_type = task_type.lower()
            self.task_code = 1 if self.task_type == "regression" else 0
            print(f"[*] User Task: {self.task_type.capitalize()} (Target '{target}')")
        else:
            target_data = self.df[target]
            is_numeric = pd.api.types.is_numeric_dtype(target_data)
            num_unique = target_data.nunique()

            if not is_numeric or (num_unique < 20 and num_unique < len(self.df) * 0.5):
                self.task_type = "classification"
                self.task_code = 0
                print(f"[*] Auto-Detected Task: Classification (Target '{target}')")
            else:
                self.task_type = "regression"
                self.task_code = 1
                print(f"🔮 Auto-Detected Task: Regression (Target '{target}')")

        self.preprocessor = Preprocessor(self.task_type)
        self.rust_model = None

        # Cached transformed data for persistence-safe prediction
        self._cached_X = None

    def train(self, epochs: int = 100, lr: float = 0.01):
        if _etna_rust is None:
            raise ImportError(
                "Rust core is not available. Please build the Rust extension "
                "before calling model.train()."
            )

        print("⚙️  Preprocessing data...")
        X, y = self.preprocessor.fit_transform(self.df, self.target)

        # Cache transformed training data
        self._cached_X = np.array(X)

        input_dim = len(X[0])
        hidden_dim = 16
        output_dim = self.preprocessor.output_dim

        print(f"🚀 Initializing Rust Core [In: {input_dim}, Out: {output_dim}]...")
        self.rust_model = _etna_rust.EtnaModel(
            input_dim, hidden_dim, output_dim, self.task_code
        )

        print("🔥 Training started...")
        self.loss_history = self.rust_model.train(X, y, epochs, lr)
        print("✅ Training complete!")

    def predict(self, data_path: str = None):
        if self.rust_model is None:
            raise Exception("Model not trained yet! Call .train() first.")

        # Case 1: Predict from new CSV
        if data_path:
            df = load_data(data_path)
            print("Transforming input data...")
            X_new = self.preprocessor.transform(df)

        # Case 2: Predict on cached training data (after load)
        else:
            if self._cached_X is None:
                raise ValueError(
                    "No data available for prediction. "
                    "Pass a CSV path to predict(data_path=...)."
                )
            X_new = self._cached_X

        preds = self.rust_model.predict(X_new)

        if self.task_type == "classification":
            inv_map = {v: k for k, v in self.preprocessor.target_mapping.items()}
            return [inv_map.get(int(p), "Unknown") for p in preds]
        else:
            results = [
                (p * self.preprocessor.target_std) + self.preprocessor.target_mean
                for p in preds
            ]
            return [float(r) for r in results]

    def save_model(self, path="model_checkpoint.json", run_name="ETNA_Run"):
        """
        Saves the model using Rust backend AND tracks it with MLflow.
        """
        if self.rust_model is None:
            raise Exception("Model not trained yet!")

        path = str(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save Rust model
        print(f"Saving model to {path}...")
        self.rust_model.save(path)

        # Save preprocessor + cached state
        preprocessor_path = path + ".preprocessor.json"
        state = self.preprocessor.get_state()
        state["_cached_X"] = (
            self._cached_X.tolist() if self._cached_X is not None else None
        )
        state["_target"] = self.target

        with open(preprocessor_path, "w") as f:
            json.dump(state, f)

        # Skip MLflow in test environment
        if os.environ.get("ETNA_DISABLE_MLFLOW") == "1":
            return

        #Lazy import
        import mlflow
        
        # Log to MLflow
        print("Logging to MLflow...")
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("ETNA_Experiments")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("target_column", self.target)

            print(f"📈 Logging {len(self.loss_history)} metrics points...")
            for epoch, loss in enumerate(self.loss_history):
                mlflow.log_metric("loss", loss, step=epoch)

            mlflow.log_artifact(path)
            mlflow.log_artifact(preprocessor_path)

        print("Model saved & tracked!")
        print("View at: http://localhost:5000")

    @classmethod
    def load(cls, path: str):
        """
        Loads a saved model checkpoint along with preprocessing state.
        """
        if _etna_rust is None:
            raise ImportError(
                "Rust core is not available. Please build the Rust extension "
                "before loading a model."
            )

        path = str(path)
        preprocessor_path = path + ".preprocessor.json"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Missing preprocessor state file: {preprocessor_path}"
            )

        print(f"📂 Loading model from {path}...")

        # Create instance without __init__
        self = cls.__new__(cls)

        # Load Rust backend
        self.rust_model = _etna_rust.EtnaModel.load(path)

        # Load preprocessor state
        with open(preprocessor_path, "r") as f:
            state = json.load(f)

        self.task_type = state["task_type"]
        self.task_code = 1 if self.task_type == "regression" else 0

        self.preprocessor = Preprocessor(self.task_type)
        self.preprocessor.set_state(state)

        cached_X = state.get("_cached_X")
        self._cached_X = np.array(cached_X) if cached_X is not None else None

        # Restore metadata
        self.target = state.get("_target")
        self.file_path = None
        self.df = None
        self.loss_history = []

        print("✅ Model loaded successfully!")
        return self
