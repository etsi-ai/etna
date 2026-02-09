# User-facing API (Classifier, Regression)

import os
import json
##import mlflow
import pandas as pd
import numpy as np
from etna.utils import set_seed
from .utils import load_data
from .preprocessing import Preprocessor
from tqdm import tqdm

# Safe Rust import
try:
    from . import _etna_rust
except ImportError:
    _etna_rust = None


class Model:
    def __init__(self, file_path: str, target: str, task_type: str = None, hidden_layers: list = [64, 32], activation: str = "relu", seed: int = None):
        """
        Initializes the ETNA model.
        Args:
            file_path: Path to the .csv dataset
            target: Name of the target column
            task_type: 'classification', 'regression', or None (auto-detect)
            hidden_layers: List of neurons per hidden layer (e.g., [64, 32])
            seed: Optional random seed for reproducibility
        """
        self.file_path = file_path
        self.target = target
        self.df = load_data(file_path)
        self.loss_history = []

        # --- SEED LOGIC ---
        self.seed = seed
        if seed is not None:
            set_seed(seed)
        # ----------------------------
        # Store architecture parameters
        self.hidden_layers = hidden_layers
        self.activation = activation
        
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
                print(f"[*] Auto-Detected Task: Regression (Target '{target}')")

        self.preprocessor = Preprocessor(self.task_type)
        self.rust_model = None

        # Cached transformed data for persistence-safe prediction
        self._cached_X = None

    def train(self, epochs: int = 100, lr: float = 0.01, batch_size: int = 32, weight_decay: float = 0.0, optimizer: str = 'sgd'):
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Number of samples per gradient update (default: 32)
            weight_decay: L2 regularization coefficient (lambda)
            optimizer: Optimizer to use ('sgd' or 'adam')
        """
        if _etna_rust is None:
            raise ImportError(
                "Rust core is not available. Please build the Rust extension "
                "before calling model.train()."
            )

        print("[*] Preprocessing data...")
        X, y = self.preprocessor.fit_transform(self.df, self.target)

        # Cache training data for predict() without arguments
        self._cached_X = np.array(X)

        self.input_dim = len(X[0])
        self.output_dim = self.preprocessor.output_dim

        optimizer_lower = optimizer.lower()
        if optimizer_lower not in ['sgd', 'adam']:
            raise ValueError(f"Unsupported optimizer '{optimizer}'. Choose 'sgd' or 'adam'.")

        # LOGICAL FIX: Only initialize if model doesn't exist
        # Only initialize if model doesn't exist (supports incremental training)
        if self.rust_model is None:
            print(f"[*] Initializing Rust Core [In: {self.input_dim}, Out: {self.output_dim}]...")
            self.rust_model = _etna_rust.EtnaModel(
                self.input_dim,
                self.hidden_layers,  
                self.output_dim,
                self.task_code,
                self.activation
                # TODO: Pass self.seed here once the Rust core supports it
            )
        else:
            print(f"[*] Resuming training on existing Core [In: {self.input_dim}, Out: {self.output_dim}]...")

        optimizer_display = optimizer_lower.upper()
        if weight_decay > 0:
            print(f"[*] Training started (Optimizer: {optimizer_display}, L2 regularization: lambda={weight_decay})...")
        else:
            print(f"[*] Training started (Optimizer: {optimizer_display})...")

        # Create tqdm progress bar
        pbar = tqdm(total=epochs, desc="Training", unit="epoch")
        
        # Callback function that Rust calls after each epoch
        def progress_callback(epoch, total, loss):
            pbar.update(1)
            pbar.set_description(f"Loss: {loss:.4f}")
        
        # Single Rust call - training loop stays in Rust for performance
        new_losses = self.rust_model.train(
            X, y, epochs, lr, batch_size, weight_decay, 
            optimizer_lower, progress_callback=progress_callback
        )
        
        pbar.close()
        self.loss_history.extend(new_losses)
        print("[+] Training complete!")

    def predict(self, data_path: str = None):
        """
        Make predictions.

        Args:
            data_path: Optional path to CSV file. If not provided, uses the
                       training data (useful for evaluating on training set).

        Returns:
            List of predictions (class labels for classification, values for regression)
        """
        if self.rust_model is None:
            raise Exception("Model not trained yet! Call .train() first.")

        # Case 1: Predict from new CSV
        if data_path:
            df = load_data(data_path)
            print("Transforming input data...")
            X_new = self.preprocessor.transform(df)

        # Case 2: Predict on cached training data
        else:
            if self._cached_X is None:
                raise ValueError(
                    "No data available for prediction. "
                    "Pass a CSV path to predict(data_path=...)."
                )
            # Convert numpy array to list for Rust
            X_new = self._cached_X.tolist() if isinstance(self._cached_X, np.ndarray) else self._cached_X

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

    def summary(self):
        print("\n Model Summary")
        print("=" * 60)

        if self.rust_model is None:
            print("Model has not been trained yet.")
            print("Call model.train() before calling summary().")
            return


        l1_params = (self.input_dim * self.hidden_dim) + self.hidden_dim
        print(
            f"Layer 1 (Linear): {self.input_dim} -> {self.hidden_dim} "
            f"| Params: {l1_params}"
        )

        l2_params = (self.hidden_dim * self.output_dim) + self.output_dim
        print(
            f"Layer 2 (Linear): {self.hidden_dim} -> {self.output_dim} "
            f"| Params: {l2_params}"
        )

        print("=" * 60)
        total_params = l1_params + l2_params
        print(f"Total Trainable Params: {total_params}\n")

    def save_model(self, path="model_checkpoint.json", run_name="ETNA_Run", mlflow_tracking_uri=None):
        """
        Saves the model using Rust backend. Optionally tracks with MLflow if a URI is provided.
        """
        if self.rust_model is None:
            raise Exception("Model not trained yet!")

        path = str(path)
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        # Always save local files
        print(f"Saving model to {path}...")
        self.rust_model.save(path)

        preprocessor_path = path + ".preprocessor.json"
        state = self.preprocessor.get_state()
        state["_cached_X"] = self._cached_X.tolist() if self._cached_X is not None else None
        state["_target"] = self.target

        with open(preprocessor_path, "w") as f:
            json.dump(state, f)

        # Only use MLflow if tracking URI is provided and not disabled
        if mlflow_tracking_uri and os.environ.get("ETNA_DISABLE_MLFLOW") != "1":
            try:
                import mlflow
                print(f"Logging to MLflow at {mlflow_tracking_uri}...")
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment("ETNA_Experiments")

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("task_type", self.task_type)
                    mlflow.log_param("target_column", self.target)
                    for epoch, loss in enumerate(self.loss_history):
                        mlflow.log_metric("loss", loss, step=epoch)
                    mlflow.log_artifact(path)
                    mlflow.log_artifact(preprocessor_path)
                print("Model saved & tracked!")
            except ImportError:
                print("MLflow not installed. Skipping remote tracking.")
        else:
            print(f"Model saved locally to {path}. (MLflow tracking skipped)")

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

        print(f"[*] Loading model from {path}...")

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

        print("[+] Model loaded successfully!")
        return self
