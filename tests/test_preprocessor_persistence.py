import sys
from unittest.mock import MagicMock
import pandas as pd

# =========================
# Mock Rust backend module
# =========================
mock_rust_module = MagicMock()

mock_model_instance = MagicMock()
mock_model_instance.train.return_value = ([0.1], [])  # Return tuple: (train_losses, val_losses)
mock_model_instance.predict.return_value = [0, 1, 0]

# IMPORTANT: simulate Rust writing a model file
def fake_save(path):
    with open(path, "w") as f:
        f.write("mock rust model")

mock_model_instance.save.side_effect = fake_save

mock_rust_module.EtnaModel.return_value = mock_model_instance
mock_rust_module.EtnaModel.load.return_value = mock_model_instance

sys.modules["etna._etna_rust"] = mock_rust_module

# =========================
# Mock MLflow
# =========================
mock_mlflow = MagicMock()
mock_mlflow.start_run.return_value.__enter__.return_value = None
mock_mlflow.start_run.return_value.__exit__.return_value = None
mock_mlflow.__version__ = "0.0.0"
sys.modules["mlflow"] = mock_mlflow

# =========================
# Import AFTER mocks
# =========================
from etna.api import Model
import etna.api  # important

# PATCH THE CACHED SYMBOL
etna.api._etna_rust = mock_rust_module


def test_model_save_load_preserves_preprocessing(tmp_path):
    df = pd.DataFrame({
        "age": [20, 30, 40],
        "city": ["A", "B", "A"],
        "label": ["yes", "no", "yes"]
    })

    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)

    model = Model(str(csv_path), target="label")
    model.train(epochs=1)

    model_path = tmp_path / "etna_model"
    model.save_model(str(model_path))

    loaded_model = Model.load(str(model_path))

    preds_original = model.predict()
    preds_loaded = loaded_model.predict()

    assert preds_original == preds_loaded
