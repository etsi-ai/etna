import sys
from unittest.mock import MagicMock
import pandas as pd

# =========================
# Mock Rust backend
# =========================
mock_rust = MagicMock()
mock_model_instance = MagicMock()

mock_model_instance.train.return_value = [0.1, 0.05]
mock_model_instance.predict.return_value = [0, 1, 0]

mock_rust.EtnaModel.return_value = mock_model_instance
mock_rust.EtnaModel.load.return_value = mock_model_instance

sys.modules["etna._etna_rust"] = mock_rust

# =========================
# Mock MLflow (CRITICAL)
# =========================
mock_mlflow = MagicMock()
mock_mlflow.start_run.return_value.__enter__.return_value = None
mock_mlflow.start_run.return_value.__exit__.return_value = None

sys.modules["mlflow"] = mock_mlflow

# =========================
# Import AFTER mocks
# =========================
from etna.api import Model


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
