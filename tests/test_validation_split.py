"""
Test script to verify validation split and MLflow val_loss tracking.
Uses mocking to bypass Rust backend compilation requirement.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _create_mock_setup(n_samples=100, n_features=4, output_dim=2, task_type="classification"):
    """Helper to create a mock Rust backend and model for testing."""
    mock_etna_rust = MagicMock()
    mock_model = MagicMock()

    # Simulate train() that calls progress_callback each epoch
    def mock_train(
        X, y, epochs, lr, batch_size, weight_decay, optimizer,
        early_stopping=False, patience=10, restore_best=True,
        progress_callback=None,
    ):
        losses = []
        for epoch in range(epochs):
            loss = 0.5 - (epoch * 0.01)
            losses.append(loss)
            if progress_callback:
                progress_callback(epoch, epochs, loss)
        return losses

    # Simulate forward() returning raw predictions
    def mock_forward(X_val):
        n = X_val.shape[0] if hasattr(X_val, 'shape') else len(X_val)
        if task_type == "classification":
            # Return softmax-like outputs
            return [[0.7, 0.3] for _ in range(n)]
        else:
            # Return regression predictions
            return [[0.5] for _ in range(n)]

    mock_model.train.side_effect = mock_train
    mock_model.forward.side_effect = mock_forward
    mock_etna_rust.EtnaModel.return_value = mock_model

    return mock_etna_rust, mock_model


def _get_patched_model(mock_etna_rust, task_type="classification", n_samples=100):
    """Create a patched Model instance for testing."""
    import importlib
    import etna.api
    importlib.reload(etna.api)
    etna.api._etna_rust = mock_etna_rust

    mock_preprocessor = MagicMock()
    X_data = np.random.randn(n_samples, 4).astype(np.float32)
    if task_type == "classification":
        y_data = np.zeros((n_samples, 2), dtype=np.float32)
        y_data[np.arange(n_samples), np.random.randint(0, 2, n_samples)] = 1.0
    else:
        y_data = np.random.randn(n_samples, 1).astype(np.float32)

    mock_preprocessor.fit_transform.return_value = (X_data, y_data)
    mock_preprocessor.output_dim = 2 if task_type == "classification" else 1

    with patch.object(etna.api, 'load_data') as mock_load:
        mock_load.return_value = MagicMock()
        with patch.object(etna.api, 'Preprocessor') as mock_prep:
            mock_prep.return_value = mock_preprocessor
            model = etna.api.Model("dummy.csv", "target", task_type=task_type, seed=42)

    return model, etna.api


# ---- Tests ----


def test_validation_split_default():
    """Verify val_loss_history is populated with default validation_split=0.2."""
    mock_etna_rust, mock_model = _create_mock_setup()
    model, api_mod = _get_patched_model(mock_etna_rust)
    model.train(epochs=5, lr=0.01)

    assert len(model.loss_history) == 5, f"Expected 5 loss entries, got {len(model.loss_history)}"
    assert len(model.val_loss_history) == 5, f"Expected 5 val_loss entries, got {len(model.val_loss_history)}"
    # forward() should have been called once per epoch for validation
    assert mock_model.forward.call_count == 5, f"Expected 5 forward() calls, got {mock_model.forward.call_count}"


def test_validation_split_zero_disables_validation():
    """Verify no validation when validation_split=0.0."""
    mock_etna_rust, mock_model = _create_mock_setup()
    model, api_mod = _get_patched_model(mock_etna_rust)
    model.train(epochs=5, lr=0.01, validation_split=0.0)

    assert len(model.loss_history) == 5
    assert len(model.val_loss_history) == 0, "val_loss_history should be empty when validation is disabled"
    assert mock_model.forward.call_count == 0, "forward() should not be called when validation is disabled"


def test_validation_split_data_sizes():
    """Verify correct train/val split sizes are passed to Rust."""
    n_samples = 100
    validation_split = 0.2
    mock_etna_rust, mock_model = _create_mock_setup(n_samples=n_samples)
    model, api_mod = _get_patched_model(mock_etna_rust, n_samples=n_samples)
    model.train(epochs=3, lr=0.01, validation_split=validation_split)

    # Check that Rust train() received the training portion only
    train_call_args = mock_model.train.call_args
    X_train_passed = train_call_args[0][0]  # First positional arg
    expected_train_size = n_samples - max(1, int(n_samples * validation_split))

    assert X_train_passed.shape[0] == expected_train_size, (
        f"Expected {expected_train_size} training samples, got {X_train_passed.shape[0]}"
    )


def test_validation_split_invalid_values():
    """Verify error for invalid validation_split values."""
    mock_etna_rust, _ = _create_mock_setup()
    model, api_mod = _get_patched_model(mock_etna_rust)

    with pytest.raises(ValueError, match="validation_split must be"):
        model.train(epochs=3, lr=0.01, validation_split=1.0)

    # Re-create model since the previous call may have modified state
    model2, _ = _get_patched_model(mock_etna_rust)
    with pytest.raises(ValueError, match="validation_split must be"):
        model2.train(epochs=3, lr=0.01, validation_split=-0.1)


def test_validation_split_regression():
    """Verify validation works for regression tasks."""
    mock_etna_rust, mock_model = _create_mock_setup(task_type="regression")
    model, api_mod = _get_patched_model(mock_etna_rust, task_type="regression")
    model.train(epochs=5, lr=0.01, validation_split=0.3)

    assert len(model.val_loss_history) == 5
    # All val_loss values should be finite numbers
    for vl in model.val_loss_history:
        assert np.isfinite(vl), f"Validation loss should be finite, got {vl}"


def test_training_loop_stays_in_rust():
    """Verify that train() is called exactly once (loop stays in Rust)."""
    mock_etna_rust, mock_model = _create_mock_setup()
    model, api_mod = _get_patched_model(mock_etna_rust)
    model.train(epochs=10, lr=0.01)

    assert mock_model.train.call_count == 1, (
        f"Expected 1 train call (all epochs in Rust), got {mock_model.train.call_count}"
    )


def test_seed_reproducibility():
    """Verify that the same seed produces the same val split."""
    mock_etna_rust1, mock_model1 = _create_mock_setup()
    model1, _ = _get_patched_model(mock_etna_rust1)
    model1.train(epochs=3, lr=0.01, validation_split=0.2)
    val_losses_1 = list(model1.val_loss_history)

    mock_etna_rust2, mock_model2 = _create_mock_setup()
    model2, _ = _get_patched_model(mock_etna_rust2)
    model2.train(epochs=3, lr=0.01, validation_split=0.2)
    val_losses_2 = list(model2.val_loss_history)

    # Same seed=42 should give same val_loss values
    assert val_losses_1 == val_losses_2, "Same seed should produce identical validation losses"


def test_val_loss_in_save_model_mlflow():
    """Verify save_model() logs val_loss to MLflow."""
    mock_etna_rust, mock_model = _create_mock_setup()
    model, api_mod = _get_patched_model(mock_etna_rust)
    model.train(epochs=3, lr=0.01)

    # Mock MLflow
    mock_mlflow = MagicMock()
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    # Mock rust_model.save to not actually write files
    model.rust_model.save = MagicMock()
    model.preprocessor = MagicMock()
    model.preprocessor.get_state.return_value = {"test": True}
    model._cached_X = np.array([[1, 2]])

    import builtins
    original_open = builtins.open

    with patch('builtins.open', MagicMock()):
        with patch.dict('sys.modules', {'mlflow': mock_mlflow}):
            with patch('os.environ.get', return_value=None):
                model.save_model(
                    path="test_model.json",
                    run_name="test_run",
                    mlflow_tracking_uri="http://localhost:5000"
                )

    # Check that val_loss was logged
    log_metric_calls = mock_mlflow.log_metric.call_args_list
    val_loss_calls = [c for c in log_metric_calls if c[0][0] == "val_loss"]
    assert len(val_loss_calls) == 3, f"Expected 3 val_loss log calls, got {len(val_loss_calls)}"


if __name__ == "__main__":
    test_validation_split_default()
    print("✅ test_validation_split_default PASSED")

    test_validation_split_zero_disables_validation()
    print("✅ test_validation_split_zero_disables_validation PASSED")

    test_validation_split_data_sizes()
    print("✅ test_validation_split_data_sizes PASSED")

    test_validation_split_invalid_values()
    print("✅ test_validation_split_invalid_values PASSED")

    test_validation_split_regression()
    print("✅ test_validation_split_regression PASSED")

    test_training_loop_stays_in_rust()
    print("✅ test_training_loop_stays_in_rust PASSED")

    test_seed_reproducibility()
    print("✅ test_seed_reproducibility PASSED")

    print("\n🎉 All validation split tests passed!")
