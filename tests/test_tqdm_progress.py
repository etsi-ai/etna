"""
Test script to verify tqdm progress bar integration with Rust-side callbacks.
Uses mocking to bypass Rust backend compilation requirement.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import StringIO

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def test_tqdm_progress_bar():
    """Verify tqdm is properly integrated with callback-based progress reporting."""
    
    # Mock the Rust backend
    mock_etna_rust = MagicMock()
    mock_model = MagicMock()
    
    # Simulate train() that accepts early-stopping params and a progress_callback and calls it
    def mock_train(
        X,
        y,
        epochs,
        lr,
        batch_size,
        weight_decay,
        optimizer,
        early_stopping=False,
        patience=10,
        restore_best=True,
        progress_callback=None,
    ):
        losses = []
        for epoch in range(epochs):
            loss = 0.5 - (epoch * 0.01)  # Simulate decreasing loss
            losses.append(loss)
            # Call the progress callback if provided (simulates Rust behavior)
            if progress_callback:
                progress_callback(epoch, epochs, loss)
        return losses
    
    mock_model.train.side_effect = mock_train
    mock_etna_rust.EtnaModel.return_value = mock_model
    
    # Patch the Rust import
    with patch.dict('sys.modules', {'etna._etna_rust': mock_etna_rust}):
        # Now import the module - it will use our mock
        import importlib
        import etna.api
        importlib.reload(etna.api)
        # Ensure the reloaded module uses our mock, not the real compiled extension
        etna.api._etna_rust = mock_etna_rust
        
        # Patch load_data and Preprocessor
        with patch.object(etna.api, 'load_data') as mock_load:
            mock_load.return_value = MagicMock()
            
            with patch.object(etna.api, 'Preprocessor') as mock_prep:
                mock_preprocessor = MagicMock()
                mock_preprocessor.fit_transform.return_value = ([[1, 2], [3, 4]], [[1], [0]])
                mock_preprocessor.output_dim = 2
                mock_prep.return_value = mock_preprocessor
                
                # Create model and train with progress bar
                model = etna.api.Model("dummy.csv", "target", task_type="classification")
                model.train(epochs=5, lr=0.01)
                
                # Verify train was called only ONCE (all epochs in Rust)
                assert mock_model.train.call_count == 1, f"Expected 1 train call, got {mock_model.train.call_count}"
                
                # Verify epochs parameter was passed correctly
                call_args = mock_model.train.call_args
                args, kwargs = call_args
                assert args[2] == 5, f"Expected epochs=5, got {args[2]}"
                # Verify early-stopping defaults were passed
                assert args[7] is False, "Expected early_stopping to default to False"
                assert args[8] == 10, "Expected default patience=10"
                assert args[9] is True, "Expected default restore_best=True"

                # Verify a callback was passed
                callback = args[10] if len(args) > 10 else kwargs.get('progress_callback')
                assert callback is not None, "Expected progress_callback to be passed"
                
                # Verify loss history was populated
                assert len(model.loss_history) == 5, f"Expected 5 loss entries, got {len(model.loss_history)}"
                
                print("✅ tqdm callback-based integration test PASSED!")
                print(f"   - train() called {mock_model.train.call_count} time (single Rust call)")
                print(f"   - loss_history has {len(model.loss_history)} entries")
                print("   - Progress callback passed to Rust ✓")

if __name__ == "__main__":
    test_tqdm_progress_bar()

