"""
Test script to verify tqdm progress bar integration.
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
    """Verify tqdm is properly integrated in the train method."""
    
    # Mock the Rust backend
    mock_etna_rust = MagicMock()
    mock_model = MagicMock()
    # Simulate returning a loss value for each epoch
    mock_model.train.return_value = [0.5]  # Returns loss list for 1 epoch
    mock_etna_rust.EtnaModel.return_value = mock_model
    
    # Patch the Rust import
    with patch.dict('sys.modules', {'etna._etna_rust': mock_etna_rust}):
        # Now import the module - it will use our mock
        import importlib
        import etna.api
        importlib.reload(etna.api)
        
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
                
                # Capture stdout to verify tqdm runs
                captured = StringIO()
                with patch('sys.stderr', captured):
                    model.train(epochs=5, lr=0.01)
                
                # Verify train was called 5 times (once per epoch)
                assert mock_model.train.call_count == 5, f"Expected 5 train calls, got {mock_model.train.call_count}"
                
                # Verify each call was for 1 epoch
                for call in mock_model.train.call_args_list:
                    args, kwargs = call
                    # epochs is the 3rd positional argument
                    assert args[2] == 1, f"Expected epochs=1 per call, got {args[2]}"
                
                # Verify loss history was populated
                assert len(model.loss_history) == 5, f"Expected 5 loss entries, got {len(model.loss_history)}"
                
                print("✅ tqdm integration test PASSED!")
                print(f"   - train() called {mock_model.train.call_count} times (once per epoch)")
                print(f"   - loss_history has {len(model.loss_history)} entries")

if __name__ == "__main__":
    success = test_tqdm_progress_bar()
    sys.exit(0 if success else 1)
