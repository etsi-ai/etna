"""
Test script for validation split and MLflow tracking feature.
This script tests the validation split functionality without requiring MLflow server.
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np

# Set environment variable to disable MLflow
os.environ["ETNA_DISABLE_MLFLOW"] = "1"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from etna import Model
    print("✅ Successfully imported Model")
except ImportError as e:
    print(f"❌ Failed to import Model: {e}")
    print("Note: This test requires the Rust extension to be built.")
    sys.exit(1)


def create_test_data_classification(n_samples=100):
    """Create a simple classification dataset."""
    np.random.seed(42)
    X1 = np.random.randn(n_samples, 2)
    y = (X1[:, 0] + X1[:, 1] > 0).astype(int)
    
    df = pd.DataFrame({
        'feature1': X1[:, 0],
        'feature2': X1[:, 1],
        'target': y
    })
    return df


def create_test_data_regression(n_samples=100):
    """Create a simple regression dataset."""
    np.random.seed(42)
    X1 = np.random.randn(n_samples, 2)
    y = X1[:, 0] * 2 + X1[:, 1] * 3 + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame({
        'feature1': X1[:, 0],
        'feature2': X1[:, 1],
        'target': y
    })
    return df


def test_validation_split_classification():
    """Test validation split for classification task."""
    print("\n" + "="*60)
    print("Testing Validation Split - Classification")
    print("="*60)
    
    # Create test data
    df = create_test_data_classification(100)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create model
        model = Model(file_path=temp_file, target='target', task_type='classification')
        
        # Test with validation split
        print("\n1. Testing with validation_split=0.2")
        model.train(epochs=10, lr=0.01, validation_split=0.2)
        
        # Check that loss histories are populated
        assert len(model.loss_history) == 10, f"Expected 10 training losses, got {len(model.loss_history)}"
        assert len(model.val_loss_history) == 10, f"Expected 10 validation losses, got {len(model.val_loss_history)}"
        print(f"   ✅ Training losses: {len(model.loss_history)}")
        print(f"   ✅ Validation losses: {len(model.val_loss_history)}")
        
        # Check that validation losses are reasonable
        assert all(isinstance(loss, (int, float)) for loss in model.val_loss_history), "Validation losses should be numeric"
        assert all(loss >= 0 for loss in model.val_loss_history), "Validation losses should be non-negative"
        print(f"   ✅ Validation losses are valid (range: {min(model.val_loss_history):.4f} - {max(model.val_loss_history):.4f})")
        
        # Test without validation split
        print("\n2. Testing with validation_split=0.0")
        model2 = Model(file_path=temp_file, target='target', task_type='classification')
        model2.train(epochs=5, lr=0.01, validation_split=0.0)
        
        assert len(model2.loss_history) == 5, f"Expected 5 training losses, got {len(model2.loss_history)}"
        assert len(model2.val_loss_history) == 0, f"Expected 0 validation losses, got {len(model2.val_loss_history)}"
        print(f"   ✅ Training losses: {len(model2.loss_history)}")
        print(f"   ✅ Validation losses: {len(model2.val_loss_history)} (disabled)")
        
        print("\n✅ Classification validation split test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Classification validation split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_validation_split_regression():
    """Test validation split for regression task."""
    print("\n" + "="*60)
    print("Testing Validation Split - Regression")
    print("="*60)
    
    # Create test data
    df = create_test_data_regression(100)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create model
        model = Model(file_path=temp_file, target='target', task_type='regression')
        
        # Test with validation split
        print("\n1. Testing with validation_split=0.2")
        model.train(epochs=10, lr=0.01, validation_split=0.2)
        
        # Check that loss histories are populated
        assert len(model.loss_history) == 10, f"Expected 10 training losses, got {len(model.loss_history)}"
        assert len(model.val_loss_history) == 10, f"Expected 10 validation losses, got {len(model.val_loss_history)}"
        print(f"   ✅ Training losses: {len(model.loss_history)}")
        print(f"   ✅ Validation losses: {len(model.val_loss_history)}")
        
        # Check that validation losses are reasonable
        assert all(isinstance(loss, (int, float)) for loss in model.val_loss_history), "Validation losses should be numeric"
        assert all(loss >= 0 for loss in model.val_loss_history), "Validation losses should be non-negative"
        print(f"   ✅ Validation losses are valid (range: {min(model.val_loss_history):.4f} - {max(model.val_loss_history):.4f})")
        
        print("\n✅ Regression validation split test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Regression validation split test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


def test_validation_split_edge_cases():
    """Test edge cases for validation split."""
    print("\n" + "="*60)
    print("Testing Validation Split - Edge Cases")
    print("="*60)
    
    # Create test data
    df = create_test_data_classification(50)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        model = Model(file_path=temp_file, target='target', task_type='classification')
        
        # Test invalid validation_split values
        print("\n1. Testing invalid validation_split values")
        try:
            model.train(epochs=5, validation_split=1.0)
            print("   ❌ Should have raised ValueError for validation_split=1.0")
            return False
        except ValueError:
            print("   ✅ Correctly raised ValueError for validation_split=1.0")
        
        try:
            model.train(epochs=5, validation_split=-0.1)
            print("   ❌ Should have raised ValueError for validation_split=-0.1")
            return False
        except ValueError:
            print("   ✅ Correctly raised ValueError for validation_split=-0.1")
        
        # Test very small validation split
        print("\n2. Testing very small validation_split=0.05")
        model.train(epochs=5, validation_split=0.05)
        assert len(model.val_loss_history) == 5, "Should have validation losses"
        print(f"   ✅ Validation losses: {len(model.val_loss_history)}")
        
        # Test large validation split
        print("\n3. Testing large validation_split=0.5")
        model.train(epochs=5, validation_split=0.5)
        assert len(model.val_loss_history) == 5, "Should have validation losses"
        print(f"   ✅ Validation losses: {len(model.val_loss_history)}")
        
        print("\n✅ Edge cases test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Edge cases test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)


if __name__ == "__main__":
    print("="*60)
    print("Validation Split & MLflow Tracking Test Suite")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Classification", test_validation_split_classification()))
    results.append(("Regression", test_validation_split_regression()))
    results.append(("Edge Cases", test_validation_split_edge_cases()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed!")
        sys.exit(1)

