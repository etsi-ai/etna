"""
Integration example showing how ETNA metrics work with the Rust neural network core.

This example demonstrates the complete pipeline from training a model
with the Rust backend to evaluating it with Python metrics.
"""

import sys
import os

# Add the parent directory to the path to import etna modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from etna.metrics import (
        ClassificationMetrics,
        CrossEntropyLoss,
        accuracy_score,
        precision_recall_f1_score
    )
    print("Metrics module imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Note: This example requires numpy. Install with: pip install numpy")


def simulate_rust_neural_network():
    """
    Simulate the behavior of the Rust neural network implementation.
    
    This function mimics what the Rust core would do:
    1. Forward pass through network
    2. Generate predictions
    3. Calculate probabilities
    """
    
    # Simulate training data (like what would be passed to Rust train function)
    print("1. Simulating Rust Neural Network Training...")
    
    # Input features (would be passed as PyList to Rust)
    X_train = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7],
        [0.8, 0.9, 1.0],
    ]
    
    # True labels (would be converted to one-hot in Rust)
    y_train = [0, 1, 1, 0, 1, 0]  # Binary classification
    
    print(f"Training samples: {len(X_train)}")
    print(f"Features per sample: {len(X_train[0])}")
    print(f"True labels: {y_train}")
    
    # Simulate what the Rust neural network would do:
    # 1. Convert labels to one-hot (as done in lib.rs)
    num_classes = max(y_train) + 1
    y_train_one_hot = CrossEntropyLoss.one_hot_encode(y_train, num_classes)
    
    print(f"One-hot encoded labels:")
    for i, one_hot in enumerate(y_train_one_hot):
        print(f"  Sample {i}: {one_hot} (class {y_train[i]})")
    
    # 2. Simulate forward pass results (what would come from SimpleNN.forward)
    # These would be the softmax probabilities from the network
    y_pred_probs = [
        [0.8, 0.2],  # Confident class 0
        [0.3, 0.7],  # Confident class 1  
        [0.2, 0.8],  # Confident class 1
        [0.9, 0.1],  # Confident class 0
        [0.4, 0.6],  # Somewhat confident class 1
        [0.7, 0.3],  # Confident class 0
    ]
    
    print(f"\nPredicted probabilities (from softmax):")
    for i, probs in enumerate(y_pred_probs):
        print(f"  Sample {i}: {probs}")
    
    # 3. Convert probabilities to class predictions (as done in SimpleNN.predict)
    y_pred = [np.argmax(probs) for probs in y_pred_probs]
    print(f"\nPredicted classes: {y_pred}")
    
    # 4. Calculate cross-entropy loss (as done in Rust loss_function.rs)
    loss = CrossEntropyLoss.calculate(y_train_one_hot, y_pred_probs)
    print(f"Cross-entropy loss: {loss:.4f}")
    
    return y_train, y_pred, y_pred_probs, loss


def evaluate_model_performance(y_true, y_pred, y_pred_probs):
    """
    Evaluate the model using comprehensive metrics.
    
    This demonstrates how to use the metrics module to evaluate
    the performance of the Rust neural network.
    """
    
    print("\n2. Evaluating Model Performance with ETNA Metrics...")
    
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    # Detailed metrics using ClassificationMetrics
    classifier = ClassificationMetrics(average='binary')
    metrics = classifier.precision_recall_f1(y_true, y_pred)
    
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    
    # Confusion matrix
    cm = classifier.confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"True\\Pred  0    1")
    print(f"    0    {cm[0,0]:2d}   {cm[0,1]:2d}")
    print(f"    1    {cm[1,0]:2d}   {cm[1,1]:2d}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    report = classifier.classification_report(y_true, y_pred)
    print(report)
    
    return accuracy, metrics


def demonstrate_training_loop():
    """
    Demonstrate how metrics would be used during training.
    
    This simulates the monitoring that would happen during
    the training process in the Rust neural network.
    """
    
    print("\n3. Simulating Training Loop with Metrics Monitoring...")
    
    # Simulate multiple epochs of training
    epochs = 5
    
    # Generate training and validation data
    np.random.seed(42)
    
    # Training data
    n_train = 50
    y_true_train = np.random.randint(0, 2, n_train)
    
    # Validation data  
    n_val = 20
    y_true_val = np.random.randint(0, 2, n_val)
    
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 30)
        
        # Simulate improving predictions over epochs
        # Training predictions (start poor, improve over time)
        train_noise = 0.5 - (epoch * 0.08)  # Reduce noise each epoch
        y_pred_train = y_true_train.copy()
        n_errors = int(train_noise * n_train)
        error_indices = np.random.choice(n_train, n_errors, replace=False)
        y_pred_train[error_indices] = 1 - y_pred_train[error_indices]
        
        # Validation predictions (similar but typically worse)
        val_noise = 0.6 - (epoch * 0.06)
        y_pred_val = y_true_val.copy()
        n_val_errors = int(val_noise * n_val)
        val_error_indices = np.random.choice(n_val, n_val_errors, replace=False)
        y_pred_val[val_error_indices] = 1 - y_pred_val[val_error_indices]
        
        # Calculate metrics
        train_acc = accuracy_score(y_true_train, y_pred_train)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        
        # Simulate loss values (decreasing over time)
        train_loss = 2.0 - (epoch * 0.3) + np.random.normal(0, 0.1)
        val_loss = 2.2 - (epoch * 0.25) + np.random.normal(0, 0.15)
        
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.3f}")
        print(f"Val Loss:   {val_loss:.4f} - Val Acc:   {val_acc:.3f}")
        
        # Check for overfitting
        if epoch > 1 and val_acc < val_accuracies[-2]:
            print("⚠️  Validation accuracy decreased!")
        
        # Early stopping check
        if val_acc > 0.9:
            print("✅ Early stopping - validation accuracy target reached!")
            break
    
    # Final training summary
    print(f"\n4. Training Summary:")
    print(f"Final train accuracy: {train_accuracies[-1]:.3f}")
    print(f"Final validation accuracy: {val_accuracies[-1]:.3f}")
    print(f"Best validation accuracy: {max(val_accuracies):.3f}")
    
    # Check for overfitting
    acc_diff = train_accuracies[-1] - val_accuracies[-1]
    if acc_diff > 0.1:
        print(f"⚠️  Potential overfitting detected! (diff: {acc_diff:.3f})")
    else:
        print("✅ Model generalizes well!")
    
    return train_accuracies, val_accuracies


def test_set_evaluation():
    """
    Demonstrate final model evaluation on test set.
    
    This shows how to perform final evaluation after training
    is complete, mimicking real-world usage.
    """
    
    print("\n5. Final Test Set Evaluation...")
    
    # Simulate test data (unseen during training)
    np.random.seed(123)  # Different seed for test data
    n_test = 30
    
    y_true_test = np.random.randint(0, 2, n_test)
    
    # Simulate final model predictions (good but not perfect)
    y_pred_test = y_true_test.copy()
    n_errors = int(0.2 * n_test)  # 20% error rate
    error_indices = np.random.choice(n_test, n_errors, replace=False)
    y_pred_test[error_indices] = 1 - y_pred_test[error_indices]
    
    print(f"Test samples: {n_test}")
    print(f"True labels: {y_true_test[:10]}... (showing first 10)")
    print(f"Predictions: {y_pred_test[:10]}... (showing first 10)")
    
    # Comprehensive evaluation
    classifier = ClassificationMetrics(average='binary')
    
    test_accuracy = classifier.accuracy(y_true_test, y_pred_test)
    test_metrics = classifier.precision_recall_f1(y_true_test, y_pred_test)
    
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_accuracy:.3f}")
    print(f"Precision: {test_metrics['precision']:.3f}")
    print(f"Recall: {test_metrics['recall']:.3f}")
    print(f"F1-Score: {test_metrics['f1']:.3f}")
    
    # Detailed report
    print("\nDetailed Test Set Report:")
    print(classifier.classification_report(y_true_test, y_pred_test))
    
    return test_accuracy, test_metrics


def main():
    """Main integration example."""
    
    print("ETNA Integration Example: Rust Neural Network + Python Metrics")
    print("=" * 70)
    
    try:
        # Step 1: Simulate Rust neural network
        y_true, y_pred, y_pred_probs, loss = simulate_rust_neural_network()
        
        # Step 2: Evaluate with metrics
        accuracy, metrics = evaluate_model_performance(y_true, y_pred, y_pred_probs)
        
        # Step 3: Training loop simulation
        train_accs, val_accs = demonstrate_training_loop()
        
        # Step 4: Final test evaluation
        test_acc, test_metrics = test_set_evaluation()
        
        # Summary
        print("\n" + "="*70)
        print("INTEGRATION EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"✅ Cross-entropy loss calculation: {loss:.4f}")
        print(f"✅ Training simulation: {len(train_accs)} epochs")
        print(f"✅ Final test accuracy: {test_acc:.3f}")
        print(f"✅ All metrics computed successfully!")
        
        print(f"\nKey Features Demonstrated:")
        print(f"• Cross-entropy loss (mirrors Rust implementation)")
        print(f"• One-hot encoding (compatible with Rust)")
        print(f"• Classification metrics (accuracy, precision, recall, F1)")
        print(f"• Training monitoring and validation")
        print(f"• Comprehensive evaluation reports")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error in integration example: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
