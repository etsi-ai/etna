"""
Example demonstrating the usage of ETNA metrics module.

This example shows how to use various evaluation metrics for both
classification and regression tasks.
"""

import sys
import os

# Add the parent directory to the path to import etna modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    from etna.metrics import (
        ClassificationMetrics,
        RegressionMetrics,
        CrossEntropyLoss,
        accuracy_score,
        precision_recall_f1_score,
        confusion_matrix_score,
        mean_squared_error_score,
        r2_score
    )
    print("All modules imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required dependencies: pip install numpy")
    sys.exit(1)


def classification_example():
    """Demonstrate classification metrics."""
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS EXAMPLE")
    print("="*50)
    
    # Simulate predictions from a neural network
    print("\n1. Binary Classification Example:")
    y_true_binary = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    y_pred_binary = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
    
    print(f"True labels:      {y_true_binary}")
    print(f"Predicted labels: {y_pred_binary}")
    
    # Calculate metrics using convenience functions
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    print(f"\nAccuracy: {accuracy:.3f}")
    
    # Calculate precision, recall, F1
    metrics = precision_recall_f1_score(y_true_binary, y_pred_binary, average='binary')
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix_score(y_true_binary, y_pred_binary)
    print(f"\nConfusion Matrix:")
    print(f"[[{cm[0,0]}, {cm[0,1]}]]  (TN, FP)")
    print(f"[[{cm[1,0]}, {cm[1,1]}]]  (FN, TP)")
    
    # Multi-class classification
    print("\n2. Multi-class Classification Example:")
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred_multi = [0, 1, 2, 0, 2, 1, 0, 1, 1, 0]
    
    print(f"True labels:      {y_true_multi}")
    print(f"Predicted labels: {y_pred_multi}")
    
    # Using ClassificationMetrics class for more options
    classifier = ClassificationMetrics(average='macro')
    
    accuracy_multi = classifier.accuracy(y_true_multi, y_pred_multi)
    print(f"\nAccuracy: {accuracy_multi:.3f}")
    
    # Macro-averaged metrics
    macro_metrics = classifier.precision_recall_f1(y_true_multi, y_pred_multi)
    print(f"Macro Precision: {macro_metrics['precision']:.3f}")
    print(f"Macro Recall: {macro_metrics['recall']:.3f}")
    print(f"Macro F1-Score: {macro_metrics['f1']:.3f}")
    
    # Micro-averaged metrics
    classifier_micro = ClassificationMetrics(average='micro')
    micro_metrics = classifier_micro.precision_recall_f1(y_true_multi, y_pred_multi)
    print(f"Micro Precision: {micro_metrics['precision']:.3f}")
    print(f"Micro Recall: {micro_metrics['recall']:.3f}")
    print(f"Micro F1-Score: {micro_metrics['f1']:.3f}")
    
    # Full classification report
    print("\n3. Classification Report:")
    report = classifier.classification_report(y_true_multi, y_pred_multi)
    print(report)


def regression_example():
    """Demonstrate regression metrics."""
    print("\n" + "="*50)
    print("REGRESSION METRICS EXAMPLE")
    print("="*50)
    
    # Simulate regression predictions
    y_true = [3.0, -0.5, 2.0, 7.0, 1.5, 4.2, -1.0, 5.5]
    y_pred = [2.5, 0.0, 2.0, 8.0, 1.2, 4.5, -0.8, 5.2]
    
    print(f"True values:      {y_true}")
    print(f"Predicted values: {y_pred}")
    
    # Using convenience functions
    mse = mean_squared_error_score(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\nMSE: {mse:.3f}")
    print(f"R²: {r2:.3f}")
    
    # Using RegressionMetrics class for more metrics
    regressor = RegressionMetrics()
    
    rmse = regressor.root_mean_squared_error(y_true, y_pred)
    mae = regressor.mean_absolute_error(y_true, y_pred)
    
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    
    # MAPE (using positive values only)
    y_true_pos = [3.0, 2.0, 7.0, 1.5, 4.2, 5.5]
    y_pred_pos = [2.5, 2.0, 8.0, 1.2, 4.5, 5.2]
    
    mape = regressor.mean_absolute_percentage_error(y_true_pos, y_pred_pos)
    print(f"MAPE: {mape:.3f}%")


def cross_entropy_example():
    """Demonstrate cross-entropy loss calculation."""
    print("\n" + "="*50)
    print("CROSS-ENTROPY LOSS EXAMPLE")
    print("="*50)
    
    # Simulate neural network output (softmax probabilities)
    y_pred_probs = [
        [0.8, 0.1, 0.1],  # Confident about class 0
        [0.2, 0.7, 0.1],  # Confident about class 1
        [0.1, 0.2, 0.7],  # Confident about class 2
        [0.6, 0.3, 0.1],  # Less confident about class 0
    ]
    
    # True labels (will be converted to one-hot)
    y_true_labels = [0, 1, 2, 0]
    
    print(f"True labels: {y_true_labels}")
    print("Predicted probabilities:")
    for i, probs in enumerate(y_pred_probs):
        print(f"  Sample {i}: {probs}")
    
    # Convert to one-hot encoding
    num_classes = 3
    y_true_one_hot = CrossEntropyLoss.one_hot_encode(y_true_labels, num_classes)
    
    print("\nOne-hot encoded true labels:")
    for i, one_hot in enumerate(y_true_one_hot):
        print(f"  Sample {i}: {one_hot}")
    
    # Calculate cross-entropy loss
    loss = CrossEntropyLoss.calculate(y_true_one_hot, y_pred_probs)
    print(f"\nCross-entropy loss: {loss:.4f}")
    
    # Compare with perfect predictions
    y_pred_perfect = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0], 
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    
    loss_perfect = CrossEntropyLoss.calculate(y_true_one_hot, y_pred_perfect)
    print(f"Perfect predictions loss: {loss_perfect:.6f}")


def neural_network_simulation():
    """Simulate a complete neural network evaluation pipeline."""
    print("\n" + "="*50)
    print("NEURAL NETWORK EVALUATION SIMULATION")
    print("="*50)
    
    # Simulate training data results
    print("\n1. Training Set Evaluation:")
    
    # Generate synthetic training results
    np.random.seed(42)  # For reproducible results
    n_samples = 100
    n_classes = 3
    
    # Generate true labels
    y_true_train = np.random.randint(0, n_classes, n_samples)
    
    # Generate predictions with some noise (simulating a good but not perfect model)
    y_pred_train = y_true_train.copy()
    # Add some incorrect predictions
    noise_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
    y_pred_train[noise_indices] = np.random.randint(0, n_classes, len(noise_indices))
    
    # Evaluate training performance
    train_classifier = ClassificationMetrics(average='macro')
    train_accuracy = train_classifier.accuracy(y_true_train, y_pred_train)
    train_metrics = train_classifier.precision_recall_f1(y_true_train, y_pred_train)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Training Precision: {train_metrics['precision']:.3f}")
    print(f"Training Recall: {train_metrics['recall']:.3f}")
    print(f"Training F1-Score: {train_metrics['f1']:.3f}")
    
    # Simulate test data results (typically lower performance)
    print("\n2. Test Set Evaluation:")
    
    n_test_samples = 30
    y_true_test = np.random.randint(0, n_classes, n_test_samples)
    y_pred_test = y_true_test.copy()
    # Add more noise for test set (simulating overfitting)
    noise_indices_test = np.random.choice(n_test_samples, size=int(0.25 * n_test_samples), replace=False)
    y_pred_test[noise_indices_test] = np.random.randint(0, n_classes, len(noise_indices_test))
    
    test_classifier = ClassificationMetrics(average='macro')
    test_accuracy = test_classifier.accuracy(y_true_test, y_pred_test)
    test_metrics = test_classifier.precision_recall_f1(y_true_test, y_pred_test)
    
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"Test Precision: {test_metrics['precision']:.3f}")
    print(f"Test Recall: {test_metrics['recall']:.3f}")
    print(f"Test F1-Score: {test_metrics['f1']:.3f}")
    
    # Performance comparison
    print(f"\n3. Performance Analysis:")
    print(f"Accuracy Drop: {train_accuracy - test_accuracy:.3f}")
    print(f"F1-Score Drop: {train_metrics['f1'] - test_metrics['f1']:.3f}")
    
    if (train_accuracy - test_accuracy) > 0.1:
        print("⚠️  Potential overfitting detected!")
    else:
        print("✅ Model generalizes well!")
    
    # Detailed test set report
    print(f"\n4. Detailed Test Set Report:")
    test_report = test_classifier.classification_report(y_true_test, y_pred_test)
    print(test_report)


def main():
    """Main function to run all examples."""
    print("ETNA Metrics Module - Examples and Demonstrations")
    print("=" * 60)
    
    try:
        # Run all examples
        classification_example()
        regression_example()
        cross_entropy_example()
        neural_network_simulation()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("You can now use these metrics to evaluate your ETNA models.")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Please ensure all dependencies are installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
