#!/usr/bin/env python3
"""
Demo script to test the metrics.py module
"""

import numpy as np
import sys
import os

# Add the etna directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'etna'))

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

def test_classification_metrics():
    """Test classification metrics"""
    print("=== Testing Classification Metrics ===")
    
    # Binary classification example
    y_true_binary = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
    y_pred_binary = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
    
    print("\n1. Binary Classification:")
    print(f"True labels:      {y_true_binary}")
    print(f"Predicted labels: {y_pred_binary}")
    
    # Using the class
    clf_metrics = ClassificationMetrics(average='binary')
    accuracy = clf_metrics.accuracy(y_true_binary, y_pred_binary)
    metrics = clf_metrics.precision_recall_f1(y_true_binary, y_pred_binary)
    cm = clf_metrics.confusion_matrix(y_true_binary, y_pred_binary)
    
    print(f"\nAccuracy: {accuracy:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Multi-class classification example
    y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
    y_pred_multi = [0, 1, 1, 0, 2, 2, 0, 1, 2, 0]
    
    print("\n2. Multi-class Classification:")
    print(f"True labels:      {y_true_multi}")
    print(f"Predicted labels: {y_pred_multi}")
    
    clf_metrics_multi = ClassificationMetrics(average='macro')
    accuracy_multi = clf_metrics_multi.accuracy(y_true_multi, y_pred_multi)
    metrics_multi = clf_metrics_multi.precision_recall_f1(y_true_multi, y_pred_multi)
    cm_multi = clf_metrics_multi.confusion_matrix(y_true_multi, y_pred_multi)
    
    print(f"\nAccuracy: {accuracy_multi:.3f}")
    print(f"Macro Precision: {metrics_multi['precision']:.3f}")
    print(f"Macro Recall: {metrics_multi['recall']:.3f}")
    print(f"Macro F1-Score: {metrics_multi['f1']:.3f}")
    print(f"Confusion Matrix:\n{cm_multi}")
    
    # Classification report
    print("\n3. Classification Report:")
    report = clf_metrics_multi.classification_report(y_true_multi, y_pred_multi)
    print(report)

def test_regression_metrics():
    """Test regression metrics"""
    print("\n=== Testing Regression Metrics ===")
    
    y_true_reg = [3.0, -0.5, 2.0, 7.0, 4.2, 1.5, -1.0, 3.5, 2.8, 5.0]
    y_pred_reg = [2.5, 0.0, 2.1, 7.5, 3.8, 1.2, -0.8, 3.2, 3.0, 4.8]
    
    print(f"True values:      {y_true_reg}")
    print(f"Predicted values: {y_pred_reg}")
    
    reg_metrics = RegressionMetrics()
    
    mse = reg_metrics.mean_squared_error(y_true_reg, y_pred_reg)
    rmse = reg_metrics.root_mean_squared_error(y_true_reg, y_pred_reg)
    mae = reg_metrics.mean_absolute_error(y_true_reg, y_pred_reg)
    r2 = reg_metrics.r_squared(y_true_reg, y_pred_reg)
    mape = reg_metrics.mean_absolute_percentage_error(y_true_reg, y_pred_reg)
    
    print(f"\nMSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")
    print(f"MAPE: {mape:.3f}%")

def test_cross_entropy_loss():
    """Test cross-entropy loss"""
    print("\n=== Testing Cross-Entropy Loss ===")
    
    # Example with 3 classes
    y_true_labels = [0, 1, 2, 0, 1]
    y_pred_probs = [
        [0.9, 0.05, 0.05],  # Predicted for class 0
        [0.1, 0.8, 0.1],    # Predicted for class 1
        [0.2, 0.3, 0.5],    # Predicted for class 2
        [0.8, 0.1, 0.1],    # Predicted for class 0
        [0.05, 0.9, 0.05]   # Predicted for class 1
    ]
    
    # Convert labels to one-hot
    y_true_onehot = CrossEntropyLoss.one_hot_encode(y_true_labels, 3)
    
    print(f"True labels: {y_true_labels}")
    print(f"True one-hot:\n{y_true_onehot}")
    print(f"Predicted probabilities:\n{np.array(y_pred_probs)}")
    
    loss = CrossEntropyLoss.calculate(y_true_onehot, y_pred_probs)
    print(f"\nCross-entropy loss: {loss:.3f}")

def test_convenience_functions():
    """Test convenience functions"""
    print("\n=== Testing Convenience Functions ===")
    
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]
    
    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    
    acc = accuracy_score(y_true, y_pred)
    metrics = precision_recall_f1_score(y_true, y_pred, average='binary')
    cm = confusion_matrix_score(y_true, y_pred)
    
    print(f"\nAccuracy (function): {acc:.3f}")
    print(f"Metrics (function): {metrics}")
    print(f"Confusion Matrix (function):\n{cm}")
    
    # Regression convenience functions
    y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred_reg = [1.1, 2.1, 2.9, 3.8, 5.2]
    
    print(f"\nTrue regression values:      {y_true_reg}")
    print(f"Predicted regression values: {y_pred_reg}")
    
    mse = mean_squared_error_score(y_true_reg, y_pred_reg)
    r2 = r2_score(y_true_reg, y_pred_reg)
    
    print(f"MSE (function): {mse:.3f}")
    print(f"R² (function):  {r2:.3f}")

def main():
    """Run all tests"""
    print("ETNA Metrics Module Test Suite")
    print("=" * 50)
    
    try:
        test_classification_metrics()
        test_regression_metrics()
        test_cross_entropy_loss()
        test_convenience_functions()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully! ✅")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
