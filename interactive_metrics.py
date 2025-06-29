#!/usr/bin/env python3
"""
Interactive metrics test script
"""

# Import the metrics module
import sys
import os
sys.path.insert(0, 'etna')

from etna.metrics import ClassificationMetrics, RegressionMetrics

print("=== ETNA Metrics Interactive Demo ===")
print("You can now use the metrics classes:")
print()

# Example usage
clf = ClassificationMetrics()
reg = RegressionMetrics()

# Test data
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

print("Classification Example:")
print(f"y_true = {y_true}")
print(f"y_pred = {y_pred}")
print(f"Accuracy = {clf.accuracy(y_true, y_pred):.3f}")

# Regression example
y_true_reg = [1.0, 2.0, 3.0, 4.0, 5.0]
y_pred_reg = [1.1, 2.1, 2.9, 3.8, 5.2]

print("\nRegression Example:")
print(f"y_true = {y_true_reg}")
print(f"y_pred = {y_pred_reg}")
print(f"MSE = {reg.mean_squared_error(y_true_reg, y_pred_reg):.3f}")
print(f"R² = {reg.r_squared(y_true_reg, y_pred_reg):.3f}")

print("\n" + "="*50)
print("Available classes and methods:")
print("- ClassificationMetrics: accuracy, confusion_matrix, precision_recall_f1, classification_report")
print("- RegressionMetrics: mean_squared_error, root_mean_squared_error, mean_absolute_error, r_squared, mean_absolute_percentage_error")
print("- CrossEntropyLoss: calculate, one_hot_encode")
