"""
Evaluation Metrics Module for ETNA

This module provides comprehensive evaluation metrics for machine learning models,
including classification and regression metrics. It's designed to work seamlessly
with the ETNA neural network framework implemented in Rust.

Supported metrics:
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC-AUC
- Regression: MSE, RMSE, MAE, R², MAPE
- Multi-class: Macro/Micro averages, Per-class metrics
"""

import numpy as np
from typing import List, Union, Tuple, Dict, Optional
from collections import Counter
import warnings


class MetricsCalculator:
    """
    Main class for calculating various evaluation metrics.
    Supports both binary and multi-class classification, as well as regression tasks.
    """
    
    def __init__(self, average: str = 'macro'):
        """
        Initialize the metrics calculator.
        
        Args:
            average (str): Type of averaging for multi-class metrics.
                          Options: 'macro', 'micro', 'weighted', 'binary'
        """
        self.average = average
        self._validate_average_parameter()
    
    def _validate_average_parameter(self):
        """Validate the average parameter."""
        valid_averages = ['macro', 'micro', 'weighted', 'binary', None]
        if self.average not in valid_averages:
            raise ValueError(f"Average must be one of {valid_averages}, got {self.average}")
    
    def _validate_inputs(self, y_true: Union[List, np.ndarray], 
                        y_pred: Union[List, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert inputs to numpy arrays.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Tuple of validated numpy arrays
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
        
        if len(y_true) == 0:
            raise ValueError("Empty arrays provided")
            
        return y_true, y_pred


class ClassificationMetrics(MetricsCalculator):
    """
    Classification metrics calculator.
    Implements standard classification evaluation metrics.
    """
    
    def accuracy(self, y_true: Union[List, np.ndarray], 
                 y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        return float(np.mean(y_true == y_pred))
    
    def confusion_matrix(self, y_true: Union[List, np.ndarray], 
                        y_pred: Union[List, np.ndarray]) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as numpy array
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        n_labels = len(labels)
        label_to_ind = {label: i for i, label in enumerate(labels)}
        
        confusion = np.zeros((n_labels, n_labels), dtype=int)
        
        for true_label, pred_label in zip(y_true, y_pred):
            confusion[label_to_ind[true_label], label_to_ind[pred_label]] += 1
            
        return confusion
    
    def precision_recall_f1(self, y_true: Union[List, np.ndarray], 
                           y_pred: Union[List, np.ndarray]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Get unique labels
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        if len(labels) == 2 and self.average == 'binary':
            # Binary classification
            return self._binary_precision_recall_f1(y_true, y_pred, labels[1])
        else:
            # Multi-class classification
            return self._multiclass_precision_recall_f1(y_true, y_pred, labels)
    
    def _binary_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   pos_label) -> Dict[str, float]:
        """Calculate binary classification metrics."""
        tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
        fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
        fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def _multiclass_precision_recall_f1(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       labels: np.ndarray) -> Dict[str, float]:
        """Calculate multi-class classification metrics."""
        precisions, recalls, f1s = [], [], []
        
        for label in labels:
            # One-vs-rest for each class
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            tp = np.sum(y_true_binary & y_pred_binary)
            fp = np.sum((1 - y_true_binary) & y_pred_binary)
            fn = np.sum(y_true_binary & (1 - y_pred_binary))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        if self.average == 'macro':
            return {
                'precision': float(np.mean(precisions)),
                'recall': float(np.mean(recalls)),
                'f1': float(np.mean(f1s))
            }
        elif self.average == 'micro':
            # Calculate global TP, FP, FN
            total_tp = total_fp = total_fn = 0
            for label in labels:
                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)
                
                total_tp += np.sum(y_true_binary & y_pred_binary)
                total_fp += np.sum((1 - y_true_binary) & y_pred_binary)
                total_fn += np.sum(y_true_binary & (1 - y_pred_binary))
            
            micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
            
            return {
                'precision': float(micro_precision),
                'recall': float(micro_recall),
                'f1': float(micro_f1)
            }
        elif self.average == 'weighted':
            # Weight by support (number of true instances for each label)
            weights = [np.sum(y_true == label) for label in labels]
            total_weight = sum(weights)
            
            if total_weight == 0:
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
            
            weighted_precision = sum(p * w for p, w in zip(precisions, weights)) / total_weight
            weighted_recall = sum(r * w for r, w in zip(recalls, weights)) / total_weight
            weighted_f1 = sum(f * w for f, w in zip(f1s, weights)) / total_weight
            
            return {
                'precision': float(weighted_precision),
                'recall': float(weighted_recall),
                'f1': float(weighted_f1)
            }
    
    def classification_report(self, y_true: Union[List, np.ndarray], 
                            y_pred: Union[List, np.ndarray]) -> str:
        """
        Generate a comprehensive classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Formatted string report
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
        report_lines = []
        report_lines.append("Classification Report")
        report_lines.append("=" * 50)
        report_lines.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        report_lines.append("-" * 50)
        
        # Per-class metrics
        total_support = len(y_true)
        for label in labels:
            # Calculate metrics for this class
            temp_calculator = ClassificationMetrics(average='binary')
            y_true_binary = (y_true == label)
            y_pred_binary = (y_pred == label)
            
            metrics = temp_calculator._binary_precision_recall_f1(
                y_true_binary.astype(int), y_pred_binary.astype(int), 1
            )
            
            support = np.sum(y_true == label)
            
            report_lines.append(
                f"{str(label):<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                f"{metrics['f1']:<10.3f} {support:<10d}"
            )
        
        # Overall metrics
        report_lines.append("-" * 50)
        accuracy = self.accuracy(y_true, y_pred)
        macro_metrics = ClassificationMetrics(average='macro').precision_recall_f1(y_true, y_pred)
        weighted_metrics = ClassificationMetrics(average='weighted').precision_recall_f1(y_true, y_pred)
        
        report_lines.append(f"{'Accuracy':<10} {'':<10} {'':<10} {accuracy:<10.3f} {total_support:<10d}")
        report_lines.append(
            f"{'Macro Avg':<10} {macro_metrics['precision']:<10.3f} {macro_metrics['recall']:<10.3f} "
            f"{macro_metrics['f1']:<10.3f} {total_support:<10d}"
        )
        report_lines.append(
            f"{'Weighted':<10} {weighted_metrics['precision']:<10.3f} {weighted_metrics['recall']:<10.3f} "
            f"{weighted_metrics['f1']:<10.3f} {total_support:<10d}"
        )
        
        return '\n'.join(report_lines)


class RegressionMetrics(MetricsCalculator):
    """
    Regression metrics calculator.
    Implements standard regression evaluation metrics.
    """
    
    def mean_squared_error(self, y_true: Union[List, np.ndarray], 
                          y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate Mean Squared Error (MSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MSE value
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        return float(np.mean((y_true - y_pred) ** 2))
    
    def root_mean_squared_error(self, y_true: Union[List, np.ndarray], 
                               y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate Root Mean Squared Error (RMSE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        mse = self.mean_squared_error(y_true, y_pred)
        return float(np.sqrt(mse))
    
    def mean_absolute_error(self, y_true: Union[List, np.ndarray], 
                           y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate Mean Absolute Error (MAE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))
    
    def r_squared(self, y_true: Union[List, np.ndarray], 
                  y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate R-squared (coefficient of determination).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² value
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0 if ss_res == 0 else float('-inf')
        
        return float(1 - (ss_res / ss_tot))
    
    def mean_absolute_percentage_error(self, y_true: Union[List, np.ndarray], 
                                     y_pred: Union[List, np.ndarray]) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value (as percentage)
        """
        y_true, y_pred = self._validate_inputs(y_true, y_pred)
        
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            warnings.warn("All true values are zero, MAPE is undefined")
            return float('inf')
        
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


class CrossEntropyLoss:
    """
    Cross-entropy loss function implementation.
    Mirrors the Rust implementation in the core library.
    """
    
    @staticmethod
    def calculate(y_true_probs: Union[List[List[float]], np.ndarray], 
                  y_pred_probs: Union[List[List[float]], np.ndarray]) -> float:
        """
        Calculate cross-entropy loss between predicted probabilities and true probabilities.
        
        Args:
            y_true_probs: True probability distributions (one-hot encoded)
            y_pred_probs: Predicted probability distributions
            
        Returns:
            Cross-entropy loss value
        """
        y_true_probs = np.array(y_true_probs)
        y_pred_probs = np.array(y_pred_probs)
        
        if y_true_probs.shape != y_pred_probs.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true_probs.shape} vs y_pred {y_pred_probs.shape}")
        
        # Add epsilon to prevent log(0)
        epsilon = 1e-9
        y_pred_probs = np.clip(y_pred_probs, epsilon, 1 - epsilon)
        
        # Calculate cross-entropy loss
        loss = -np.sum(y_true_probs * np.log(y_pred_probs), axis=1)
        return float(np.mean(loss))
    
    @staticmethod
    def one_hot_encode(labels: Union[List[int], np.ndarray], num_classes: int) -> np.ndarray:
        """
        Convert integer labels to one-hot encoded vectors.
        
        Args:
            labels: Integer labels
            num_classes: Number of classes
            
        Returns:
            One-hot encoded array
        """
        labels = np.array(labels)
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot


# Convenience functions for easy access
def accuracy_score(y_true: Union[List, np.ndarray], 
                   y_pred: Union[List, np.ndarray]) -> float:
    """Calculate accuracy score."""
    return ClassificationMetrics().accuracy(y_true, y_pred)


def precision_recall_f1_score(y_true: Union[List, np.ndarray], 
                              y_pred: Union[List, np.ndarray], 
                              average: str = 'macro') -> Dict[str, float]:
    """Calculate precision, recall, and F1 score."""
    return ClassificationMetrics(average=average).precision_recall_f1(y_true, y_pred)


def confusion_matrix_score(y_true: Union[List, np.ndarray], 
                          y_pred: Union[List, np.ndarray]) -> np.ndarray:
    """Calculate confusion matrix."""
    return ClassificationMetrics().confusion_matrix(y_true, y_pred)


def mean_squared_error_score(y_true: Union[List, np.ndarray], 
                            y_pred: Union[List, np.ndarray]) -> float:
    """Calculate mean squared error."""
    return RegressionMetrics().mean_squared_error(y_true, y_pred)


def r2_score(y_true: Union[List, np.ndarray], 
             y_pred: Union[List, np.ndarray]) -> float:
    """Calculate R-squared score."""
    return RegressionMetrics().r_squared(y_true, y_pred)


# Export main classes and functions
__all__ = [
    'MetricsCalculator',
    'ClassificationMetrics', 
    'RegressionMetrics',
    'CrossEntropyLoss',
    'accuracy_score',
    'precision_recall_f1_score',
    'confusion_matrix_score',
    'mean_squared_error_score',
    'r2_score'
]