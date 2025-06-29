"""
Unit tests for the metrics module.

This module tests all evaluation metrics implemented in the ETNA framework,
including classification and regression metrics.
"""

import pytest
import numpy as np
import sys
import os

# Add the parent directory to the path to import etna modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


class TestClassificationMetrics:
    """Test cases for classification metrics."""
    
    def setup_method(self):
        """Set up test data."""
        # Binary classification data
        self.y_true_binary = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        self.y_pred_binary = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        
        # Multi-class classification data
        self.y_true_multi = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        self.y_pred_multi = [0, 1, 2, 0, 2, 1, 0, 1, 1, 0]
        
        # Perfect predictions
        self.y_true_perfect = [0, 1, 1, 0, 1]
        self.y_pred_perfect = [0, 1, 1, 0, 1]
        
        self.classifier = ClassificationMetrics()
    
    def test_accuracy_binary(self):
        """Test accuracy calculation for binary classification."""
        accuracy = self.classifier.accuracy(self.y_true_binary, self.y_pred_binary)
        expected = 7 / 10  # 7 correct predictions out of 10
        assert abs(accuracy - expected) < 1e-6
    
    def test_accuracy_perfect(self):
        """Test accuracy with perfect predictions."""
        accuracy = self.classifier.accuracy(self.y_true_perfect, self.y_pred_perfect)
        assert accuracy == 1.0
    
    def test_accuracy_all_wrong(self):
        """Test accuracy with all wrong predictions."""
        y_true = [0, 0, 0, 0]
        y_pred = [1, 1, 1, 1]
        accuracy = self.classifier.accuracy(y_true, y_pred)
        assert accuracy == 0.0
    
    def test_confusion_matrix_binary(self):
        """Test confusion matrix for binary classification."""
        cm = self.classifier.confusion_matrix(self.y_true_binary, self.y_pred_binary)
        expected = np.array([[3, 1], [2, 4]])  # [[TN, FP], [FN, TP]]
        np.testing.assert_array_equal(cm, expected)
    
    def test_confusion_matrix_multi(self):
        """Test confusion matrix for multi-class classification."""
        cm = self.classifier.confusion_matrix(self.y_true_multi, self.y_pred_multi)
        # Expected: class 0: 4 samples, class 1: 3 samples, class 2: 3 samples
        assert cm.shape == (3, 3)
        assert np.sum(cm) == len(self.y_true_multi)
    
    def test_precision_recall_f1_binary(self):
        """Test precision, recall, and F1 for binary classification."""
        classifier_binary = ClassificationMetrics(average='binary')
        metrics = classifier_binary.precision_recall_f1(self.y_true_binary, self.y_pred_binary)
        
        # Manual calculation for positive class (1)
        # TP=4, FP=1, FN=2
        expected_precision = 4 / (4 + 1)  # 0.8
        expected_recall = 4 / (4 + 2)  # 0.667
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        
        assert abs(metrics['precision'] - expected_precision) < 1e-6
        assert abs(metrics['recall'] - expected_recall) < 1e-6
        assert abs(metrics['f1'] - expected_f1) < 1e-6
    
    def test_precision_recall_f1_macro(self):
        """Test macro-averaged precision, recall, and F1."""
        classifier_macro = ClassificationMetrics(average='macro')
        metrics = classifier_macro.precision_recall_f1(self.y_true_multi, self.y_pred_multi)
        
        # All metrics should be between 0 and 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    def test_precision_recall_f1_micro(self):
        """Test micro-averaged precision, recall, and F1."""
        classifier_micro = ClassificationMetrics(average='micro')
        metrics = classifier_micro.precision_recall_f1(self.y_true_multi, self.y_pred_multi)
        
        # For micro averaging, precision and recall should be equal to accuracy
        accuracy = classifier_micro.accuracy(self.y_true_multi, self.y_pred_multi)
        assert abs(metrics['precision'] - accuracy) < 1e-6
        assert abs(metrics['recall'] - accuracy) < 1e-6
    
    def test_classification_report(self):
        """Test classification report generation."""
        report = self.classifier.classification_report(self.y_true_binary, self.y_pred_binary)
        
        # Check that report contains expected sections
        assert "Classification Report" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1-Score" in report
        assert "Accuracy" in report
        assert "Macro Avg" in report
    
    def test_empty_inputs(self):
        """Test behavior with empty inputs."""
        with pytest.raises(ValueError):
            self.classifier.accuracy([], [])
    
    def test_mismatched_shapes(self):
        """Test behavior with mismatched input shapes."""
        with pytest.raises(ValueError):
            self.classifier.accuracy([0, 1, 2], [0, 1])
    
    def test_invalid_average_parameter(self):
        """Test invalid average parameter."""
        with pytest.raises(ValueError):
            ClassificationMetrics(average='invalid')


class TestRegressionMetrics:
    """Test cases for regression metrics."""
    
    def setup_method(self):
        """Set up test data."""
        self.y_true = [3.0, -0.5, 2.0, 7.0]
        self.y_pred = [2.5, 0.0, 2.0, 8.0]
        
        # Perfect predictions
        self.y_true_perfect = [1.0, 2.0, 3.0, 4.0]
        self.y_pred_perfect = [1.0, 2.0, 3.0, 4.0]
        
        self.regressor = RegressionMetrics()
    
    def test_mean_squared_error(self):
        """Test MSE calculation."""
        mse = self.regressor.mean_squared_error(self.y_true, self.y_pred)
        # Manual calculation: ((3-2.5)^2 + (-0.5-0)^2 + (2-2)^2 + (7-8)^2) / 4
        expected = (0.25 + 0.25 + 0 + 1) / 4  # 0.375
        assert abs(mse - expected) < 1e-6
    
    def test_mean_squared_error_perfect(self):
        """Test MSE with perfect predictions."""
        mse = self.regressor.mean_squared_error(self.y_true_perfect, self.y_pred_perfect)
        assert mse == 0.0
    
    def test_root_mean_squared_error(self):
        """Test RMSE calculation."""
        rmse = self.regressor.root_mean_squared_error(self.y_true, self.y_pred)
        mse = self.regressor.mean_squared_error(self.y_true, self.y_pred)
        expected = np.sqrt(mse)
        assert abs(rmse - expected) < 1e-6
    
    def test_mean_absolute_error(self):
        """Test MAE calculation."""
        mae = self.regressor.mean_absolute_error(self.y_true, self.y_pred)
        # Manual calculation: (|3-2.5| + |-0.5-0| + |2-2| + |7-8|) / 4
        expected = (0.5 + 0.5 + 0 + 1) / 4  # 0.5
        assert abs(mae - expected) < 1e-6
    
    def test_r_squared(self):
        """Test R² calculation."""
        r2 = self.regressor.r_squared(self.y_true, self.y_pred)
        
        # R² should be between -∞ and 1
        assert r2 <= 1.0
    
    def test_r_squared_perfect(self):
        """Test R² with perfect predictions."""
        r2 = self.regressor.r_squared(self.y_true_perfect, self.y_pred_perfect)
        assert abs(r2 - 1.0) < 1e-6
    
    def test_mean_absolute_percentage_error(self):
        """Test MAPE calculation."""
        # Use positive values only for MAPE
        y_true = [2.0, 4.0, 6.0, 8.0]
        y_pred = [1.5, 4.5, 5.5, 8.5]
        
        mape = self.regressor.mean_absolute_percentage_error(y_true, y_pred)
        
        # MAPE should be positive
        assert mape >= 0
    
    def test_mape_with_zeros(self):
        """Test MAPE with zero values (should issue warning)."""
        y_true = [0.0, 1.0, 2.0]
        y_pred = [0.5, 1.5, 2.5]
        
        with pytest.warns(UserWarning):
            mape = self.regressor.mean_absolute_percentage_error([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            assert mape == float('inf')


class TestCrossEntropyLoss:
    """Test cases for cross-entropy loss function."""
    
    def setup_method(self):
        """Set up test data."""
        # One-hot encoded true labels
        self.y_true_probs = [
            [1.0, 0.0, 0.0],  # Class 0
            [0.0, 1.0, 0.0],  # Class 1
            [0.0, 0.0, 1.0],  # Class 2
        ]
        
        # Predicted probabilities
        self.y_pred_probs = [
            [0.8, 0.1, 0.1],  # Close to class 0
            [0.2, 0.7, 0.1],  # Close to class 1
            [0.1, 0.2, 0.7],  # Close to class 2
        ]
        
        # Perfect predictions
        self.y_pred_perfect = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    
    def test_cross_entropy_calculation(self):
        """Test cross-entropy loss calculation."""
        loss = CrossEntropyLoss.calculate(self.y_true_probs, self.y_pred_probs)
        
        # Loss should be positive
        assert loss > 0
    
    def test_cross_entropy_perfect(self):
        """Test cross-entropy with perfect predictions."""
        loss = CrossEntropyLoss.calculate(self.y_true_probs, self.y_pred_perfect)
        
        # Loss should be very close to 0 (epsilon prevents exactly 0)
        assert loss < 1e-6
    
    def test_one_hot_encode(self):
        """Test one-hot encoding function."""
        labels = [0, 1, 2, 0]
        num_classes = 3
        
        one_hot = CrossEntropyLoss.one_hot_encode(labels, num_classes)
        expected = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
        
        np.testing.assert_array_equal(one_hot, expected)
    
    def test_cross_entropy_shape_mismatch(self):
        """Test cross-entropy with mismatched shapes."""
        y_true = [[1.0, 0.0]]
        y_pred = [[0.8, 0.1, 0.1]]
        
        with pytest.raises(ValueError):
            CrossEntropyLoss.calculate(y_true, y_pred)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.y_true = [0, 1, 1, 0, 1]
        self.y_pred = [0, 1, 0, 0, 1]
    
    def test_accuracy_score_function(self):
        """Test accuracy_score convenience function."""
        accuracy = accuracy_score(self.y_true, self.y_pred)
        expected = ClassificationMetrics().accuracy(self.y_true, self.y_pred)
        assert accuracy == expected
    
    def test_precision_recall_f1_score_function(self):
        """Test precision_recall_f1_score convenience function."""
        metrics = precision_recall_f1_score(self.y_true, self.y_pred, average='macro')
        expected = ClassificationMetrics(average='macro').precision_recall_f1(self.y_true, self.y_pred)
        assert metrics == expected
    
    def test_confusion_matrix_score_function(self):
        """Test confusion_matrix_score convenience function."""
        cm = confusion_matrix_score(self.y_true, self.y_pred)
        expected = ClassificationMetrics().confusion_matrix(self.y_true, self.y_pred)
        np.testing.assert_array_equal(cm, expected)
    
    def test_mean_squared_error_score_function(self):
        """Test mean_squared_error_score convenience function."""
        y_true_reg = [1.0, 2.0, 3.0]
        y_pred_reg = [1.1, 2.1, 2.9]
        
        mse = mean_squared_error_score(y_true_reg, y_pred_reg)
        expected = RegressionMetrics().mean_squared_error(y_true_reg, y_pred_reg)
        assert mse == expected
    
    def test_r2_score_function(self):
        """Test r2_score convenience function."""
        y_true_reg = [1.0, 2.0, 3.0]
        y_pred_reg = [1.1, 2.1, 2.9]
        
        r2 = r2_score(y_true_reg, y_pred_reg)
        expected = RegressionMetrics().r_squared(y_true_reg, y_pred_reg)
        assert r2 == expected


# Integration tests that simulate the Rust core behavior
class TestIntegrationWithRustCore:
    """Integration tests that mirror the Rust core implementation."""
    
    def test_cross_entropy_mirrors_rust(self):
        """Test that Python cross-entropy matches Rust implementation behavior."""
        # Simulate data that would come from the Rust neural network
        batch_size = 4
        num_classes = 3
        
        # True labels (as would be one-hot encoded in Rust)
        y_true = [0, 1, 2, 0]
        y_true_one_hot = CrossEntropyLoss.one_hot_encode(y_true, num_classes)
        
        # Predicted probabilities (as would come from softmax in Rust)
        y_pred_probs = [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.3, 0.5],
            [0.6, 0.3, 0.1],
        ]
        
        # Calculate loss
        loss = CrossEntropyLoss.calculate(y_true_one_hot, y_pred_probs)
        
        # Verify loss is reasonable
        assert 0 < loss < 10  # Should be positive but not too large
    
    def test_classification_pipeline(self):
        """Test a complete classification evaluation pipeline."""
        # Simulate predictions from the Rust neural network
        y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]
        
        # Calculate all classification metrics
        classifier = ClassificationMetrics(average='macro')
        
        accuracy = classifier.accuracy(y_true, y_pred)
        metrics = classifier.precision_recall_f1(y_true, y_pred)
        cm = classifier.confusion_matrix(y_true, y_pred)
        report = classifier.classification_report(y_true, y_pred)
        
        # Verify all metrics are reasonable
        assert 0 <= accuracy <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert cm.shape == (2, 2)
        assert len(report) > 100  # Report should be substantial


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__, "-v"])
