# ETNA Metrics Module

The ETNA metrics module provides comprehensive evaluation metrics for machine learning models, designed to work seamlessly with the ETNA neural network framework's Rust core implementation.

## Features

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision, Recall, F1-Score**: With support for binary, macro, micro, and weighted averaging
- **Confusion Matrix**: Detailed breakdown of predictions vs. true labels
- **Classification Report**: Comprehensive summary of all metrics

### Regression Metrics
- **Mean Squared Error (MSE)**: Standard regression loss
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Error (MAE)**: Average absolute differences
- **R-squared (R²)**: Coefficient of determination
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error metric

### Loss Functions
- **Cross-Entropy Loss**: Mirrors the Rust implementation for seamless integration
- **One-Hot Encoding**: Convert integer labels to one-hot vectors

## Installation

The metrics module requires NumPy. Install dependencies with:

```bash
pip install numpy
```

## Quick Start

### Basic Classification Example

```python
from etna.metrics import accuracy_score, precision_recall_f1_score

# Your model predictions
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 0, 0, 1, 1]

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
metrics = precision_recall_f1_score(y_true, y_pred, average='binary')

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1-Score: {metrics['f1']:.3f}")
```

### Advanced Classification Analysis

```python
from etna.metrics import ClassificationMetrics

# Create metrics calculator
classifier = ClassificationMetrics(average='macro')

# Multi-class data
y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 2, 0, 2, 1, 0, 1, 1]

# Calculate comprehensive metrics
accuracy = classifier.accuracy(y_true, y_pred)
metrics = classifier.precision_recall_f1(y_true, y_pred)
confusion_matrix = classifier.confusion_matrix(y_true, y_pred)

# Generate detailed report
report = classifier.classification_report(y_true, y_pred)
print(report)
```

### Regression Example

```python
from etna.metrics import RegressionMetrics

regressor = RegressionMetrics()

y_true = [3.0, -0.5, 2.0, 7.0]
y_pred = [2.5, 0.0, 2.0, 8.0]

mse = regressor.mean_squared_error(y_true, y_pred)
rmse = regressor.root_mean_squared_error(y_true, y_pred)
mae = regressor.mean_absolute_error(y_true, y_pred)
r2 = regressor.r_squared(y_true, y_pred)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")
```

### Integration with Rust Neural Network

```python
from etna.metrics import CrossEntropyLoss

# Convert integer labels to one-hot (as done in Rust)
y_true_labels = [0, 1, 2, 0]
y_true_one_hot = CrossEntropyLoss.one_hot_encode(y_true_labels, num_classes=3)

# Predicted probabilities from neural network softmax
y_pred_probs = [
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.6, 0.3, 0.1],
]

# Calculate loss (same algorithm as Rust implementation)
loss = CrossEntropyLoss.calculate(y_true_one_hot, y_pred_probs)
print(f"Cross-entropy loss: {loss:.4f}")
```

## API Reference

### ClassificationMetrics

```python
ClassificationMetrics(average='macro')
```

**Parameters:**
- `average` (str): Averaging method for multi-class metrics
  - `'binary'`: Binary classification
  - `'macro'`: Unweighted mean of per-class metrics
  - `'micro'`: Calculate globally across all classes
  - `'weighted'`: Weighted by support (number of true instances)

**Methods:**
- `accuracy(y_true, y_pred)`: Calculate accuracy score
- `precision_recall_f1(y_true, y_pred)`: Calculate precision, recall, and F1
- `confusion_matrix(y_true, y_pred)`: Generate confusion matrix
- `classification_report(y_true, y_pred)`: Generate comprehensive report

### RegressionMetrics

**Methods:**
- `mean_squared_error(y_true, y_pred)`: Calculate MSE
- `root_mean_squared_error(y_true, y_pred)`: Calculate RMSE
- `mean_absolute_error(y_true, y_pred)`: Calculate MAE
- `r_squared(y_true, y_pred)`: Calculate R²
- `mean_absolute_percentage_error(y_true, y_pred)`: Calculate MAPE

### CrossEntropyLoss

**Static Methods:**
- `calculate(y_true_probs, y_pred_probs)`: Calculate cross-entropy loss
- `one_hot_encode(labels, num_classes)`: Convert integer labels to one-hot

### Convenience Functions

For quick access without creating class instances:

```python
from etna.metrics import (
    accuracy_score,
    precision_recall_f1_score,
    confusion_matrix_score,
    mean_squared_error_score,
    r2_score
)
```

## Integration with ETNA Framework

The metrics module is designed to work seamlessly with the ETNA Rust neural network core:

1. **Cross-Entropy Loss**: Uses the same algorithm as the Rust `cross_entropy` function
2. **One-Hot Encoding**: Compatible with the Rust `one_hot_encode` function  
3. **Data Types**: Handles the same data formats used by the Rust-Python bridge
4. **Performance**: Optimized for the prediction formats output by the Rust neural network

## Examples

See the `examples/` directory for comprehensive usage examples:

- `metrics_example.py`: Complete demonstration of all metrics
- `rust_integration_example.py`: Integration with Rust neural network simulation

## Error Handling

The module includes comprehensive error handling:

- **Shape Validation**: Ensures y_true and y_pred have matching shapes
- **Empty Data**: Handles empty input arrays gracefully
- **Division by Zero**: Protects against undefined metric calculations
- **Type Conversion**: Automatically converts lists to NumPy arrays

## Performance Considerations

- Uses NumPy for efficient numerical computations
- Minimizes memory usage with in-place operations where possible
- Optimized algorithms for large datasets
- Supports both lists and NumPy arrays as input

## Testing

Run the test suite with:

```bash
python -m pytest tests/test_metrics.py -v
```

The test suite includes:
- Unit tests for all metrics
- Integration tests with simulated Rust behavior
- Edge case handling
- Performance benchmarks

## Contributing

When contributing to the metrics module:

1. Ensure all new metrics include comprehensive tests
2. Maintain compatibility with the Rust core implementation
3. Add examples for new functionality
4. Update this documentation

## License

This module is part of the ETNA framework and is licensed under the MIT License.
