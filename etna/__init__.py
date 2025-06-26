"""
ETNA - A Neural Network Framework with Rust Core and Python Interface

ETNA provides a high-performance neural network framework that combines
the speed of Rust with the convenience of Python.

Main modules:
- metrics: Comprehensive evaluation metrics for ML models
- api: User-facing API for classifiers and regression
- preprocessing: Data preprocessing utilities  
- utils: Utility functions and helpers
- cli: Command-line interface

The core neural network implementation is written in Rust for optimal performance,
while the Python interface provides ease of use and integration with the ML ecosystem.
"""

__version__ = "0.1.0"
__author__ = "ETNA Team"

# Import main modules
try:
    from . import metrics
    from . import api
    from . import preprocessing
    from . import utils
    from . import cli
    
    # Import commonly used functions for convenience
    from .metrics import (
        accuracy_score,
        precision_recall_f1_score,
        confusion_matrix_score,
        mean_squared_error_score,
        r2_score,
        ClassificationMetrics,
        RegressionMetrics,
        CrossEntropyLoss
    )
    
    __all__ = [
        # Modules
        'metrics',
        'api', 
        'preprocessing',
        'utils',
        'cli',
        
        # Commonly used functions
        'accuracy_score',
        'precision_recall_f1_score', 
        'confusion_matrix_score',
        'mean_squared_error_score',
        'r2_score',
        
        # Classes
        'ClassificationMetrics',
        'RegressionMetrics', 
        'CrossEntropyLoss',
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    import warnings
    warnings.warn(f"Some ETNA modules could not be imported: {e}")
    
    # Minimal exports when dependencies are missing
    __all__ = ['__version__', '__author__']