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
from . import metrics
# from . import api
from . import preprocessing
from . import utils
from . import cli

# --- THIS IS THE KEY FIX ---
##from .api import Model
# ---------------------------------------------------------------------
# Optional Rust-backed API
# This allows tests to run without building the Rust extension
# ---------------------------------------------------------------------
try:
    from . import api
    from .api import Model
except ImportError:
    api = None
    Model = None
# ---------------------------

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
    'Model',
    'metrics',
    'api', 
    'preprocessing',
    'utils',
    'cli',
    'accuracy_score',
    'precision_recall_f1_score', 
    'confusion_matrix_score',
    'mean_squared_error_score',
    'r2_score',
    'ClassificationMetrics',
    'RegressionMetrics', 
    'CrossEntropyLoss',
]
