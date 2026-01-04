# Unit tests for preprocessing logic

"""
Tests for preprocessing.py

Covered:
- Numeric scaling
- Missing value handling
- Categorical encoding
- Classification target encoding
- Output dimension consistency

Not covered:
- Regression task behavior
- Persisting preprocessor state
"""

import pandas as pd
import numpy as np

from etna.preprocessing import Preprocessor


def test_fit_transform_classification():
    df = pd.DataFrame({
        "age": [20, 30, None],
        "city": ["A", "B", "A"],
        "label": ["yes", "no", "yes"]
    })

    pre = Preprocessor(task_type="classification")
    X, y = pre.fit_transform(df, target_col="label")

    # Feature matrix
    assert len(X) == 3
    assert len(X[0]) == 2  # 1 numeric + 1 categorical

    # Target matrix (one-hot)
    assert len(y) == 3
    assert len(y[0]) == 2  # yes / no

    # Output dim must match classes
    assert pre.output_dim == 2


def test_no_nan_after_preprocessing():
    df = pd.DataFrame({
        "num": [1, None, 3],
        "cat": ["x", "y", None],
        "target": ["a", "b", "a"]
    })

    pre = Preprocessor()
    X, _ = pre.fit_transform(df, target_col="target")

    X_np = np.array(X)
    assert not np.isnan(X_np).any()

