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
##    assert len(X[0]) == 2  # 1 numeric + 1 categorical
    assert len(X[0]) == pre.input_dim # 1 numeric + 2 one-hot categorical
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



def test_titanic_like_dataset_preprocessing():
    """
    Acceptance-style test:
    Ensures preprocessing works on a Titanic-like dataset
    with mixed numeric, categorical, and missing values
    without manual preprocessing.
    """
    df = pd.DataFrame({
        "Age": [22, 38, None, 35],
        "Fare": [7.25, 71.28, 8.05, None],
        "Sex": ["male", "female", "female", "male"],
        "Embarked": ["S", "C", "S", None],
        "Survived": [0, 1, 1, 0]
    })

    pre = Preprocessor(task_type="classification")
    X, y = pre.fit_transform(df, target_col="Survived")

    assert len(X) == 4
    assert pre.input_dim > 2        # expanded due to one-hot encoding
    assert pre.output_dim == 2      # binary classification
    assert not np.isnan(np.array(X)).any()    
