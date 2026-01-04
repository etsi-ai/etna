 # Unit tests for utility functions
"""
Tests for utils.py

Covered:
- CSV loading from disk
- File existence validation
- DataFrame shape and type

Not covered:
- Non-CSV formats
- Large file performance
"""

import pandas as pd
import pytest
from pathlib import Path

from etna.utils import load_data


def test_load_data_success(tmp_path: Path):
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("a,b\n1,2\n3,4")

    df = load_data(str(csv_file))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["a", "b"]


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent.csv")
