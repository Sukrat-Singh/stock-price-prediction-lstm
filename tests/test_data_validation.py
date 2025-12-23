import os
import pandas as pd
import pytest
from datetime import datetime, timedelta

from src.components.data_validation import DataValidation


def _create_dummy_stock_data(tmp_path, rows=100):
    """
    Creates a dummy stock CSV with strictly increasing dates
    and valid OHLCV columns.
    """
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(rows)]

    df = pd.DataFrame({
        "Date": dates,
        "Open": range(rows),
        "High": range(rows),
        "Low": range(rows),
        "Close": range(rows),
        "Volume": range(rows)
    })

    file_path = tmp_path / "dummy_stock.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


def test_validation_creates_all_splits(tmp_path):
    """
    Ensures train, validation, and test files are created successfully.
    """
    raw_path = _create_dummy_stock_data(tmp_path)
    validator = DataValidation()

    paths = validator.validate_and_split(raw_path)

    assert os.path.exists(paths["train"])
    assert os.path.exists(paths["val"])
    assert os.path.exists(paths["test"])


def test_splits_are_chronological(tmp_path):
    """
    Ensures there is no temporal overlap between train, validation,
    and test splits.
    """
    raw_path = _create_dummy_stock_data(tmp_path)
    validator = DataValidation()

    paths = validator.validate_and_split(raw_path)

    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])

    assert train_df["Date"].max() < val_df["Date"].min()
    assert val_df["Date"].max() < test_df["Date"].min()


def test_missing_required_column_raises_error(tmp_path):
    """
    Ensures validation fails if a required column is missing.
    """
    start_date = datetime(2020, 1, 1)
    df = pd.DataFrame({
        "Date": [start_date + timedelta(days=i) for i in range(10)],
        "Open": range(10),
        "High": range(10),
        "Low": range(10),
        "Close": range(10)
    })

    file_path = tmp_path / "missing_volume.csv"
    df.to_csv(file_path, index=False)

    validator = DataValidation()

    with pytest.raises(Exception):
        validator.validate_and_split(str(file_path))


def test_duplicate_timestamps_raise_error(tmp_path):
    """
    Ensures duplicate timestamps are detected and rejected.
    """
    start_date = datetime(2020, 1, 1)
    dates = [start_date for _ in range(10)]

    df = pd.DataFrame({
        "Date": dates,
        "Open": range(10),
        "High": range(10),
        "Low": range(10),
        "Close": range(10),
        "Volume": range(10)
    })

    file_path = tmp_path / "duplicate_dates.csv"
    df.to_csv(file_path, index=False)

    validator = DataValidation()

    with pytest.raises(Exception):
        validator.validate_and_split(str(file_path))


def test_split_metadata_file_created():
    """
    Ensures split metadata JSON file is created after validation.
    """
    validator = DataValidation()

    assert os.path.exists(
        os.path.join("artifacts", "processed", "split_metadata.json")
    )
