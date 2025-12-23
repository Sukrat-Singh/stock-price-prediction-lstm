import os
import pandas as pd
import pytest

from src.components.data_ingestion import DataIngestion


def test_data_ingestion_creates_raw_file():
    """
    Ensures data ingestion creates the raw CSV file
    at the expected artifacts location.
    """
    ingestion = DataIngestion()
    raw_path = ingestion.initiate_data_ingestion()

    assert os.path.exists(raw_path)
    assert raw_path.endswith(".csv")


def test_ingested_data_is_not_empty():
    """
    Ensures ingested dataset contains rows and columns.
    """
    ingestion = DataIngestion()
    raw_path = ingestion.initiate_data_ingestion()

    df = pd.read_csv(raw_path)

    assert not df.empty
    assert df.shape[0] > 100


def test_required_columns_exist_in_raw_data():
    """
    Ensures required OHLCV columns exist in ingested data.
    """
    ingestion = DataIngestion()
    raw_path = ingestion.initiate_data_ingestion()

    df = pd.read_csv(raw_path)

    required_columns = {
        "Date", "Open", "High", "Low", "Close", "Volume"
    }

    assert required_columns.issubset(set(df.columns))


def test_dates_are_sorted_in_ingested_data():
    """
    Ensures ingested data is sorted chronologically by Date.
    """
    ingestion = DataIngestion()
    raw_path = ingestion.initiate_data_ingestion()

    df = pd.read_csv(raw_path)
    dates = pd.to_datetime(df["Date"])

    assert dates.is_monotonic_increasing
