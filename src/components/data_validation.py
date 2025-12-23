from src.logger import logging
from src.exceptions import CustomException
from src.config_loader import load_config

import os
import json
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataValidationConfig:
    processed_data_dir: str = "artifacts/processed"
    metadata_file: str = "split_metadata.json"


class DataValidation:
    """
    Validates raw time-series data and performs leakage-safe,
    configuration-driven chronological splitting.
    """

    def __init__(self):
        self.validation_config = DataValidationConfig()
        self.config = load_config("config/data.yaml")

    def validate_and_split(self, raw_data_path: str) -> dict:
        """
        Validates raw stock data and splits it into train, validation,
        and test sets using strict time-based ordering.

        Parameters
        ----------
        raw_data_path : str
            Path to raw CSV file.

        Returns
        -------
        dict
            Dictionary containing paths to train, validation, and test files.
        """
        try:
            logging.info("Starting data validation and splitting")

            df = pd.read_csv(raw_data_path)

            self._validate_schema(df)

            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            self._validate_timestamps(df)

            split_paths, metadata = self._time_based_split(df)

            self._save_metadata(metadata)

            logging.info(
                f"Data split completed | "
                f"Train: {metadata['train_rows']} | "
                f"Val: {metadata['val_rows']} | "
                f"Test: {metadata['test_rows']}"
            )

            return split_paths

        except Exception as e:
            raise CustomException(e)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """
        Ensures required columns are present in the dataset.
        """
        required_columns = {
            "Date", "Open", "High", "Low", "Close", "Volume"
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _validate_timestamps(self, df: pd.DataFrame) -> None:
        """
        Ensures timestamps are valid, non-null, and non-duplicated.
        """
        if df["Date"].isnull().any():
            raise ValueError("Null values found in Date column")

        if df["Date"].duplicated().any():
            raise ValueError("Duplicate timestamps detected")

    def _time_based_split(self, df: pd.DataFrame) -> tuple:
        """
        Performs configuration-driven chronological splitting
        with explicit leakage protection.
        """
        n = len(df)

        train_ratio = self.config["split"]["train"]
        val_ratio = self.config["split"]["val"]

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        self._check_leakage(train_df, val_df, test_df)

        os.makedirs(self.validation_config.processed_data_dir, exist_ok=True)

        train_path = os.path.join(self.validation_config.processed_data_dir, "train.csv")
        val_path = os.path.join(self.validation_config.processed_data_dir, "val.csv")
        test_path = os.path.join(self.validation_config.processed_data_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        metadata = {
            "total_rows": n,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_end_date": str(train_df["Date"].max()),
            "val_start_date": str(val_df["Date"].min()),
            "test_start_date": str(test_df["Date"].min()),
            "split_ratios": self.config["split"]
        }

        return {
            "train": train_path,
            "val": val_path,
            "test": test_path
        }, metadata

    def _check_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        Explicitly prevents temporal overlap between splits.
        """
        if train_df["Date"].max() >= val_df["Date"].min():
            raise ValueError("Train/validation time overlap detected")

        if val_df["Date"].max() >= test_df["Date"].min():
            raise ValueError("Validation/test time overlap detected")

    def _save_metadata(self, metadata: dict) -> None:
        """
        Persists split metadata for reproducibility and auditability.
        """
        metadata_path = os.path.join(
            self.validation_config.processed_data_dir,
            self.validation_config.metadata_file
        )

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
