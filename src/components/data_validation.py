from src.exceptions import CustomException
from src.logger import logging
from src.config_loader import load_config

import pandas as pd
import os
from dataclasses import dataclass

@dataclass
class DataValidationConfig:
    processed_data_dir: str = 'artifacts/processed'
    metadata_file: str = 'split_metadata.json'

class DataValidation:
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

        