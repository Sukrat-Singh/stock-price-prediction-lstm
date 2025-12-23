from src.logger import logging
from src.exceptions import CustomException
from src.config_loader import load_config

import os
import pandas as pd
import yfinance as yf
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_dir: str = "artifacts/raw"
    raw_data_file: str = "stock_data.csv"


class DataIngestion:
    def __init__(self):
        self.config = load_config("config/data.yaml")
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> str:
        try:
            logging.info("Starting data ingestion")

            ticker = self.config["source"]["ticker"]
            start = self.config["source"]["start_date"]
            end = self.config["source"]["end_date"]

            df = yf.download(ticker, start=start, end=end)

            if df.empty:
                raise ValueError("Downloaded data is empty")

            df = df.reset_index()
            df = df.sort_values("Date")

            required_columns = {"Open", "High", "Low", "Close", "Volume"}
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            os.makedirs(self.ingestion_config.raw_data_dir, exist_ok=True)
            raw_path = os.path.join(
                self.ingestion_config.raw_data_dir,
                self.ingestion_config.raw_data_file
            )

            df.to_csv(raw_path, index=False)

            logging.info(
                f"Data ingestion completed | Rows: {df.shape[0]} | Path: {raw_path}"
            )

            return raw_path

        except Exception as e:
            raise CustomException(e)
