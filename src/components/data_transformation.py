from src.exceptions import CustomException
from src.logger import logging

import pandas as pd
import os
import numpy as np
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    transformed_data_dir: str = 'artifacts/transformed'

class DataTransformation:
    """
    Performs leakage-safe data transformation on time-series stock data
    """
    def __init__(self):
        self.config = DataTransformationConfig()

    def transform(
            self,
            train_path: str,
            val_path: str,
            test_path: str 
    ) -> dict:
        """
        Docstring for transform
        Transform train, validation and test dataset using features derived strictly from historical data.
        
        :param train_path: path to training csv file
        :type train_path: str
        :param val_path: path to validation csv file
        :type val_path: str
        :param test_path: path to test csv file
        :type test_path: str
        :return: paths to transformed train, val and test csv files
        :rtype: dict
        """
        try:
            logging.info("starting data transformation")

            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)

            train_df = self._create_features(train_df)
            val_df = self._create_features(val_df)
            test_df = self._create_features(test_df)

            train_df = train_df.dropna().reset_index(drop=True)
            val_df = val_df.dropna().reset_index(drop=True)
            test_df = test_df.dropna().reset_index(drop=True)

            os.makedirs(self.config.transformed_data_dir, exist_ok=True)

            train_out = os.path.join(self.config.transformed_data_dir, "train_features.csv")
            val_out = os.path.join(self.config.transformed_data_dir, "val_features.csv")
            test_out = os.path.join(self.config.transformed_data_dir, "test_features.csv")

            train_df.to_csv(train_out, index=False)
            val_df.to_csv(val_out, index=False)
            test_df.to_csv(test_out, index=False)

            logging.info(
                f"Feature transformation completed | "
                f"Train: {train_df.shape} | "
                f"Val: {val_df.shape} | "
                f"Test: {test_df.shape}"
            )

            return {
                "train": train_out,
                "val": val_out,
                "test": test_out
            }

        except Exception as e:
            raise CustomException(e)
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time-series features without using future information.
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])

        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        df["rolling_mean_5"] = df["Close"].rolling(window=5).mean()
        df["rolling_mean_10"] = df["Close"].rolling(window=10).mean()

        df["rolling_std_5"] = df["Close"].rolling(window=5).std()
        df["rolling_std_10"] = df["Close"].rolling(window=10).std()

        df["volume_pct_change"] = df["Volume"].pct_change()

        return df