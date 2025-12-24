from src.exceptions import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class WindowingConfig:
    window_size: int = 30 
    target_column: str = 'Close'

class WindowGenerator:
    """
    Converts tabular time-series data into fixed-length
    sequences suitable for sequence models such as LSTM.
    """

    def __init__(self, window_size: int = 30, target_column: str = "Close"):
        self.config = WindowingConfig(
            window_size=window_size,
            target_column=target_column
        )


    def create_windows(self, data_path: str) -> tuple:
        """
        Creates sliding windows and corresponding targets
        from a feature-engineered dataset.
        
        :param data_path: path to training feature csv file
        :type data_path: str
        :return: X and y numpy arrays where:
            X has shape (num_samples, window_size, num_features)
            y has shape (num_samples,)
        :rtype: tuple
        """
        try:
            logging.info(f'Creating windows from {data_path}')

            df = pd.read_csv(data_path)

            if self.config.target_column not in df.columns:
                raise ValueError(f"target column {self.config.target_column} not found!")
            
            df = df.drop(columns=['Date'])

            feature_columns = [
                col for col in df.columns
                if col != self.config.target_column
            ]

            X, y = self._generate_sequences(
                df,
                feature_columns,
                self.config.target_column
            )

            logging.info(
                f"Windowing completed | "
                f"X shape: {X.shape} | "
                f"y shape: {y.shape}"
            )

            return X, y
        
        except Exception as e:
            raise CustomException(e)
        
    def _generate_sequences(
            self, 
            df: pd.DataFrame,
            feature_columns: list,
            target_column: str
    ) -> tuple:
        """
        Generates sliding window sequences from dataframe.
        """
        data = df.reset_index(drop=True)

        X = []
        y = []

        window_size = self.config.window_size

        for i in range(window_size, len(data)):
            X.append(
                data.loc[i - window_size:i - 1, feature_columns].values
            )
            y.append(
                data.loc[i, target_column]
            )

        return np.array(X), np.array(y)