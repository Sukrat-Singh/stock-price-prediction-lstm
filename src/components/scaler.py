from src.exceptions import CustomException
from src.logger import logging

import os
import joblib
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler

@dataclass
class ScalerConfig:
    scaler_dir: str = "artifacts/scalers"
    scaler_file: str = "feature_scaler.pkl"


class TimeSeriesScaler:
    """
    Applies train-only feature scaling for windowed time-series data.
    """

    def __init__(self):
        self.config = ScalerConfig()
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fits scaler on training data and transforms it.

        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (N, T, F).

        Returns
        -------
        np.ndarray
            Scaled training data with same shape.
        """
        try:
            logging.info("Fitting scaler on training data")

            X_train_2d = self._to_2d(X_train)
            X_train_scaled = self.scaler.fit_transform(X_train_2d)

            self._save_scaler()

            return self._to_3d(X_train_scaled, X_train.shape)

        except Exception as e:
            raise CustomException(e)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies pre-fitted scaler to validation or test data.

        Parameters
        ----------
        X : np.ndarray
            Data of shape (N, T, F).

        Returns
        -------
        np.ndarray
            Scaled data with same shape.
        """
        try:
            logging.info("Applying scaler to data")

            X_2d = self._to_2d(X)
            X_scaled = self.scaler.transform(X_2d)

            return self._to_3d(X_scaled, X.shape)

        except Exception as e:
            raise CustomException(e)

    def _to_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Reshapes 3D windowed data into 2D for scaling.
        """
        return X.reshape(-1, X.shape[-1])

    def _to_3d(self, X: np.ndarray, original_shape: tuple) -> np.ndarray:
        """
        Restores scaled 2D data back to original 3D shape.
        """
        return X.reshape(original_shape)

    def _save_scaler(self) -> None:
        """
        Saves fitted scaler to disk for reuse during inference.
        """
        os.makedirs(self.config.scaler_dir, exist_ok=True)
        path = os.path.join(self.config.scaler_dir, self.config.scaler_file)
        joblib.dump(self.scaler, path)

        logging.info(f"Scaler saved at {path}")
