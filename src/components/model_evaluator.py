import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.logger import logging
from src.exceptions import CustomException
from src.components.model_trainer import LSTMModel


@dataclass
class ModelEvaluatorConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ModelEvaluator:
    """
    Evaluates a trained LSTM model on unseen test data.
    """

    def __init__(self, input_size: int, model_path: str):
        self.config = ModelEvaluatorConfig()
        self.device = self.config.device

        self.model = LSTMModel(input_size=input_size)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        self.criterion = nn.MSELoss()

    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluates model performance on test data.

        Parameters
        ----------
        X_test, y_test
            Windowed and scaled NumPy arrays.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        try:
            logging.info("Evaluating model on test set")

            X = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y = torch.tensor(y_test, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                preds = self.model(X)

            preds_np = preds.cpu().numpy()
            y_np = y.cpu().numpy()

            mse = np.mean((preds_np - y_np) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(preds_np - y_np))

            directional_acc = np.mean(
                np.sign(preds_np[1:] - preds_np[:-1])
                == np.sign(y_np[1:] - y_np[:-1])
            )

            metrics = {
                "RMSE": rmse,
                "MAE": mae,
                "Directional_Accuracy": directional_acc
            }

            logging.info(f"Test metrics: {metrics}")

            return metrics

        except Exception as e:
            raise CustomException(e)
