import os
import torch
import torch.nn as nn
from dataclasses import dataclass

from src.logger import logging
from src.exceptions import CustomException


@dataclass
class ModelTrainerConfig:
    model_dir: str = "artifacts/models"
    model_name: str = "lstm_model.pt"
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMModel(nn.Module):
    """
    Simple LSTM model for time-series regression.
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model.
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


class ModelTrainer:
    """
    Handles training and validation of the LSTM model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2
    ):
        self.config = ModelTrainerConfig()
        self.device = self.config.device

        self.model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> str:
        """
        Trains the LSTM model using training and validation data.

        Parameters
        ----------
        X_train, y_train, X_val, y_val
            NumPy arrays produced after windowing and scaling.

        Returns
        -------
        str
            Path to the saved best model.
        """
        try:
            logging.info("Starting model training")

            X_train, y_train = self._to_tensor(X_train, y_train)
            X_val, y_val = self._to_tensor(X_val, y_val)

            best_val_loss = float("inf")

            for epoch in range(self.config.epochs):
                train_loss = self._train_one_epoch(X_train, y_train)
                val_loss = self._validate(X_val, y_val)

                logging.info(
                    f"Epoch [{epoch + 1}/{self.config.epochs}] "
                    f"Train Loss: {train_loss:.6f} "
                    f"Val Loss: {val_loss:.6f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_model()

            return os.path.join(
                self.config.model_dir,
                self.config.model_name
            )

        except Exception as e:
            raise CustomException(e)

    def _train_one_epoch(self, X, y) -> float:
        """
        Performs one training epoch.
        """
        self.model.train()
        total_loss = 0.0

        for i in range(0, len(X), self.config.batch_size):
            xb = X[i:i + self.config.batch_size]
            yb = y[i:i + self.config.batch_size]

            self.optimizer.zero_grad()
            preds = self.model(xb)
            loss = self.criterion(preds, yb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(xb)

        return total_loss / len(X)

    def _validate(self, X, y) -> float:
        """
        Evaluates model on validation data.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, len(X), self.config.batch_size):
                xb = X[i:i + self.config.batch_size]
                yb = y[i:i + self.config.batch_size]

                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                total_loss += loss.item() * len(xb)

        return total_loss / len(X)

    def _to_tensor(self, X, y):
        """
        Converts NumPy arrays to PyTorch tensors.
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        return X_tensor, y_tensor

    def _save_model(self) -> None:
        """
        Saves the current model state to disk.
        """
        os.makedirs(self.config.model_dir, exist_ok=True)
        path = os.path.join(
            self.config.model_dir,
            self.config.model_name
        )
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model checkpoint saved at {path}")
