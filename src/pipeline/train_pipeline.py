import mlflow
import mlflow.pytorch

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.windowing import WindowGenerator
from src.components.scaler import TimeSeriesScaler
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluator import ModelEvaluator
from src.logger import logging


class TrainPipeline:
    """
    End-to-end training pipeline with MLflow tracking.
    """

    def run(self):
        """
        Executes the full training pipeline and logs results to MLflow.
        """
        logging.info("Starting training pipeline")

        with mlflow.start_run(run_name="lstm_stock_prediction"):

            # -------------------------
            # Data ingestion
            # -------------------------
            ingestion = DataIngestion()
            raw_path = ingestion.initiate_data_ingestion()

            # -------------------------
            # Data validation
            # -------------------------
            validator = DataValidation()
            split_paths = validator.validate_and_split(raw_path)

            # -------------------------
            # Feature transformation
            # -------------------------
            transformer = DataTransformation()
            feature_paths = transformer.transform(
                split_paths["train"],
                split_paths["val"],
                split_paths["test"]
            )

            # -------------------------
            # Windowing
            # -------------------------
            window_size = 30
            wg = WindowGenerator(window_size=window_size)

            X_train, y_train = wg.create_windows(feature_paths["train"])
            X_val, y_val = wg.create_windows(feature_paths["val"])
            X_test, y_test = wg.create_windows(feature_paths["test"])

            # -------------------------
            # Scaling
            # -------------------------
            scaler = TimeSeriesScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # -------------------------
            # Model training
            # -------------------------
            trainer = ModelTrainer(input_size=X_train.shape[-1])
            model_path = trainer.train(
                X_train, y_train,
                X_val, y_val
            )

            # -------------------------
            # Evaluation
            # -------------------------
            evaluator = ModelEvaluator(
                input_size=X_test.shape[-1],
                model_path=model_path
            )
            metrics = evaluator.evaluate(X_test, y_test)

            # -------------------------
            # MLflow logging
            # -------------------------
            mlflow.log_param("window_size", window_size)
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("input_features", X_train.shape[-1])

            mlflow.log_metric("RMSE", metrics["RMSE"])
            mlflow.log_metric("MAE", metrics["MAE"])
            mlflow.log_metric("Directional_Accuracy", metrics["Directional_Accuracy"])

            mlflow.pytorch.log_model(
                trainer.model,
                artifact_path="model"
            )

            logging.info("Training pipeline completed successfully")
