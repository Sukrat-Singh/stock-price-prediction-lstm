# Stock Price Prediction using LSTM

A production-grade, end-to-end time series forecasting system for stock prices, built with a strong focus on **data leakage prevention**, **reproducibility**, and **ML engineering best practices**.

This project implements the complete ML lifecycle — from raw data ingestion to model training, evaluation, and experiment tracking — using modular components and automated pipelines.

> ⚠️ **Disclaimer**: This project is for educational purposes only and does not provide financial or investment advice.

---

## 🚀 Project Highlights

- End-to-end ML pipeline (ingestion → training → evaluation)
- Strictly leakage-safe preprocessing for time-series data
- Modular, testable components
- Automated training pipeline with MLflow tracking
- Honest evaluation on unseen test data
- Designed for extensibility and production readiness

---

## 🧠 Problem Statement

Given historical stock price data, the goal is to **predict the next closing price** using an LSTM-based sequence model.

This is framed as a **time-series regression problem**, with additional analysis on **directional accuracy** (up/down movement), which is often more meaningful than magnitude alone in financial forecasting.

---

## 🗂️ Project Structure

```text
├── .pytest_cache/                 # Pytest cache (ignored)
├── artifacts/                     # Generated data & model artifacts
│   ├── raw/
│   ├── processed/
│   ├── transformed/
│   ├── models/
│   └── scalers/
│
├── config/
│   └── data.yaml                  # Data source & split configuration
│
├── logs/                           # Application logs (ignored)
│
├── lstm_venv/                     # Virtual environment (ignored)
│
├── mlruns/                        # MLflow experiment tracking (ignored)
├── mlflow.db                      # MLflow backend store (ignored)
│
├── src/
│   ├── components/               # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── windowing.py
│   │   ├── scaler.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   │
│   ├── pipeline/                 # Orchestration pipelines
│   │   └── train_pipeline.py
│   │
│   ├── config_loader.py           # YAML configuration loader
│   ├── exceptions.py              # Custom exception handling
│   ├── logger.py                  # Centralized logging
│   └── __init__.py
│
├── tests/                         # Unit tests
│   ├── test_data_ingestion.py
│   ├── test_data_validation.py
│   ├── test_windowing.py
│   └── test_scaler.py
│
├── .gitignore                     # Git ignore rules
├── LICENSE                        # MIT License
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup
├── temp.ipynb                     # Experimental notebook (ignored)
└── temp.py                        # Scratch script (ignored)
```

---

## 🔄 Machine Learning Pipeline

The project follows a **modular, leakage-safe machine learning pipeline** designed to reflect real-world production workflows for time-series forecasting.

Each stage has a **single responsibility**, is **independently testable**, and produces explicit artifacts for reproducibility.

---

### 1️⃣ Data Ingestion
- Downloads historical stock price data from an external source
- Handles provider-specific schema quirks (e.g., MultiIndex columns)
- Persists raw data to disk as an immutable artifact

**Output**
- `artifacts/raw/stock_data.csv`

---

### 2️⃣ Data Validation & Splitting
- Validates schema and required columns
- Enforces strict chronological ordering
- Performs leakage-safe train / validation / test split
- Stores split metadata for auditability

**Output**
- `artifacts/processed/train.csv`
- `artifacts/processed/val.csv`
- `artifacts/processed/test.csv`
- `artifacts/processed/split_metadata.json`

---

### 3️⃣ Feature Engineering
- Generates time-series features using only historical information:
  - Log returns
  - Rolling means
  - Rolling volatility
  - Volume-based features
- Ensures no future data is used during feature creation

**Output**
- `artifacts/transformed/train_features.csv`
- `artifacts/transformed/val_features.csv`
- `artifacts/transformed/test_features.csv`

---

### 4️⃣ Windowing (Sequence Creation)
- Converts tabular time-series data into fixed-length sequences
- Uses a sliding window approach:
  - Inputs: past `T` timesteps
  - Target: next timestep value
- Guarantees correct temporal alignment (no look-ahead bias)

**Output**
- NumPy arrays with shape `(N, T, F)` for model input

---

### 5️⃣ Feature Scaling
- Fits scaler **only on training data**
- Applies the same scaler to validation and test sets
- Preserves temporal structure while normalizing feature magnitudes
- Saves scaler artifact for reuse during inference

**Output**
- Scaled windowed arrays
- `artifacts/scalers/feature_scaler.pkl`

---

### 6️⃣ Model Training
- Trains an LSTM-based regression model
- Uses validation loss for checkpoint selection
- Logs training progress and metrics
- Saves the best-performing model

**Output**
- `artifacts/models/lstm_model.pt`

---

### 7️⃣ Model Evaluation
- Evaluates model performance on an unseen test set
- Reports:
  - RMSE
  - MAE
  - Directional Accuracy (up/down movement)
- Ensures no retraining or refitting on test data

---

### 8️⃣ Experiment Tracking
- Entire pipeline is orchestrated via a training pipeline
- Parameters, metrics, and model artifacts are logged using **MLflow**
- Enables experiment comparison and reproducibility

---

### 🧠 Key Design Principles
- **No data leakage** at any stage
- **Train-only fitting** for all learned transformations
- **Modular components** instead of monolithic scripts
- **Reproducible artifacts** at each pipeline stage

---

## 🧪 Testing Strategy

This project emphasizes correctness tests, not performance tests.

Covered invariants include:

 - Schema validation

 - Chronological data splits

 - Window shape correctness

 - Target alignment (no future leakage)

 - Deterministic scaling behavior

```bash
pytest
```

All core preprocessing and sequence logic is covered by unit tests.

---

## 📊 Model Evaluation (Current Baseline)

The current model is a baseline LSTM, trained without aggressive tuning.

Example test-set metrics:

RMSE: ~125

MAE: ~124

Directional Accuracy: ~0.55

While absolute error remains high (expected for a first baseline), the model demonstrates better-than-random directional prediction, indicating learned temporal structure.

These results are intentionally reported without overfitting or test-set leakage.

---

## 📈 Visual Analysis (Planned)

The following plots will be added to support model interpretation:

 - Training vs Validation Loss

 - Actual vs Predicted Prices (Test Set)

 - Prediction Error Over Time

 - Directional Accuracy Summary

 - These will be included in the reports/ directory.

---

## 🛠️ How to Run

### 1️⃣ Setup environment
```bash
python -m venv lstm_venv
source lstm_venv/bin/activate  # or activate.ps1 on Windows
pip install -r requirements.txt
```
### 2️⃣ Run full training pipeline
```bash
python -m src.pipeline.train_pipeline
```

### 3️⃣ Launch MLflow UI
```bash
mlflow ui
```


Open:
```bash
http://127.0.0.1:5000
```

---

## 🧠 Design Decisions

 - **No K-Fold Cross Validation**:- Standard K-Fold violates temporal ordering and causes leakage in time-series problems.

 - **Train-only preprocessing**:- All learned statistics (scalers, features) are fit strictly on training data.

 - **Simple baseline before complexity**:- Model complexity is intentionally limited to establish a trustworthy baseline.

---

## 📌 Future Improvements

 - Baseline comparisons (naive, moving average)

 - Hyperparameter tuning

 - Error analysis by volatility regime

 - FastAPI-based prediction service

 - Model monitoring & drift detection

---

### 📜 License

MIT License

---


### 👤 Author

Built by **Sukrat Singh**

Engineering Student, IIT Dhanbad
