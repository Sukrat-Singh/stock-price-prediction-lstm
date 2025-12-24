import numpy as np
from src.components.scaler import TimeSeriesScaler


def test_scaler_preserves_shape():
    """
    Ensures scaling does not change input shape.
    """
    X = np.random.rand(10, 5, 3)

    scaler = TimeSeriesScaler()
    X_scaled = scaler.fit_transform(X)

    assert X_scaled.shape == X.shape


def test_scaler_is_deterministic():
    """
    Ensures transform is deterministic after fitting.
    """
    X_train = np.random.rand(10, 5, 3)
    X_test = np.random.rand(5, 5, 3)

    scaler = TimeSeriesScaler()
    scaler.fit_transform(X_train)

    X_test_1 = scaler.transform(X_test)
    X_test_2 = scaler.transform(X_test)

    assert np.allclose(X_test_1, X_test_2)
