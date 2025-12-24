import numpy as np
import pandas as pd
from src.components.windowing import WindowGenerator


def test_window_shapes(tmp_path):
    """
    Ensures windowing produces correct 3D input and 1D target.
    """
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=50),
        "Close": np.arange(50),
        "feat1": np.arange(50),
        "feat2": np.arange(50)
    })

    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    wg = WindowGenerator(window_size=10, target_column="Close")
    X, y = wg.create_windows(str(path))

    assert X.shape == (40, 10, 2)
    assert y.shape == (40,)


def test_window_target_alignment(tmp_path):
    """
    Ensures target corresponds to timestep immediately after window.
    """
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=20),
        "Close": np.arange(20),
        "feat": np.arange(20)
    })

    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    wg = WindowGenerator(window_size=5, target_column="Close")
    X, y = wg.create_windows(str(path))

    assert y[0] == df.loc[5, "Close"]
