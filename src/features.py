import numpy as np
import pandas as pd


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def compute_volatility(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    df = df.copy()
    df["volatility"] = (
        df["log_return"]
        .rolling(window=window)
        .std()
        * np.sqrt(252)
    )
    return df


def compute_volume_features(df: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """
    Add volume-based features:
    - volume_change: day-over-day % change in volume (captures sudden spikes)
    - volume_ma: rolling mean volume (captures sustained elevated activity)
    """
    df = df.copy()
    df["volume_change"] = df["Volume"].pct_change()
    df["volume_ma"] = df["Volume"].rolling(window=window).mean()
    return df


def create_features(df: pd.DataFrame, n_lags: int = 5, include_volume: bool = False) -> pd.DataFrame:
    """
    Create lagged volatility features: vol_lag_1 ... vol_lag_n.
    If include_volume=True, also adds lagged volume_change and volume_ma.
    """
    df = df.copy()

    for i in range(1, n_lags + 1):
        df[f"vol_lag_{i}"] = df["volatility"].shift(i)

    if include_volume:
        df["volume_change_lag_1"] = df["volume_change"].shift(1)
        df["volume_ma_lag_1"] = df["volume_ma"].shift(1)

    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df["volatility"].shift(-1)
    return df


def prepare_dataset(df: pd.DataFrame, n_lags: int = 5, include_volume: bool = False):
    """
    Full feature pipeline: raw prices -> X, y ready for model.
    Set include_volume=True to add volume_change_lag_1 and volume_ma_lag_1
    alongside the volatility lag features.

    Returns:
        X (np.ndarray): feature matrix
        y (np.ndarray): target vector
        df (pd.DataFrame): full dataframe with all intermediate columns
    """
    df = compute_log_returns(df)
    df = compute_volatility(df)

    if include_volume:
        df = compute_volume_features(df)

    df = create_features(df, n_lags, include_volume=include_volume)
    df = create_target(df)

    df = df.dropna()

    feature_cols = [f"vol_lag_{i}" for i in range(1, n_lags + 1)]
    if include_volume:
        feature_cols += ["volume_change_lag_1", "volume_ma_lag_1"]

    X = df[feature_cols].values
    y = df["target"].values

    return X, y, df