import os
import numpy as np
from datetime import date

from src.data_loader import get_data
from src.features import compute_log_returns, compute_volatility, create_features
from src.model import predict

N_LAGS = 5
MODEL_PATH = "outputs/trained_model.npz"
LIVE_DATA_FILENAME = "sp500_live.csv"  # separate from training's sp500_raw.csv


def load_trained_model(path: str = MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m main` first to train and save a model."
        )
    data = np.load(path)
    return data["w"], float(data["b"]), data["X_mean"], data["X_std"]


def get_latest_features(n_lags: int = N_LAGS):
    """
    Build the feature row for the most recent available trading day.
    Fetches into a separate cache file so it never overwrites the
    fixed training dataset used to produce the reported test metrics.
    """
    today = date.today().strftime("%Y-%m-%d")

    df = get_data(start="2015-01-01", end=today, filename=LIVE_DATA_FILENAME)
    df = compute_log_returns(df)
    df = compute_volatility(df)
    df = create_features(df, n_lags)

    feature_cols = [f"vol_lag_{i}" for i in range(1, n_lags + 1)]
    df = df.dropna(subset=feature_cols)

    latest_row = df.iloc[[-1]]
    latest_date = latest_row.index[0]
    X_latest = latest_row[feature_cols].values

    return X_latest, latest_date


def predict_next_day_volatility():
    w, b, X_mean, X_std = load_trained_model()
    X_latest, latest_date = get_latest_features()

    X_latest_norm = (X_latest - X_mean) / X_std
    forecast = predict(X_latest_norm, w, b)

    return forecast[0], latest_date


if __name__ == "__main__":
    forecast, latest_date = predict_next_day_volatility()
    print(f"Most recent trading data as of: {latest_date.date()}")
    print(f"Predicted next-day (annualized) volatility: {forecast:.5f}")