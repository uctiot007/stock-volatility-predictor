import os
import pandas as pd
import yfinance as yf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def fetch_data(ticker: str = "^GSPC", start: str = "2015-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    assert data is not None, "yf.download returned None — check your internet connection or ticker symbol"

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    df = df.loc[:, required_cols].copy()
    df = df.dropna()

    return df


def save_data(df: pd.DataFrame, filename: str = "sp500_raw.csv") -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(path)
    print(f"Data saved to {path}")


def load_data(filename: str = "sp500_raw.csv") -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run get_data() first.")

    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_data(
    ticker: str = "^GSPC",
    start: str = "2015-01-01",
    end: str = "2023-01-01",
    filename: str = "sp500_raw.csv",
) -> pd.DataFrame:
    """
    Full pipeline: fetch -> clean -> save -> return.
    `filename` lets callers (like live prediction) cache to a separate
    file instead of overwriting the main training dataset.
    """
    df = fetch_data(ticker, start, end)
    df = clean_data(df)
    save_data(df, filename=filename)
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(df.shape)