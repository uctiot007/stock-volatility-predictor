import os
import pandas as pd
import yfinance as yf

# Project root = one level up from src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")


def fetch_data(ticker: str = "^GSPC", start: str = "2015-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    """
    Fetch historical stock/index data from Yahoo Finance.

    Parameters:
        ticker: Stock/index ticker
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        Raw downloaded data with a flat column index.
    """
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' between {start} and {end}.")

    # yfinance can return MultiIndex columns (e.g. ('Close', '^GSPC')) — flatten them
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw dataframe: keep required columns, drop missing values.
    """
    required_cols = ["Close", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    df = df[required_cols].copy()
    df = df.dropna()

    return df


def save_data(df: pd.DataFrame, filename: str = "sp500_raw.csv") -> None:
    """
    Save dataframe to data/raw directory (relative to project root).
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    path = os.path.join(RAW_DATA_DIR, filename)
    df.to_csv(path)
    print(f"Data saved to {path}")


def load_data(filename: str = "sp500_raw.csv") -> pd.DataFrame:
    """
    Load dataframe from data/raw directory (relative to project root).
    """
    path = os.path.join(RAW_DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. Run get_data() first.")

    return pd.read_csv(path, index_col=0, parse_dates=True)


def get_data(ticker: str = "^GSPC", start: str = "2015-01-01", end: str = "2023-01-01") -> pd.DataFrame:
    """
    Full pipeline: Fetch -> Clean -> Save -> Return.
    """
    df = fetch_data(ticker, start, end)
    df = clean_data(df)
    save_data(df)
    return df


if __name__ == "__main__":
    df = get_data()
    print(df.head())
    print(df.shape)