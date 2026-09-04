import yfinance as yf
import pandas as pd


def fetch_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index.name = "Date"
    return df.dropna()
