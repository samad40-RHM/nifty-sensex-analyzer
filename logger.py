import pandas as pd
import os
from datetime import date

LOG_PATH = "signal_log.csv"


def log_daily_signal(ticker, latest_row):
    today = date.today().isoformat()
    new_row = {
        "date": today,
        "ticker": ticker,
        "close": latest_row["Close"],
        "rule_signal": latest_row["SIGNAL"],
        "supertrend_signal": latest_row["ST_SIGNAL"],
        "final_signal": latest_row["FINAL_SIGNAL"],
    }

    if os.path.exists(LOG_PATH):
        log_df = pd.read_csv(LOG_PATH)
    else:
        log_df = pd.DataFrame(columns=list(new_row.keys()))

    already_logged = ((log_df["date"] == today) & (log_df["ticker"] == ticker)).any() if len(log_df) else False

    if not already_logged:
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
        log_df.to_csv(LOG_PATH, index=False)

    return log_df


def compute_accuracy(log_df):
    if log_df is None or len(log_df) < 2:
        return None, 0

    df = log_df.sort_values(["ticker", "date"]).copy()
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["actual_move"] = df["next_close"] - df["close"]

    def is_correct(row):
        if pd.isna(row["next_close"]):
            return None
        if row["final_signal"] in ["STRONG BUY", "BUY"]:
            return row["actual_move"] > 0
        if row["final_signal"] in ["STRONG SELL", "SELL"]:
            return row["actual_move"] < 0
        return None

    df["correct"] = df.apply(is_correct, axis=1)
    valid = df.dropna(subset=["correct"])

    if len(valid) == 0:
        return None, 0

    accuracy = valid["correct"].mean() * 100
    return accuracy, len(valid)
