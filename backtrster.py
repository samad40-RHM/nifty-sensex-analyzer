import pandas as pd
import numpy as np
import config as cfg


def run_backtest(df_with_signals, allow_short=True):
    df = df_with_signals.copy()
    position_map = {"BUY": 1, "SELL": -1 if allow_short else 0, "HOLD": 0}
    df["POSITION"] = df["SIGNAL"].map(position_map).shift(1).fillna(0)
    df["STRATEGY_RETURN"] = df["POSITION"] * df["DAILY_RETURN"]
    df["POSITION_CHANGE"] = df["POSITION"].diff().abs().fillna(0)
    df["COST"] = df["POSITION_CHANGE"] * cfg.TRANSACTION_COST_PCT
    df["STRATEGY_RETURN_NET"] = df["STRATEGY_RETURN"] - df["COST"]
    df["EQUITY_CURVE"] = cfg.BACKTEST_CAPITAL * (1 + df["STRATEGY_RETURN_NET"]).cumprod()
    df["BUY_HOLD_EQUITY"] = cfg.BACKTEST_CAPITAL * (1 + df["DAILY_RETURN"]).cumprod()

    total_return = df["EQUITY_CURVE"].iloc[-1] / cfg.BACKTEST_CAPITAL - 1
    buy_hold_return = df["BUY_HOLD_EQUITY"].iloc[-1] / cfg.BACKTEST_CAPITAL - 1
    days = (df.index[-1] - df.index[0]).days
    years = max(days / 365.25, 1e-6)
    cagr = (1 + total_return) ** (1 / years) - 1
    daily_std = df["STRATEGY_RETURN_NET"].std()
    sharpe = (df["STRATEGY_RETURN_NET"].mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    correct = np.sign(df["POSITION"]) == np.sign(df["DAILY_RETURN"])
    traded_mask = df["POSITION"] != 0
    direction_accuracy = correct[traded_mask].mean() if traded_mask.sum() > 0 else np.nan
    running_max = df["EQUITY_CURVE"].cummax()
    max_drawdown = ((df["EQUITY_CURVE"] - running_max) / running_max).min()
    num_trades = int(df["POSITION_CHANGE"].sum())

    return {
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(buy_hold_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "direction_accuracy_pct": round(direction_accuracy * 100, 2) if not np.isnan(direction_accuracy) else None,
        "num_trades": num_trades,
        "equity_curve": df[["EQUITY_CURVE", "BUY_HOLD_EQUITY"]],
        "df": df,
    }
