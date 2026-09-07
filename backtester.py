
import pandas as pd
import numpy as np
import config as cfg


def _compute_trade_segments(df):
    """
    Breaks the POSITION series into discrete trade segments (a trade = a
    contiguous run of a non-zero position between two position changes) and
    computes the compounded net return for each segment. This gives us
    real trade-level stats (win rate, profit factor, avg win/loss) instead
    of only bar-level return stats.
    """
    segments = []
    pos = df["POSITION"].values
    ret = df["STRATEGY_RETURN_NET"].values
    n = len(df)

    seg_start = None
    seg_direction = 0
    seg_return = 1.0

    for i in range(n):
        current_pos = pos[i]
        if current_pos != 0 and seg_direction == 0:
            # starting a new trade
            seg_start = i
            seg_direction = current_pos
            seg_return = 1.0 + ret[i]
        elif current_pos == seg_direction and seg_direction != 0:
            # continuing the same trade
            seg_return *= (1.0 + ret[i])
        elif current_pos != seg_direction:
            # trade ended (flipped or went flat) - close out previous segment
            if seg_direction != 0:
                segments.append({
                    "start": df.index[seg_start],
                    "end": df.index[i - 1] if i > 0 else df.index[seg_start],
                    "direction": "LONG" if seg_direction > 0 else "SHORT",
                    "return_pct": (seg_return - 1.0) * 100,
                })
            if current_pos != 0:
                seg_start = i
                seg_direction = current_pos
                seg_return = 1.0 + ret[i]
            else:
                seg_direction = 0
                seg_return = 1.0

    # close any open trade at the end of the data
    if seg_direction != 0:
        segments.append({
            "start": df.index[seg_start],
            "end": df.index[-1],
            "direction": "LONG" if seg_direction > 0 else "SHORT",
            "return_pct": (seg_return - 1.0) * 100,
        })

    return pd.DataFrame(segments)


def _trade_stats_from_segments(trades_df):
    """Computes win rate, profit factor, avg win/loss, and max consecutive
    losing streak from a DataFrame of individual trade returns."""
    if trades_df.empty:
        return {
            "win_rate_pct": None,
            "profit_factor": None,
            "avg_win_pct": None,
            "avg_loss_pct": None,
            "avg_r_multiple": None,
            "max_consecutive_losses": 0,
            "num_completed_trades": 0,
        }

    wins = trades_df[trades_df["return_pct"] > 0]["return_pct"]
    losses = trades_df[trades_df["return_pct"] <= 0]["return_pct"]

    win_rate = (len(wins) / len(trades_df)) * 100
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else None)
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    avg_r_multiple = (abs(avg_win / avg_loss)) if avg_loss != 0 else None

    # longest streak of consecutive losing trades - the "can you survive this
    # psychologically" number that raw win-rate/return figures hide.
    max_streak, current_streak = 0, 0
    for r in trades_df["return_pct"]:
        if r <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return {
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2) if profit_factor not in (None, np.inf) else profit_factor,
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "avg_r_multiple": round(avg_r_multiple, 2) if avg_r_multiple is not None else None,
        "max_consecutive_losses": int(max_streak),
        "num_completed_trades": int(len(trades_df)),
    }


def _core_metrics(df):
    """Shared bar-level metrics (equity curve, Sharpe, drawdown, etc.) used
    by both the full-period backtest and each walk-forward slice."""
    total_return = df["EQUITY_CURVE"].iloc[-1] / df["EQUITY_CURVE"].iloc[0] - 1
    buy_hold_return = df["BUY_HOLD_EQUITY"].iloc[-1] / df["BUY_HOLD_EQUITY"].iloc[0] - 1
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

    trades_df = _compute_trade_segments(df)
    trade_stats = _trade_stats_from_segments(trades_df)

    return {
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(buy_hold_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "direction_accuracy_pct": round(direction_accuracy * 100, 2) if not np.isnan(direction_accuracy) else None,
        "num_trades": num_trades,
        **trade_stats,
        "equity_curve": df[["EQUITY_CURVE", "BUY_HOLD_EQUITY"]],
        "trades": trades_df,
        "df": df,
    }


def _prepare_positions(df_with_signals, allow_short=True):
    """Builds POSITION / returns / equity curves from raw signals. Pulled out
    as its own step so both the full backtest and walk-forward slices can
    reuse identical logic without duplicating it."""
    df = df_with_signals.copy()
    position_map = {"BUY": 1, "SELL": -1 if allow_short else 0, "HOLD": 0}
    df["POSITION"] = df["SIGNAL"].map(position_map).shift(1).fillna(0)
    df["STRATEGY_RETURN"] = df["POSITION"] * df["DAILY_RETURN"]
    df["POSITION_CHANGE"] = df["POSITION"].diff().abs().fillna(0)
    df["COST"] = df["POSITION_CHANGE"] * cfg.TRANSACTION_COST_PCT
    df["STRATEGY_RETURN_NET"] = df["STRATEGY_RETURN"] - df["COST"]
    df["EQUITY_CURVE"] = cfg.BACKTEST_CAPITAL * (1 + df["STRATEGY_RETURN_NET"]).cumprod()
    df["BUY_HOLD_EQUITY"] = cfg.BACKTEST_CAPITAL * (1 + df["DAILY_RETURN"]).cumprod()
    return df


def run_backtest(df_with_signals, allow_short=True):
    """Full-period backtest (unchanged behavior/signature from before), now
    additionally returns trade-level stats: win_rate_pct, profit_factor,
    avg_win_pct, avg_loss_pct, avg_r_multiple, max_consecutive_losses,
    num_completed_trades - alongside all the original bar-level metrics."""
    df = _prepare_positions(df_with_signals, allow_short=allow_short)
    return _core_metrics(df)


def walk_forward_backtest(df_with_signals, allow_short=True, train_frac=0.7):
    """
    Splits the loaded history chronologically into an earlier IN-SAMPLE
    window and a later OUT-OF-SAMPLE window, then runs the exact same
    backtest independently on each slice.

    IMPORTANT / HONEST LIMITATION: this app's signal rules (SMA/RSI/MACD/
    Supertrend thresholds in indicators.py & signal_engine.py) are fixed,
    not re-fitted per slice - so this is NOT a full parameter-refitting
    walk-forward optimization. What it DOES tell you is whether the
    strategy's edge is consistent across time or was a one-off historical
    fluke concentrated in a specific period - which is the single most
    common way retail backtests mislead people (looking great on the full
    history, but only because of one lucky stretch).
    """
    df = _prepare_positions(df_with_signals, allow_short=allow_short)

    n = len(df)
    split_idx = int(n * train_frac)
    split_idx = max(30, min(n - 30, split_idx))  # keep both slices non-trivial in size

    in_sample = df.iloc[:split_idx].copy()
    out_sample = df.iloc[split_idx:].copy()

    # Re-anchor each slice's equity curve to start at BACKTEST_CAPITAL so the
    # out-of-sample results aren't just a continuation of in-sample compounding.
    for slice_df in (in_sample, out_sample):
        slice_df["EQUITY_CURVE"] = cfg.BACKTEST_CAPITAL * (1 + slice_df["STRATEGY_RETURN_NET"]).cumprod()
        slice_df["BUY_HOLD_EQUITY"] = cfg.BACKTEST_CAPITAL * (1 + slice_df["DAILY_RETURN"]).cumprod()

    in_sample_metrics = _core_metrics(in_sample)
    out_sample_metrics = _core_metrics(out_sample)

    # A simple, honest verdict: does the strategy still beat buy & hold AND
    # stay profitable in the out-of-sample slice it was never "seen" tuning against?
    holds_up = (
        out_sample_metrics["total_return_pct"] > 0
        and out_sample_metrics["total_return_pct"] > out_sample_metrics["buy_hold_return_pct"]
    )

    return {
        "train_frac": train_frac,
        "in_sample": in_sample_metrics,
        "out_sample": out_sample_metrics,
        "holds_up_out_of_sample": holds_up,
        "split_date": df.index[split_idx],
    }
