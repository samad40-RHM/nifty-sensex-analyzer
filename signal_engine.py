import pandas as pd
import config as cfg


def _trend_score(row):
    score = 0.0
    score += 0.5 if row["EMA_FAST"] > row["EMA_SLOW"] else -0.5
    score += 0.5 if row["SMA_FAST"] > row["SMA_SLOW"] else -0.5
    return score


def _momentum_score(row):
    rsi = row["RSI"]
    if rsi >= 70:
        return -1.0
    if rsi <= 30:
        return 1.0
    return (rsi - 50) / 66.7


def _macd_score(row):
    if row["MACD"] > row["MACD_SIGNAL"] and row["MACD_HIST"] > 0:
        return 1.0
    if row["MACD"] < row["MACD_SIGNAL"] and row["MACD_HIST"] < 0:
        return -1.0
    return 0.0


def _volatility_score(row):
    band_width = row["BB_UPPER"] - row["BB_LOWER"]
    if band_width == 0:
        return 0.0
    position = (row["Close"] - row["BB_LOWER"]) / band_width
    return (0.5 - position) * 2


def _volume_score(row):
    if row["VOL_SMA20"] == 0:
        return 0.0
    ratio = row["Volume"] / row["VOL_SMA20"]
    direction = 1 if row["DAILY_RETURN"] >= 0 else -1
    strength = min(max((ratio - 1), -1), 1)
    return direction * max(strength, 0)


def composite_score(row):
    w = cfg.WEIGHTS
    score = (w["trend"] * _trend_score(row) + w["momentum"] * _momentum_score(row) +
             w["macd"] * _macd_score(row) + w["volatility"] * _volatility_score(row) +
             w["volume"] * _volume_score(row))
    return max(min(score, 1.0), -1.0)


def decision_from_score(score):
    if score >= cfg.BUY_THRESHOLD:
        return "BUY"
    if score <= cfg.SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def annotate_signals(df):
    out = df.copy()
    out["SCORE"] = out.apply(composite_score, axis=1)
    out["SIGNAL"] = out["SCORE"].apply(decision_from_score)
    return out


def explain_latest(df_row):
    return {
        "trend": round(_trend_score(df_row), 3),
        "momentum": round(_momentum_score(df_row), 3),
        "macd": round(_macd_score(df_row), 3),
        "volatility": round(_volatility_score(df_row), 3),
        "volume": round(_volume_score(df_row), 3),
        "composite": round(composite_score(df_row), 3),
        "decision": decision_from_score(composite_score(df_row)),
    }
