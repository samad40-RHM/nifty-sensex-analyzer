import pandas as pd
import numpy as np
import config as cfg


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(series, period=20, num_std=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    return mid + num_std * std, mid, mid - num_std * std


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_all_indicators(df):
    out = df.copy()
    out["SMA_FAST"] = out["Close"].rolling(cfg.SMA_FAST).mean()
    out["SMA_SLOW"] = out["Close"].rolling(cfg.SMA_SLOW).mean()
    out["EMA_FAST"] = out["Close"].ewm(span=cfg.EMA_FAST, adjust=False).mean()
    out["EMA_SLOW"] = out["Close"].ewm(span=cfg.EMA_SLOW, adjust=False).mean()
    out["RSI"] = rsi(out["Close"], cfg.RSI_PERIOD)
    macd_line, signal_line, hist = macd(out["Close"], cfg.MACD_FAST, cfg.MACD_SLOW, cfg.MACD_SIGNAL)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = macd_line, signal_line, hist
    upper, mid, lower = bollinger_bands(out["Close"], cfg.BOLLINGER_PERIOD, cfg.BOLLINGER_STD)
    out["BB_UPPER"], out["BB_MID"], out["BB_LOWER"] = upper, mid, lower
    out["ATR"] = atr(out, cfg.ATR_PERIOD)
    out["VOL_SMA20"] = out["Volume"].rolling(20).mean()
    out["DAILY_RETURN"] = out["Close"].pct_change()
    return out.dropna()

def supertrend(df, period=10, multiplier=3):
    hl2 = (df["High"] + df["Low"]) / 2
    atr_val = atr(df, period)
    upperband = hl2 + multiplier * atr_val
    lowerband = hl2 - multiplier * atr_val
    close = df["Close"]

    final_upper = upperband.copy()
    final_lower = lowerband.copy()
    supertrend_line = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(len(df)):
        if i == 0:
            supertrend_line.iloc[i] = final_upper.iloc[i]
            direction.iloc[i] = 1
            continue
        if close.iloc[i-1] <= final_upper.iloc[i-1]:
            final_upper.iloc[i] = min(upperband.iloc[i], final_upper.iloc[i-1])
        else:
            final_upper.iloc[i] = upperband.iloc[i]

        if close.iloc[i-1] >= final_lower.iloc[i-1]:
            final_lower.iloc[i] = max(lowerband.iloc[i], final_lower.iloc[i-1])
        else:
            final_lower.iloc[i] = lowerband.iloc[i]

        if close.iloc[i] > final_upper.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < final_lower.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]

        supertrend_line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return supertrend_line, direction


def adx(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0

    tr = atr(df, 1) * 1  # true range per bar (period=1 gives raw TR via rolling mean of 1)
    atr_smooth = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr_smooth)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(period).mean()


def add_supertrend_adx(df, st_period=10, st_multiplier=3, adx_period=14):
    out = df.copy()
    st_line, st_dir = supertrend(out, st_period, st_multiplier)
    out["SUPERTREND"] = st_line
    out["SUPERTREND_DIR"] = st_dir
    out["ADX"] = adx(out, adx_period)
    return out.dropna()

