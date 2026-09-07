import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime
import urllib.request
import xml.etree.ElementTree as ET
import logger
import config as cfg
import data_fetcher
import indicators
import signal_engine
import backtester

st.set_page_config(page_title="Nifty & Sensex Daily Analyzer", layout="wide")

# ============ AUTO-REFRESH SETUP (for Live Market Snapshot) ============
AUTOREFRESH_AVAILABLE = False
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# 🌍 GLOBAL MARKET PULSE — overnight global cues that drive tomorrow's Indian
# market open (US close, Europe close, Asia live during IST morning, crude
# oil, dollar index/rupee) + a curated headline feed. Placed at the very top
# so it's the first thing you see on refresh, before anything else loads.
# ============================================================================

GLOBAL_TICKERS = {
    "🇺🇸 US": {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "Nasdaq": "^IXIC",
    },
    "🇪🇺 Europe": {
        "FTSE 100": "^FTSE",
        "DAX (Germany)": "^GDAXI",
        "CAC 40 (France)": "^FCHI",
    },
    "🌏 Asia": {
        "Nikkei 225 (Japan)": "^N225",
        "Hang Seng (HK)": "^HSI",
        "Shanghai Comp. (China)": "000001.SS",
    },
    "🛢️ Commodities & FX": {
        "Crude Oil (WTI)": "CL=F",
        "Dollar Index": "DX-Y.NYB",
        "USD/INR": "INR=X",
    },
}

@st.cache_data(ttl=240, show_spinner=False)
def fetch_global_snapshot():
    """Pulls latest close + % change for each global market/commodity that
    typically influences the next Indian trading session's opening gap."""
    results = {}
    for group, tickers in GLOBAL_TICKERS.items():
        group_results = {}
        for label, tk in tickers.items():
            try:
                data = yf.download(tk, period="5d", interval="1d", progress=False)
                if data is None or data.empty:
                    group_results[label] = None
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data = data.dropna(subset=["Close"])
                if len(data) < 2:
                    group_results[label] = None
                    continue
                last_close = float(data["Close"].iloc[-1])
                prev_close = float(data["Close"].iloc[-2])
                chg_pct = (last_close - prev_close) / prev_close * 100 if prev_close else 0
                last_date = data.index[-1]
                group_results[label] = {"price": last_close, "pct": chg_pct, "date": last_date}
            except Exception:
                group_results[label] = None
        results[group] = group_results
    return results


@st.cache_data(ttl=480, show_spinner=False)
def fetch_global_headlines(max_items=6):
    """Pulls top market-moving headlines from public, no-auth-required RSS
    feeds (Economic Times Markets + Reuters-style world/business feeds).
    Uses only Python's built-in urllib/ElementTree - no extra dependency
    needs to be added to requirements.txt."""
    feeds = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/business.xml",
    ]
    headlines = []
    for url in feeds:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                raw = resp.read()
            root = ET.fromstring(raw)
            for item in root.findall(".//item")[:max_items]:
                title_el = item.find("title")
                link_el = item.find("link")
                pub_el = item.find("pubDate")
                if title_el is not None and title_el.text:
                    headlines.append({
                        "title": title_el.text.strip(),
                        "link": link_el.text.strip() if link_el is not None and link_el.text else "",
                        "pub": pub_el.text.strip() if pub_el is not None and pub_el.text else "",
                    })
            if len(headlines) >= max_items:
                break
        except Exception:
            continue
    return headlines[:max_items]


gp_col1, gp_col2, gp_col3 = st.columns([2.4, 1, 1])
with gp_col1:
    st.markdown("## 🌍 Global Market Pulse — overnight cues for tomorrow's open")
with gp_col2:
    global_refresh_minutes = st.selectbox(
        "Auto-refresh", [5, 10], index=0,
        format_func=lambda m: f"every {m} min", key="global_refresh_interval",
        label_visibility="collapsed"
    )
with gp_col3:
    if st.button("🔄 Refresh Now", key="refresh_global_btn"):
        fetch_global_snapshot.clear()
        fetch_global_headlines.clear()

# Dedicated auto-refresh timer for this section, independent of the Live
# Market Snapshot's own refresh cadence - keeps global cues feeling "live"
# without forcing the whole page's other data (daily analysis, backtest) to
# re-run on a shorter cycle than needed.
if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=global_refresh_minutes * 60 * 1000, key="global_pulse_autorefresh")
else:
    st.markdown(
        f'<meta http-equiv="refresh" content="{global_refresh_minutes * 60}">',
        unsafe_allow_html=True
    )

global_data = fetch_global_snapshot()

# --- Compute an overall global sentiment score (how many markets are up vs down) ---
all_pct_changes = []
for group, items in global_data.items():
    if group == "🛢️ Commodities & FX":
        continue  # commodities/FX shown separately, not counted in equity sentiment
    for label, vals in items.items():
        if vals is not None:
            all_pct_changes.append(vals["pct"])

if all_pct_changes:
    up_count = sum(1 for v in all_pct_changes if v > 0)
    down_count = sum(1 for v in all_pct_changes if v < 0)
    total = len(all_pct_changes)
    avg_change = sum(all_pct_changes) / total
    if up_count > down_count and avg_change > 0.15:
        sentiment, sentiment_color, sentiment_icon = "BULLISH", "#0b8043", "🟢"
        sentiment_msg = "Global markets broadly UP — this typically supports a GAP-UP open for Nifty/Sensex."
    elif down_count > up_count and avg_change < -0.15:
        sentiment, sentiment_color, sentiment_icon = "BEARISH", "#c5221f", "🔴"
        sentiment_msg = "Global markets broadly DOWN — this typically pressures a GAP-DOWN open for Nifty/Sensex."
    else:
        sentiment, sentiment_color, sentiment_icon = "MIXED", "#e37400", "🟡"
        sentiment_msg = "Global markets are MIXED — no strong overnight directional cue either way."
else:
    sentiment, sentiment_color, sentiment_icon, sentiment_msg, avg_change, up_count, down_count, total = (
        "UNAVAILABLE", "#888888", "⚪", "Global data temporarily unavailable.", 0, 0, 0, 0
    )

st.markdown(
    f"""
    <div style='background:linear-gradient(90deg, {sentiment_color}22, {sentiment_color}05);
                border:2px solid {sentiment_color};border-radius:14px;padding:16px 20px;margin-bottom:12px'>
        <h2 style='margin:0;color:{sentiment_color}'>{sentiment_icon} Global Sentiment: {sentiment}</h2>
        <p style='margin:6px 0 0 0;color:#333;font-size:14.5px'>{sentiment_msg}</p>
        <p style='margin:4px 0 0 0;color:#666;font-size:12.5px'>
            {up_count} up / {down_count} down out of {total} major global indices tracked &nbsp;|&nbsp;
            Average move: <b>{avg_change:+.2f}%</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Color-coded cards, grouped by region ---
group_cols = st.columns(len(GLOBAL_TICKERS))
for gcol, (group_name, items) in zip(group_cols, global_data.items()):
    with gcol:
        st.markdown(f"**{group_name}**")
        for label, vals in items.items():
            if vals is None:
                st.markdown(
                    f"<div style='padding:6px 8px;color:#999;font-size:12.5px'>{label}: data unavailable</div>",
                    unsafe_allow_html=True
                )
                continue
            pct = vals["pct"]
            is_fx_or_commodity = group_name == "🛢️ Commodities & FX"
            # For USD/INR, a RISING rupee value (i.e. Rupee weakening) is a mild negative
            # cue for Indian equities (imported inflation, FII outflows) - flag with a note.
            up_is_good = not (label == "USD/INR")
            positive = pct > 0
            good = positive if up_is_good else (not positive)
            bg = "#e6f4ea" if good else ("#fdecea" if pct != 0 else "#f5f5f5")
            fg = "#0b8043" if good else ("#c5221f" if pct != 0 else "#888888")
            arrow = "▲" if positive else ("▼" if pct < 0 else "—")
            st.markdown(
                f"""
                <div style='background-color:{bg};border-radius:8px;padding:8px 10px;margin-bottom:6px'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                        <span style='font-size:12.5px;color:#333'>{label}</span>
                        <span style='font-size:12.5px;font-weight:bold;color:{fg}'>{arrow} {pct:+.2f}%</span>
                    </div>
                    <div style='font-size:11px;color:#777'>{vals['price']:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

now_ist = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
st.caption(
    f"🔁 Auto-refreshing every {global_refresh_minutes} min — last checked {now_ist.strftime('%d-%b-%Y %H:%M:%S')} IST. "
    "ℹ️ US & Europe show their LAST CLOSED session (overnight for India). Asia shows the LATEST available "
    "session (may be live-trading during Indian morning hours). USD/INR rising = rupee weakening, which is "
    "a mild negative cue for Indian equities. Data via Yahoo Finance, ~15-20 min delayed."
)

# --- Curated headlines panel ---
with st.expander("📰 Key Global Headlines (tap to expand)", expanded=False):
    headlines = fetch_global_headlines()
    if not headlines:
        st.info("Headlines feed temporarily unavailable. Check back after refreshing.")
    else:
        for h in headlines:
            if h["link"]:
                st.markdown(f"🔹 [{h['title']}]({h['link']})  \n<span style='color:#999;font-size:11px'>{h['pub']}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"🔹 {h['title']}")
    st.caption("Headlines are pulled from public market-news RSS feeds and are NOT curated or verified by this app for accuracy.")

st.markdown("---")


st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1.5rem; padding-right: 1.5rem;}
        [data-testid="stMetricValue"] {font-size: 1.1rem;}

        @media (max-width: 768px) {
            .block-container {padding-left: 0.6rem; padding-right: 0.6rem; padding-top: 0.5rem;}
            div[data-testid="stHorizontalBlock"] {
                flex-direction: column !important;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                width: 100% !important;
                min-width: 100% !important;
                margin-bottom: 6px;
            }
            [data-testid="stMetricValue"] {font-size: 1.3rem;}
            [data-testid="stMetricLabel"] {font-size: 0.8rem;}
            h1 {font-size: 1.4rem !important;}
            h2 {font-size: 1.15rem !important;}
            h3 {font-size: 1.0rem !important;}
            div[style*="display:flex"] {
                flex-wrap: wrap !important;
                row-gap: 6px;
            }
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                overflow-x: auto !important;
            }
            div[style*="border-radius:10px"] h2 {font-size: 1.05rem !important;}
        }
    </style>
""", unsafe_allow_html=True)

st.info("📱 Tip: On mobile, tap the **>›** arrow at the top-left to open Settings (index choice, chart filters, position sizing).", icon="📱")

@st.cache_data(ttl=30, show_spinner=False)
def fetch_live_quote(yf_ticker):
    try:
        data = yf.download(yf_ticker, period="1d", interval="1m", progress=False)
        if data is None or data.empty:
            data = yf.download(yf_ticker, period="5d", interval="5m", progress=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna(subset=["Close"])
        if data.empty:
            return None
        last_price = float(data["Close"].iloc[-1])
        day_high = float(data["High"].max())
        day_low = float(data["Low"].min())
        day_open = float(data["Open"].iloc[0])
        change = last_price - day_open
        pct_change = (change / day_open) * 100 if day_open else 0
        last_time = data.index[-1]
        return {
            "price": last_price, "change": change, "pct_change": pct_change,
            "high": day_high, "low": day_low, "time": last_time
        }
    except Exception:
        return None


refresh_col1, refresh_col2 = st.columns([3, 1])
with refresh_col1:
    st.markdown("## 📡 Live Market Snapshot")
with refresh_col2:
    refresh_seconds = st.selectbox(
        "Auto-refresh every", [60, 90, 120], index=1,
        format_func=lambda s: f"{s}s", key="refresh_interval",
        label_visibility="collapsed"
    )

if AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=refresh_seconds * 1000, key="live_kpi_autorefresh")
    st.caption(f"🔁 Auto-refreshing every {refresh_seconds}s — no action needed.")
else:
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_seconds}">',
        unsafe_allow_html=True
    )
    st.caption(
        f"🔁 Auto-refreshing every {refresh_seconds}s (fallback mode — full page reload). "
        f"For a smoother experience without page reloads, add `streamlit-autorefresh` to requirements.txt."
    )

lk1, lk2, lk3 = st.columns([1, 1, 0.6])
with lk3:
    if st.button("🔄 Refresh Now"):
        fetch_live_quote.clear()

live_cols = st.columns(len(cfg.INDICES))
latest_update_time = None

for col, (idx_name, idx_ticker) in zip(live_cols, cfg.INDICES.items()):
    quote = fetch_live_quote(idx_ticker)
    with col:
        if quote is None:
            st.warning(f"{idx_name}: live quote unavailable right now.")
            continue
        arrow = "🟢▲" if quote["change"] >= 0 else "🔴▼"
        card_color = "#e6f4ea" if quote["change"] >= 0 else "#fdecea"
        text_color = "#0b8043" if quote["change"] >= 0 else "#c5221f"
        st.markdown(
            f"""
            <div style='background-color:{card_color};border-radius:10px;padding:16px;text-align:center;border:1px solid #ddd'>
                <h3 style='margin:0;color:#333'>{idx_name}</h3>
                <h1 style='margin:4px 0;color:{text_color};font-size:2.2rem'>{quote['price']:,.2f}</h1>
                <p style='margin:0;color:{text_color};font-weight:bold;font-size:1.05rem'>
                    {arrow} {quote['change']:+,.2f} ({quote['pct_change']:+.2f}%)
                </p>
                <p style='margin:6px 0 0 0;color:#666;font-size:12px'>
                    Day H: {quote['high']:,.2f} &nbsp;|&nbsp; Day L: {quote['low']:,.2f}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if latest_update_time is None or quote["time"] > latest_update_time:
            latest_update_time = quote["time"]

if latest_update_time is not None:
    try:
        display_time = latest_update_time.tz_convert("Asia/Kolkata").strftime("%d-%b-%Y %H:%M:%S IST")
    except Exception:
        display_time = str(latest_update_time)
    st.caption(
        f"⏱️ Last updated: {display_time}. Refreshes automatically every ~30s on page interaction, "
        f"or click '🔄 Refresh Live Prices' to force an update. Note: Yahoo Finance index data is typically "
        f"delayed a few minutes and updates only during NSE/BSE market hours (9:15 AM–3:30 PM IST, Mon–Fri) — "
        f"this is NOT a paid real-time feed."
    )

st.markdown("---")

st.title("Nifty & Sensex Daily Analyzer")
st.caption("Educational tool only - not financial advice. Manage your own risk.")

with st.sidebar:
    st.header("Settings")
    index_name = st.selectbox("Index", list(cfg.INDICES.keys()))
    period = st.selectbox("History window", ["1y", "2y", "5y", "10y", "max"], index=2)
    allow_short = st.checkbox("Allow SELL signals in backtest", value=True)
    run_button = st.button("Fetch data & analyze", type="primary")

    st.markdown("---")
    st.header("🔍 Chart Filters")
    timeframe = st.selectbox(
        "Candle Resolution (Swing chart)",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Applies to the 1M/3M/6M/9M/1Y/All buttons. 1D/5D/1W below use real intraday data instead."
    )
    show_sma = st.checkbox("Show SMA lines", value=True)
    show_supertrend = st.checkbox("Show Supertrend line", value=True)
    show_rule_signals = st.checkbox("Show rule-engine BUY/SELL markers (small)", value=False)
    show_final_signals = st.checkbox("Show STRONG BUY/SELL (final) markers", value=True)

    st.markdown("---")
    fullscreen_mode = st.checkbox(
        "🖥️ Full-Screen Chart Mode",
        value=False,
        help="Hides everything except the chart so you can analyze it in maximum space."
    )
    compact_mobile = st.checkbox(
        "📱 Compact Mobile View",
        value=False,
        help="Shrinks the chart height and simplifies spacing for small phone screens. Turn this on if you're viewing on a mobile device."
    )

    st.markdown("---")
    st.header("💰 Position Sizing")
    capital = st.number_input("Capital (₹)", min_value=1000, value=100000, step=1000)
    risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)

ticker = cfg.INDICES[index_name]

@st.cache_data(ttl=3600, show_spinner=False)
def load_and_process(ticker, period):
    raw = data_fetcher.fetch_history(ticker, period, cfg.INTERVAL)
    enriched = indicators.add_all_indicators(raw)
    enriched = indicators.add_supertrend_adx(enriched)
    scored = signal_engine.annotate_signals(enriched)
    scored["ST_SIGNAL"] = scored.apply(signal_engine.supertrend_adx_signal, axis=1)

    def combined(r):
        if r["SIGNAL"] == "BUY" and r["ST_SIGNAL"] == "BUY":
            return "STRONG BUY"
        if r["SIGNAL"] == "SELL" and r["ST_SIGNAL"] == "SELL":
            return "STRONG SELL"
        if r["SIGNAL"] == "HOLD" and r["ST_SIGNAL"] == "HOLD":
            return "HOLD"
        return "MIXED / CAUTION"

    scored["FINAL_SIGNAL"] = scored.apply(combined, axis=1)
    return scored


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday(ticker, yf_period, yf_interval):
    try:
        data = yf.download(ticker, period=yf_period, interval=yf_interval, progress=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna(subset=["Close"])
        if data.empty or len(data) < 20:
            return data if not data.empty else None

        try:
            enriched = indicators.add_all_indicators(data)
            enriched = indicators.add_supertrend_adx(enriched)
            scored = signal_engine.annotate_signals(enriched)
            scored["ST_SIGNAL"] = scored.apply(signal_engine.supertrend_adx_signal, axis=1)

            def combined(r):
                if r["SIGNAL"] == "BUY" and r["ST_SIGNAL"] == "BUY":
                    return "STRONG BUY"
                if r["SIGNAL"] == "SELL" and r["ST_SIGNAL"] == "SELL":
                    return "STRONG SELL"
                if r["SIGNAL"] == "HOLD" and r["ST_SIGNAL"] == "HOLD":
                    return "HOLD"
                return "MIXED / CAUTION"

            scored["FINAL_SIGNAL"] = scored.apply(combined, axis=1)
            return scored
        except Exception:
            return data
    except Exception:
        return None


def resample_ohlc(data, tf):
    if tf == "Daily":
        return data
    rule = "W" if tf == "Weekly" else "M"
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    other_cols = [c for c in data.columns if c not in agg]
    for c in other_cols:
        agg[c] = "last"
    resampled = data.resample(rule).agg(agg).dropna(subset=["Close"])
    return resampled


def compute_atr(data, period=14):
    high = data["High"]; low = data["Low"]; close = data["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def compute_confidence(latest_row, explain_dict):
    directions = []
    for key in ["trend", "momentum", "macd", "volatility", "volume"]:
        val = explain_dict.get(key, 0)
        directions.append(1 if val > 0 else (-1 if val < 0 else 0))
    st_signal = latest_row.get("ST_SIGNAL", "HOLD")
    directions.append(1 if st_signal == "BUY" else (-1 if st_signal == "SELL" else 0))

    total = len(directions)
    bullish = sum(1 for d in directions if d == 1)
    bearish = sum(1 for d in directions if d == -1)
    dominant = max(bullish, bearish)
    base_confidence = (dominant / total) * 100 if total else 0

    adx = latest_row.get("ADX", 20)
    if adx >= 25:
        adx_adjustment, strength_label = 10, "strong trend"
    elif adx < 15:
        adx_adjustment, strength_label = -10, "weak/choppy trend"
    else:
        adx_adjustment, strength_label = 0, "moderate trend"

    confidence = max(0, min(100, base_confidence + adx_adjustment))
    direction = "BULLISH" if bullish > bearish else ("BEARISH" if bearish > bullish else "NEUTRAL")
    return round(confidence), direction, dominant, total, strength_label


def compute_trade_plan(latest_row, final_signal_val, atr_val):
    entry = latest_row["Close"]
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = entry * 0.01
    supertrend_val = latest_row.get("SUPERTREND", np.nan)

    if final_signal_val == "STRONG BUY":
        stop_loss = entry - 1.5 * atr_val
        if not pd.isna(supertrend_val) and supertrend_val < entry:
            stop_loss = max(stop_loss, supertrend_val)
        target = entry + 3 * atr_val
        risk, reward, direction = entry - stop_loss, target - entry, "LONG"
    elif final_signal_val == "STRONG SELL":
        stop_loss = entry + 1.5 * atr_val
        if not pd.isna(supertrend_val) and supertrend_val > entry:
            stop_loss = min(stop_loss, supertrend_val)
        target = entry - 3 * atr_val
        risk, reward, direction = stop_loss - entry, entry - target, "SHORT"
    else:
        return None

    risk = max(risk, 0.01)
    return {
        "direction": direction, "entry": entry, "stop_loss": stop_loss, "target": target,
        "risk_pts": risk, "reward_pts": reward,
        "risk_pct": (risk / entry) * 100, "reward_pct": (reward / entry) * 100,
        "rr_ratio": reward / risk,
    }


def compute_btst_signal(full_df, atr_val):
    """
    BTST = Buy Today, Sell Tomorrow (NSE T+1 settlement overnight strategy).
    Returns a CALL / PUT / WAIT verdict (options-style directional wording) plus
    numeric predicted levels for tomorrow, and a historical win-rate backtested
    on the user's own loaded data for the strict "strong setup" rule.
    """
    d = full_df.copy()
    d["DAY_RANGE"] = (d["High"] - d["Low"]).replace(0, np.nan)
    d["CLOSE_POSITION"] = ((d["Close"] - d["Low"]) / d["DAY_RANGE"]).clip(0, 1)

    def rule_fires_bull(row):
        try:
            return (
                row.get("FINAL_SIGNAL") == "STRONG BUY"
                and row.get("CLOSE_POSITION", 0) >= 0.65
                and row.get("ADX", 0) >= 20
                and 50 <= row.get("RSI", 50) <= 75
            )
        except Exception:
            return False

    def rule_fires_bear(row):
        try:
            return (
                row.get("FINAL_SIGNAL") == "STRONG SELL"
                and row.get("CLOSE_POSITION", 1) <= 0.35
                and row.get("ADX", 0) >= 20
                and 25 <= row.get("RSI", 50) <= 50
            )
        except Exception:
            return False

    d["BTST_BULL_FIRED"] = d.apply(rule_fires_bull, axis=1)
    d["BTST_BEAR_FIRED"] = d.apply(rule_fires_bear, axis=1)

    latest_row = d.iloc[-1]
    bull_today = bool(latest_row["BTST_BULL_FIRED"])
    bear_today = bool(latest_row["BTST_BEAR_FIRED"])

    if bull_today:
        call = "CALL"
        fired_mask = d["BTST_BULL_FIRED"]
    elif bear_today:
        call = "PUT"
        fired_mask = d["BTST_BEAR_FIRED"]
    else:
        call = "WAIT"
        fired_mask = pd.Series(False, index=d.index)

    fired_idx = d.index[fired_mask]
    wins_open, wins_close, total_checked = 0, 0, 0
    for ts in fired_idx:
        loc = d.index.get_loc(ts)
        if loc + 1 >= len(d):
            continue
        today_close = d.iloc[loc]["Close"]
        next_open = d.iloc[loc + 1]["Open"]
        next_close = d.iloc[loc + 1]["Close"]
        total_checked += 1
        is_bull = d.iloc[loc]["BTST_BULL_FIRED"]
        if is_bull:
            if next_open > today_close:
                wins_open += 1
            if next_close > today_close:
                wins_close += 1
        else:
            if next_open < today_close:
                wins_open += 1
            if next_close < today_close:
                wins_close += 1

    win_rate_open = (wins_open / total_checked * 100) if total_checked else None
    win_rate_close = (wins_close / total_checked * 100) if total_checked else None

    entry = float(latest_row["Close"])
    if pd.isna(atr_val) or atr_val <= 0:
        atr_val = entry * 0.008
    move = max(atr_val * 0.8, entry * 0.004)

    if call == "CALL":
        pred_target = entry + move
        pred_stop = entry - move * 0.5
    elif call == "PUT":
        pred_target = entry - move
        pred_stop = entry + move * 0.5
    else:
        pred_target = entry
        pred_stop = entry

    reasons = []
    if call == "CALL":
        reasons = [
            ("STRONG BUY signal today", True),
            ("Strong close (upper 35% of range)", latest_row.get("CLOSE_POSITION", 0) >= 0.65),
            ("ADX >= 20 (real trend)", latest_row.get("ADX", 0) >= 20),
            ("RSI healthy 50-75", 50 <= latest_row.get("RSI", 50) <= 75),
        ]
    elif call == "PUT":
        reasons = [
            ("STRONG SELL signal today", True),
            ("Weak close (lower 35% of range)", latest_row.get("CLOSE_POSITION", 1) <= 0.35),
            ("ADX >= 20 (real trend)", latest_row.get("ADX", 0) >= 20),
            ("RSI weak 25-50", 25 <= latest_row.get("RSI", 50) <= 50),
        ]
    else:
        reasons = [
            ("STRONG BUY signal today", latest_row.get("FINAL_SIGNAL") == "STRONG BUY"),
            ("STRONG SELL signal today", latest_row.get("FINAL_SIGNAL") == "STRONG SELL"),
            ("ADX >= 20 (real trend)", latest_row.get("ADX", 0) >= 20),
        ]

    return {
        "call": call,
        "entry": entry,
        "pred_target": pred_target,
        "pred_stop": pred_stop,
        "total_checked": total_checked,
        "win_rate_open": win_rate_open,
        "win_rate_close": win_rate_close,
        "reasons": reasons,
        "close_position": latest_row.get("CLOSE_POSITION", np.nan),
        "adx": latest_row.get("ADX", np.nan),
        "rsi": latest_row.get("RSI", np.nan),
    }


if run_button or "last_df" not in st.session_state:
    try:
        st.session_state["last_df"] = load_and_process(ticker, period)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

full_df = st.session_state.get("last_df")
if full_df is None:
    st.info("Click 'Fetch data & analyze' to begin.")
    st.stop()

full_df["ATR14"] = compute_atr(full_df, period=14)
swing_df = resample_ohlc(full_df, timeframe)

if swing_df.empty or len(swing_df) < 2:
    st.warning("Not enough data for this timeframe. Try Daily view or a longer History window.")
    st.stop()

latest = full_df.iloc[-1]
explain = signal_engine.explain_latest(latest)
final_signal = latest["FINAL_SIGNAL"]
final_color = {"STRONG BUY": "darkgreen", "STRONG SELL": "darkred", "HOLD": "gray", "MIXED / CAUTION": "orange"}[final_signal]
confidence, conf_direction, agree_count, total_count, strength_label = compute_confidence(latest, explain)

st.markdown(
    f"<div style='background-color:{final_color};padding:14px;border-radius:10px;text-align:center;margin-bottom:6px'>"
    f"<h2 style='color:white;margin:0'>Today's Signal: {final_signal}  |  {index_name} @ {latest['Close']:,.2f}</h2>"
    f"<p style='color:white;margin:4px 0 0 0;font-size:15px'>Confidence: <b>{confidence}%</b> "
    f"({agree_count}/{total_count} signals aligned {conf_direction.lower()}, {strength_label}, ADX {latest['ADX']:.1f})</p>"
    f"</div>",
    unsafe_allow_html=True
)

conf_bar_color = "#00c853" if confidence >= 70 else ("#ffab00" if confidence >= 50 else "#d50000")
st.markdown(
    f"""
    <div style='background-color:#e0e0e0;border-radius:8px;height:18px;width:100%;margin-bottom:14px'>
        <div style='background-color:{conf_bar_color};width:{confidence}%;height:18px;border-radius:8px;
                    text-align:right;color:white;font-size:11px;padding-right:6px;line-height:18px'>
            {confidence}%
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

trade_plan = compute_trade_plan(latest, final_signal, latest["ATR14"])

st.subheader("🎯 Trade Plan")
if trade_plan is None:
    st.info(f"No actionable trade plan for '{final_signal}' — plans are only generated for STRONG BUY / STRONG SELL signals.")
else:
    risk_amount = capital * (risk_pct / 100.0)
    position_size = int(risk_amount / trade_plan["risk_pts"]) if trade_plan["risk_pts"] > 0 else 0
    is_long = trade_plan["direction"] == "LONG"

    card_bg = "#eafaf1" if is_long else "#fdf1ee"
    card_border = "#2e7d32" if is_long else "#c0392b"
    dir_color = "#1b7a3d" if is_long else "#c0392b"
    dir_icon = "📈" if is_long else "📉"

    action_word = "buying (going long)" if is_long else "shorting (going short)"
    summary_sentence = (
        f"The system currently believes <b>{index_name}</b> is heading "
        f"{'up' if is_long else 'down'}. If you were {action_word} at "
        f"<b>{trade_plan['entry']:,.2f}</b>, this plan caps your loss if price "
        f"{'falls' if is_long else 'rises'} to <b>{trade_plan['stop_loss']:,.2f}</b>, and books profit if "
        f"price {'rises' if is_long else 'falls'} to <b>{trade_plan['target']:,.2f}</b> — sizing at "
        f"<b>{position_size:,} units</b> keeps your worst-case loss near your intended "
        f"{risk_pct}% of ₹{capital:,.0f}."
    )

    st.markdown(
        f"""
        <div style='background-color:{card_bg};border:1.5px solid {card_border};border-radius:12px;padding:16px 18px;margin-bottom:10px'>
            <p style='margin:0 0 10px 0;color:#333;font-size:14.5px;line-height:1.5'>{dir_icon} {summary_sentence}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tp1, tp2, tp3, tp4, tp5 = st.columns(5)
    tp1.markdown(f"<p style='margin:0;color:gray;font-size:13px'>Direction</p><h3 style='margin:0;color:{dir_color}'>{dir_icon} {trade_plan['direction']}</h3>", unsafe_allow_html=True)
    tp2.metric("Entry", f"{trade_plan['entry']:,.2f}")
    tp3.metric("Stop-Loss", f"{trade_plan['stop_loss']:,.2f}", f"-{trade_plan['risk_pct']:.2f}%")
    tp4.metric("Target", f"{trade_plan['target']:,.2f}", f"+{trade_plan['reward_pct']:.2f}%")
    tp5.metric("Risk : Reward", f"1 : {trade_plan['rr_ratio']:.1f}")
    tp6, tp7 = st.columns(2)
    tp6.metric("Risk per unit", f"{trade_plan['risk_pts']:,.2f} pts")
    tp7.metric(f"Suggested Position Size (risking {risk_pct}% of ₹{capital:,.0f})", f"{position_size:,} units")

    if is_long:
        low_pt, mid_pt, high_pt = trade_plan["stop_loss"], trade_plan["entry"], trade_plan["target"]
        low_label, high_label = "Stop-Loss", "Target"
        low_color, high_color = "#e74c3c", "#27ae60"
    else:
        low_pt, mid_pt, high_pt = trade_plan["target"], trade_plan["entry"], trade_plan["stop_loss"]
        low_label, high_label = "Target", "Stop-Loss"
        low_color, high_color = "#27ae60", "#e74c3c"

    span = high_pt - low_pt if high_pt != low_pt else 1
    mid_pct = max(0, min(100, (mid_pt - low_pt) / span * 100))

    st.markdown(
        f"""
        <div style='margin:6px 0 4px 0'>
          <div style='display:flex;justify-content:space-between;font-size:12px;color:#555;margin-bottom:3px'>
            <span><b style='color:{low_color}'>{low_label}</b> {low_pt:,.2f}</span>
            <span><b>Entry</b> {mid_pt:,.2f}</span>
            <span><b style='color:{high_color}'>{high_label}</b> {high_pt:,.2f}</span>
          </div>
          <div style='position:relative;height:14px;border-radius:7px;
                      background:linear-gradient(to right, {low_color} 0%, {low_color} {mid_pct}%, {high_color} {mid_pct}%, {high_color} 100%);
                      opacity:0.85'>
            <div style='position:absolute;left:{mid_pct}%;top:-4px;transform:translateX(-50%);
                        width:2px;height:22px;background:#222'></div>
          </div>
          <div style='display:flex;justify-content:space-between;font-size:11px;color:#888;margin-top:2px'>
            <span>← higher risk</span><span>1 : {trade_plan['rr_ratio']:.1f} risk-reward</span><span>higher reward →</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        f"Stop-loss = 1.5× ATR({latest['ATR14']:.1f}) from entry (tightened to Supertrend if closer). "
        f"Target = 3× ATR (built-in 1:2 risk-reward). Not investment advice."
    )

# ============ 🌙 BTST SIGNAL — BOTH NIFTY & SENSEX SIDE BY SIDE ============
st.subheader("🌙 BTST Signal for Tomorrow — Nifty & Sensex")
st.caption("Buy Today, Sell Tomorrow. Short, numeric view for BOTH indices — with a clear CALL / PUT / WAIT verdict.")

@st.cache_data(ttl=3600, show_spinner=False)
def load_for_btst(ticker_, period_):
    raw = data_fetcher.fetch_history(ticker_, period_, cfg.INTERVAL)
    enriched = indicators.add_all_indicators(raw)
    enriched = indicators.add_supertrend_adx(enriched)
    scored = signal_engine.annotate_signals(enriched)
    scored["ST_SIGNAL"] = scored.apply(signal_engine.supertrend_adx_signal, axis=1)

    def combined(r):
        if r["SIGNAL"] == "BUY" and r["ST_SIGNAL"] == "BUY":
            return "STRONG BUY"
        if r["SIGNAL"] == "SELL" and r["ST_SIGNAL"] == "SELL":
            return "STRONG SELL"
        if r["SIGNAL"] == "HOLD" and r["ST_SIGNAL"] == "HOLD":
            return "HOLD"
        return "MIXED / CAUTION"

    scored["FINAL_SIGNAL"] = scored.apply(combined, axis=1)
    scored["ATR14"] = compute_atr(scored, period=14)
    return scored

btst_cols = st.columns(len(cfg.INDICES))
for col, (idx_nm, idx_tk) in zip(btst_cols, cfg.INDICES.items()):
    with col:
        try:
            if idx_nm == index_name:
                df_for_btst = full_df  # reuse already-loaded data, avoid a second fetch
            else:
                df_for_btst = load_for_btst(idx_tk, period)
            atr_latest = df_for_btst["ATR14"].iloc[-1]
            btst = compute_btst_signal(df_for_btst, atr_latest)
        except Exception as e:
            st.warning(f"{idx_nm}: BTST unavailable ({e})")
            continue

        call = btst["call"]
        if call == "CALL":
            bg, border, txt_color, icon = "#eafaf1", "#2e7d32", "#1b7a3d", "📈"
            headline = "CALL (bullish overnight)"
        elif call == "PUT":
            bg, border, txt_color, icon = "#fdf1ee", "#c0392b", "#c0392b", "📉"
            headline = "PUT (bearish overnight)"
        else:
            bg, border, txt_color, icon = "#f5f5f5", "#bbbbbb", "#666666", "⏸️"
            headline = "WAIT (no clear edge)"

        win_open = f"{btst['win_rate_open']:.0f}%" if btst["win_rate_open"] is not None else "N/A"
        win_close = f"{btst['win_rate_close']:.0f}%" if btst["win_rate_close"] is not None else "N/A"

        st.markdown(
            f"""
            <div style='background-color:{bg};border:1.5px solid {border};border-radius:12px;padding:14px 16px'>
                <h4 style='margin:0;color:#333'>{idx_nm}</h4>
                <h2 style='margin:4px 0;color:{txt_color}'>{icon} {headline}</h2>
                <table style='width:100%;font-size:13.5px;color:#333;border-collapse:collapse'>
                    <tr><td style='padding:3px 0;color:#777'>Today's Close</td><td style='text-align:right;font-weight:bold'>{btst['entry']:,.2f}</td></tr>
                    <tr><td style='padding:3px 0;color:#777'>Predicted Tomorrow Target</td><td style='text-align:right;font-weight:bold;color:{txt_color}'>{btst['pred_target']:,.2f}</td></tr>
                    <tr><td style='padding:3px 0;color:#777'>Predicted Tomorrow Stop</td><td style='text-align:right;font-weight:bold'>{btst['pred_stop']:,.2f}</td></tr>
                    <tr><td style='padding:6px 0 0 0;color:#777'>Win-rate (next OPEN)</td><td style='text-align:right;padding-top:6px'>{win_open}</td></tr>
                    <tr><td style='padding:3px 0;color:#777'>Win-rate (next CLOSE)</td><td style='text-align:right'>{win_close}</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander(f"Why this {call} call for {idx_nm}?"):
            for label, passed in btst["reasons"]:
                st.markdown(f"{'✅' if passed else '❌'} {label}")
            st.caption(f"Backtested on {btst['total_checked']} similar past setups in loaded history.")

st.caption(
    "⚠️ CALL = bullish overnight bias, PUT = bearish overnight bias, WAIT = no clear edge today. "
    "Predicted Target/Stop are ATR-based overnight estimates, not guarantees. BTST carries real gap risk "
    "from global cues (SGX/GIFT Nifty) before the next session opens. Win-rates are historical backtests on "
    "your own loaded data, not a live-verified record. Use small size, set your stop before market open. "
    "Not investment advice."
)

# ============ CHART RANGE SELECTOR (drives which data source is used) ============
st.markdown("### 📊 Chart")
range_choice = st.radio(
    "Select range",
    ["1D", "5D", "1W", "1M", "3M", "6M", "9M", "1Y", "All"],
    index=0, horizontal=True,
    help="1D/5D/1W load REAL intraday candles (minute-level). 1M and beyond use daily candles from your swing data. "
         "Defaults to 1D on page load so the chart isn't cluttered with years of candles."
)

intraday_map = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "15m"),
    "1W": ("5d", "30m"),
}

chart_df = None
is_intraday = False
data_note = ""

if range_choice in intraday_map:
    yf_period, yf_interval = intraday_map[range_choice]
    intraday_df = fetch_intraday(ticker, yf_period, yf_interval)
    if intraday_df is not None and len(intraday_df) >= 3:
        chart_df = intraday_df
        is_intraday = True
        data_note = f"Showing REAL intraday candles ({yf_interval} interval, last {yf_period}) — like a live NSE chart."
    else:
        chart_df = swing_df.tail(10)
        data_note = "⚠️ Intraday data unavailable right now (market closed / feed limit) — showing last 10 daily candles instead."
else:
    n = len(swing_df)
    rows_map = {"1M": 22, "3M": 66, "6M": 132, "9M": 198, "1Y": 264, "All": n}
    rows = rows_map.get(range_choice, n)
    chart_df = swing_df.tail(rows)
    data_note = f"Showing daily candles ({timeframe} resolution), last {range_choice}."

st.caption(f"ℹ️ {data_note}")

o_val = float(chart_df["Open"].iloc[0])
h_val = float(chart_df["High"].max())
l_val = float(chart_df["Low"].min())
c_val = float(chart_df["Close"].iloc[-1])
chg_val = c_val - o_val
chg_pct = (chg_val / o_val * 100) if o_val else 0
quote_color = "#089981" if chg_val >= 0 else "#f23645"

st.markdown(
    f"""
    <div style='display:flex;align-items:center;gap:18px;padding:6px 4px;font-family:monospace;font-size:14px;color:#333'>
        <b style='font-size:15px;color:#111'>{index_name}</b>
        <span>O <b>{o_val:,.2f}</b></span>
        <span>H <b>{h_val:,.2f}</b></span>
        <span>L <b>{l_val:,.2f}</b></span>
        <span>C <b>{c_val:,.2f}</b></span>
        <span style='color:{quote_color};font-weight:bold'>{chg_val:+,.2f} ({chg_pct:+.2f}%)</span>
    </div>
    """,
    unsafe_allow_html=True
)

show_volume = st.checkbox("Show Volume bars", value=False, key="show_volume_toggle",
                            help="Off by default for a cleaner TradingView-style price chart.")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=chart_df.index, open=chart_df["Open"], high=chart_df["High"], low=chart_df["Low"], close=chart_df["Close"],
    name="Price", increasing_line_color="#089981", decreasing_line_color="#f23645",
    increasing_fillcolor="#089981", decreasing_fillcolor="#f23645",
    line=dict(width=1)
))

if show_volume and "Volume" in chart_df.columns:
    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df["Volume"], name="Volume", marker_color="#b2b5be", yaxis="y2", opacity=0.35))

last_price = float(chart_df["Close"].iloc[-1])
last_price_color = "#089981" if chg_val >= 0 else "#f23645"
fig.add_hline(
    y=last_price, line_dash="dot", line_color=last_price_color, line_width=1,
    annotation_text=f"{last_price:,.2f}", annotation_position="right",
    annotation=dict(font=dict(color="white", size=11), bgcolor=last_price_color, bordercolor=last_price_color)
)

if show_sma and "SMA_FAST" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA_FAST"], name="SMA Fast", line=dict(width=1, color="teal")))
if show_sma and "SMA_SLOW" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SMA_SLOW"], name="SMA Slow", line=dict(width=1, color="purple")))
if show_supertrend and "SUPERTREND" in chart_df.columns:
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["SUPERTREND"], name="Supertrend", line=dict(width=1.5, color="magenta", dash="dot")))

if show_rule_signals and "SIGNAL" in chart_df.columns:
    buys, sells = chart_df[chart_df["SIGNAL"] == "BUY"], chart_df[chart_df["SIGNAL"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="BUY (rule)", marker=dict(color="lightgreen", size=7, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="SELL (rule)", marker=dict(color="lightcoral", size=7, symbol="triangle-down")))

if show_final_signals and "FINAL_SIGNAL" in chart_df.columns:
    buy_points = chart_df[chart_df["FINAL_SIGNAL"] == "STRONG BUY"]
    sell_points = chart_df[chart_df["FINAL_SIGNAL"] == "STRONG SELL"]
    fig.add_trace(go.Scatter(
        x=buy_points.index, y=buy_points["Low"] * 0.985, mode="markers+text", name="STRONG BUY",
        marker=dict(symbol="triangle-up", size=18, color="#00c853", line=dict(width=1.5, color="darkgreen")),
        text=["BUY"] * len(buy_points), textposition="bottom center", textfont=dict(color="darkgreen", size=10)
    ))
    fig.add_trace(go.Scatter(
        x=sell_points.index, y=sell_points["High"] * 1.015, mode="markers+text", name="STRONG SELL",
        marker=dict(symbol="triangle-down", size=18, color="#d50000", line=dict(width=1.5, color="darkred")),
        text=["SELL"] * len(sell_points), textposition="top center", textfont=dict(color="darkred", size=10)
    ))

if trade_plan is not None:
    fig.add_hline(y=trade_plan["stop_loss"], line_dash="dash", line_color="red", annotation_text="Stop-Loss", annotation_position="top left")
    fig.add_hline(y=trade_plan["target"], line_dash="dash", line_color="green", annotation_text="Target", annotation_position="bottom left")

if is_intraday:
    fig.update_xaxes(
        rangebreaks=[
            dict(bounds=["sat", "mon"]),
            dict(bounds=[15.5, 9.25], pattern="hour"),
        ],
        rangeslider=dict(visible=False), type="date"
    )
else:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])], rangeslider=dict(visible=False), type="date")

visible_high = chart_df["High"].max()
visible_low = chart_df["Low"].min()
price_padding = (visible_high - visible_low) * 0.08 if visible_high > visible_low else visible_high * 0.01
y_range = [visible_low - price_padding, visible_high + price_padding]

if fullscreen_mode:
    chart_height = 900
elif compact_mobile:
    chart_height = 420
else:
    chart_height = 700
fig.update_layout(
    height=chart_height,
    margin=dict(l=10, r=55, t=25, b=10),
    yaxis=dict(
        title=None, range=y_range, autorange=False, side="right",
        showgrid=True, gridcolor="#eeeeee", gridwidth=1,
        tickformat=",.0f", zeroline=False
    ),
    yaxis2=dict(title=None, overlaying="y", side="left", showgrid=False, visible=show_volume),
    plot_bgcolor="white",
    paper_bgcolor="white",
    template="plotly_white",
    showlegend=False,
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor="#f2f2f2"),
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True, "scrollZoom": True, "responsive": True, "doubleClick": "reset", "displaylogo": False,
})

st.caption(
    "💡 1D/5D/1W now use real minute-level intraday data, WITH SMA/Supertrend/STRONG BUY-SELL markers "
    "computed directly on those intraday candles (green ▲ / red ▼ = where the rule engine and Supertrend "
    "agree on that timeframe). 1M and beyond use your daily swing data with the same overlays. The price "
    "axis is locked to only the visible candles so it always fills the chart properly. The dotted line = "
    "last traded price. Dashed red/green lines = current Stop-Loss/Target from the Trade Plan above (which "
    "is always based on the DAILY signal, not the intraday one - so it may not always line up with intraday markers)."
)

if fullscreen_mode:
    st.info("Full-Screen Chart Mode is ON — other sections are hidden. Turn it off in the sidebar to see everything again.")
    st.stop()

df = swing_df

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"{latest['Close']:,.2f}")
prev_close = full_df.iloc[-2]["Close"] if len(full_df) > 1 else latest["Close"]
col2.metric("Daily Change", f"{(latest['Close']/prev_close-1)*100:+.2f}%")
col3.metric("Composite Score", f"{explain['composite']:+.2f}")
color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[explain["decision"]]
col4.markdown(f"<h4 style='color:{color};text-align:center'>Rule Engine: {explain['decision']}</h4>", unsafe_allow_html=True)

st.subheader("Special Indicator: Supertrend + ADX (Trend Strength)")
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Supertrend Direction", "Uptrend" if latest["SUPERTREND_DIR"] == 1 else "Downtrend")
sc2.metric("ADX (trend strength)", f"{latest['ADX']:.1f}")
st_signal = latest["ST_SIGNAL"]
st_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}[st_signal]
sc3.markdown(f"<h4 style='color:{st_color};text-align:center'>Supertrend: {st_signal}</h4>", unsafe_allow_html=True)

st.caption("STRONG BUY/SELL = both the rule engine and Supertrend+ADX agree. MIXED/CAUTION = they disagree.")
log_df = logger.log_daily_signal(ticker, latest)
accuracy, valid_count = logger.compute_accuracy(log_df)

st.subheader("📒 Daily Signal Log & Track Record")
if accuracy is not None:
    st.metric("Historical Accuracy (this app's own track record)", f"{accuracy:.1f}%", help=f"Based on {valid_count} completed signal days so far")
else:
    st.info("Not enough logged days yet to calculate accuracy.")

st.markdown("**Filter log:**")
lf1, lf2 = st.columns([2, 1])
with lf1:
    search_term = st.text_input("Search log (matches any column)", "")
with lf2:
    signal_filter = st.multiselect(
        "Filter by signal",
        options=sorted(log_df["FINAL_SIGNAL"].dropna().unique()) if "FINAL_SIGNAL" in log_df.columns else [],
        default=[]
    )

filtered_log = log_df.copy()
if signal_filter and "FINAL_SIGNAL" in filtered_log.columns:
    filtered_log = filtered_log[filtered_log["FINAL_SIGNAL"].isin(signal_filter)]
if search_term:
    mask_search = filtered_log.apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
    filtered_log = filtered_log[mask_search]

st.dataframe(filtered_log.sort_values("date", ascending=False), use_container_width=True)

csv_data = log_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download signal log as CSV (backup)", data=csv_data, file_name="signal_log_backup.csv", mime="text/csv")

st.subheader("Why this signal?")
st.dataframe(pd.DataFrame({
    "component": ["Trend", "Momentum(RSI)", "MACD", "Volatility(BB)", "Volume"],
    "score": [explain["trend"], explain["momentum"], explain["macd"], explain["volatility"], explain["volume"]],
}), use_container_width=True, hide_index=True)

st.subheader("Backtest vs Buy & Hold")
result = backtester.run_backtest(df, allow_short=allow_short)
b1, b2, b3, b4 = st.columns(4)
b1.metric("Strategy Return", f"{result['total_return_pct']}%")
b2.metric("Buy & Hold", f"{result['buy_hold_return_pct']}%")
b3.metric("Sharpe", f"{result['sharpe_ratio']}")
b4.metric("Max Drawdown", f"{result['max_drawdown_pct']}%")

eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=result["equity_curve"].index, y=result["equity_curve"]["EQUITY_CURVE"], name="Strategy"))
eq_fig.add_trace(go.Scatter(x=result["equity_curve"].index, y=result["equity_curve"]["BUY_HOLD_EQUITY"], name="Buy & Hold"))
st.plotly_chart(eq_fig, use_container_width=True)

st.dataframe(
    df[["Close", "RSI", "MACD", "SCORE", "SIGNAL", "ADX", "ST_SIGNAL", "FINAL_SIGNAL"]].tail(20).sort_index(ascending=False),
    use_container_width=True
)

st.caption("Not financial advice. For educational use only.")
