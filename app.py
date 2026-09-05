import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import logger
import config as cfg
import data_fetcher
import indicators
import signal_engine
import backtester

st.set_page_config(page_title="Nifty & Sensex Daily Analyzer", layout="wide")

# Tighten Streamlit's default padding so the chart gets more real screen space
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; padding-left: 1.5rem; padding-right: 1.5rem;}
        [data-testid="stMetricValue"] {font-size: 1.1rem;}
    </style>
""", unsafe_allow_html=True)

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
        "Candle Resolution",
        ["Daily", "Weekly", "Monthly"],
        index=0,
        help="Daily = 1 candle/day, Weekly = 1 candle/week, Monthly = 1 candle/month."
    )
    show_sma = st.checkbox("Show SMA lines", value=True)
    show_supertrend = st.checkbox("Show Supertrend line", value=True)
    show_rule_signals = st.checkbox("Show rule-engine BUY/SELL markers (small)", value=False)
    show_final_signals = st.checkbox("Show STRONG BUY/SELL (final) markers", value=True)

    st.markdown("---")
    fullscreen_mode = st.checkbox(
        "🖥️ Full-Screen Chart Mode",
        value=False,
        help="Hides everything except the chart so you can analyze it in maximum space. Use the chart's own expand icon (top-right of chart) to go true browser full-screen."
    )

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


def resample_ohlc(data, tf):
    """Resample daily-indexed data to Weekly/Monthly, NSE/Moneycontrol style."""
    if tf == "Daily":
        return data

    rule = "W" if tf == "Weekly" else "M"

    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    other_cols = [c for c in data.columns if c not in agg]
    for c in other_cols:
        agg[c] = "last"

    resampled = data.resample(rule).agg(agg).dropna(subset=["Close"])
    return resampled


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

# Apply Daily / Weekly / Monthly resolution (full history - range buttons on chart handle zoom)
df = resample_ohlc(full_df, timeframe)

if df.empty or len(df) < 2:
    st.warning("Not enough data for this timeframe. Try Daily view or a longer History window.")
    st.stop()

latest = df.iloc[-1]
explain = signal_engine.explain_latest(latest)

# ============ MAIN CHART (always shown, even in full-screen mode) ============
final_signal = latest["FINAL_SIGNAL"]
final_color = {"STRONG BUY": "darkgreen", "STRONG SELL": "darkred", "HOLD": "gray", "MIXED / CAUTION": "orange"}[final_signal]

st.markdown(
    f"<div style='background-color:{final_color};padding:14px;border-radius:10px;text-align:center;margin-bottom:10px'>"
    f"<h2 style='color:white;margin:0'>Today's Signal: {final_signal}  |  {index_name} @ {latest['Close']:,.2f}</h2></div>",
    unsafe_allow_html=True
)

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
    name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
))

if "Volume" in df.columns:
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", marker_color="lightblue",
        yaxis="y2", opacity=0.3
    ))

if show_sma and "SMA_FAST" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_FAST"], name="SMA Fast", line=dict(width=1, color="teal")))
if show_sma and "SMA_SLOW" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["SMA_SLOW"], name="SMA Slow", line=dict(width=1, color="purple")))
if show_supertrend and "SUPERTREND" in df.columns:
    fig.add_trace(go.Scatter(x=df.index, y=df["SUPERTREND"], name="Supertrend", line=dict(width=1.5, color="magenta", dash="dot")))

if show_rule_signals:
    buys, sells = df[df["SIGNAL"] == "BUY"], df[df["SIGNAL"] == "SELL"]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Close"], mode="markers", name="BUY (rule)", marker=dict(color="lightgreen", size=7, symbol="triangle-up")))
    fig.add_trace(go.Scatter(x=sells.index, y=sells["Close"], mode="markers", name="SELL (rule)", marker=dict(color="lightcoral", size=7, symbol="triangle-down")))

if show_final_signals:
    buy_points = df[df["FINAL_SIGNAL"] == "STRONG BUY"]
    sell_points = df[df["FINAL_SIGNAL"] == "STRONG SELL"]
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

# TradingView-style quick range buttons directly on the chart
fig.update_xaxes(
    rangeselector=dict(
        bgcolor="#f0f2f6",
        activecolor="#4c8bf5",
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=5, label="5d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=9, label="9m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ])
    ),
    rangeslider=dict(visible=False),
    type="date"
)

chart_height = 900 if fullscreen_mode else 700

fig.update_layout(
    title=f"{index_name} — {timeframe} Chart",
    height=chart_height,
    margin=dict(l=10, r=10, t=60, b=10),
    yaxis=dict(title="Price"),
    yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, use_container_width=True, config={
    "displayModeBar": True,
    "scrollZoom": True,
    "responsive": True,
    "doubleClick": "reset",
    "displaylogo": False,
})

st.caption("💡 Tip: Use the 1d / 5d / 1w / 1m / 3m / 6m / 9m / 1y / All buttons above the chart to zoom the time range instantly — just like NSE/Moneycontrol. Click the camera icon in the chart toolbar to save it as an image, or drag directly on the chart to zoom into any custom range.")

if fullscreen_mode:
    st.info("Full-Screen Chart Mode is ON — other sections (metrics, logs, backtest) are hidden. Turn it off in the sidebar to see everything again.")
    st.stop()

# ============ EVERYTHING BELOW ONLY SHOWS WHEN NOT IN FULL-SCREEN MODE ============

col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest Close", f"{latest['Close']:,.2f}")
prev_close = df.iloc[-2]["Close"] if len(df) > 1 else latest["Close"]
col2.metric(f"{timeframe} Change", f"{(latest['Close']/prev_close-1)*100:+.2f}%")
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

st.caption("STRONG BUY/SELL = both the rule engine and Supertrend+ADX agree. MIXED/CAUTION = they disagree, meaning conditions are less clear-cut today.")
log_df = logger.log_daily_signal(ticker, latest)
accuracy, valid_count = logger.compute_accuracy(log_df)

st.subheader("📒 Daily Signal Log & Track Record")

if accuracy is not None:
    st.metric("Historical Accuracy (this app's own track record)", f"{accuracy:.1f}%", help=f"Based on {valid_count} completed signal days so far")
else:
    st.info("Not enough logged days yet to calculate accuracy. Check back after a few days of daily visits.")

st.markdown("**Filter log:**")
lf1, lf2 = st.columns([2, 1])
with lf1:
    search_term = st.text_input("Search log (matches any column, e.g. date, signal, index name)", "")
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
    mask_search = filtered_log.apply(
        lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1
    )
    filtered_log = filtered_log[mask_search]

st.dataframe(filtered_log.sort_values("date", ascending=False), use_container_width=True)

csv_data = log_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download signal log as CSV (backup)",
    data=csv_data,
    file_name="signal_log_backup.csv",
    mime="text/csv"
)
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
