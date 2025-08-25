import io
import os
from datetime import date
from typing import List, Dict, Tuple, Optional
import math

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

# Optional PDF dependency
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import cm
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

TRADING_DAYS = 252

DEFAULT_TOP40 = [
    "NPN.JO", "PRX.JO", "CFR.JO", "BTI.JO", "GLN.JO", "BHG.JO", "ANH.JO",
    "AGL.JO", "KIO.JO", "EXX.JO", "IMP.JO", "SSW.JO", "S32.JO",
    "FSR.JO", "SBK.JO", "ABG.JO", "NED.JO", "CPI.JO", "DSY.JO", "SLM.JO",
    "OMU.JO", "RNI.JO", "INP.JO", "INL.JO", "MTN.JO", "VOD.JO", "BID.JO",
    "BVT.JO", "APN.JO", "WHL.JO", "MRP.JO", "TFG.JO", "MNP.JO", "REM.JO",
    "SOL.JO", "PPH.JO", "GRT.JO", "RDF.JO"
]


def pct(x: float) -> str:
    """Format a float as a percentage string with two decimals."""
    return f"{x * 100:.2f}%"


def cagr(equity: pd.Series) -> float:
    """Calculate the Compound Annual Growth Rate (CAGR) from an equity curve."""
    eq = equity.dropna()
    if eq.empty:
        return 0.0
    start, end = float(eq.iloc[0]), float(eq.iloc[-1])
    n_years = len(eq) / TRADING_DAYS
    if start <= 0 or end <= 0 or n_years <= 0:
        return 0.0
    return (end / start) ** (1 / n_years) - 1


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Calculate the annualized Sharpe ratio of daily returns."""
    daily_ret = daily_returns.dropna()
    if daily_ret.empty:
        return 0.0
    annual_ret = daily_ret.mean() * TRADING_DAYS
    annual_volatility = daily_ret.std(ddof=0) * math.sqrt(TRADING_DAYS)
    extra_ret = annual_ret - rf
    return float(extra_ret / annual_volatility) if annual_volatility > 0 else 0.0


def max_drawdown(equity: pd.Series) -> Tuple[float, int]:
    """Calculate the maximum drawdown and its duration in days."""
    eq = equity.ffill()
    peaks = eq.cummax()
    draw_down = eq / peaks - 1.0
    max_draw_down = float(draw_down.min())
    end_idx = draw_down.idxmin() if not draw_down.empty else None
    if end_idx is None:
        return 0.0, 0
    start_idx = (eq.loc[:end_idx]).idxmax()
    dur = (end_idx - start_idx).days if hasattr(end_idx, "to_pydatetime") else 0
    return max_draw_down, int(dur)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compute the cumulative equity curve from daily returns."""
    return (1 + returns.fillna(0.0)).cumprod()


def apply_turnover_costs(
    daily_returns: pd.Series, positions: pd.Series, cost_bps: float
) -> pd.Series:
    """Apply turnover costs to daily returns based on position changes."""
    pos = positions.fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    return daily_returns - (cost_bps / 10000.0) * turnover


def download_prices(
    tickers: List[str], start: str, end: str, batch_size: int = 10
) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers from Yahoo Finance.
    Downloads in batches to avoid API limits.
    """
    all_px = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(
                sorted(set(batch)), start=start, end=end,
                auto_adjust=True, progress=False
            )
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                px = data["Close"].copy()
            else:
                px = data[["Close"]].copy()
            px.columns = [c if isinstance(c, str) else c[1] for c in px.columns]
            all_px.append(px)
        except Exception as e:
            st.warning(f"Error downloading batch {batch}: {e}")
    if not all_px:
        return pd.DataFrame()
    return pd.concat(all_px, axis=1).dropna(how="all")


def parse_uploaded_tickers(file) -> List[str]:
    """
    Parse uploaded file for tickers.
    Supports CSV (with 'ticker' column or first column) and TXT (comma or newline separated).
    """
    if file is None:
        return []
    name = file.name.lower()
    content = file.read().decode("utf-8", errors="ignore")
    try:
        df = pd.read_csv(io.StringIO(content))
        if "ticker" in df.columns:
            return [t.strip() for t in df["ticker"].astype(str).tolist() if t and isinstance(t, str)]
        return [t.strip() for t in df.iloc[:, 0].astype(str).tolist() if t and isinstance(t, str)]
    except Exception:
        pass
    sep = "," if "," in content else "\n"
    return [t.strip() for t in content.split(sep) if t.strip()]


# -------------------- Strategies --------------------
def strat_mean_reversion_bbands(
    close: pd.Series, lookback: int, z_entry: float, z_exit: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Mean reversion strategy using Bollinger Bands z-score.
    Returns daily returns and positions.
    """
    if len(close) < lookback + 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ma = close.rolling(lookback).mean()
    sd = close.rolling(lookback).std()
    z = (close - ma) / sd
    pos = np.zeros(len(close))
    in_pos = False
    for i, zi in enumerate(z):
        if not in_pos and zi <= -z_entry:
            in_pos = True
        elif in_pos and zi >= -z_exit:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    pos_series = pd.Series(pos, index=close.index)
    ret = close.pct_change().fillna(0.0) * pos_series.shift(1).fillna(0.0)
    return ret, pos_series


def strat_trend_ema_crossover(
    close: pd.Series, short: int, long: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Trend following strategy using EMA crossover.
    Returns daily returns and positions.
    """
    if len(close) < long + 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    ema_s = close.ewm(span=short, adjust=False).mean()
    ema_l = close.ewm(span=long, adjust=False).mean()
    pos = (ema_s > ema_l).astype(float)
    ret = close.pct_change().fillna(0.0) * pos.shift(1).fillna(0.0)
    return ret, pos


def strat_breakout_donchian(
    close: pd.Series, entry: int, exit_: int
) -> Tuple[pd.Series, pd.Series]:
    """
    Breakout strategy using Donchian channels.
    Returns daily returns and positions.
    """
    if len(close) < max(entry, exit_) + 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    hh = close.rolling(entry).max()
    ll = close.rolling(exit_).min()
    pos = np.zeros(len(close))
    in_pos = False
    for i in range(len(close)):
        c = close.iloc[i]
        if not in_pos and i > 0 and c > hh.iloc[i - 1]:
            in_pos = True
        elif in_pos and i > 0 and c < ll.iloc[i - 1]:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    pos_series = pd.Series(pos, index=close.index)
    ret = close.pct_change().fillna(0.0) * pos_series.shift(1).fillna(0.0)
    return ret, pos_series


def strat_pairs_spread_z(
    x_close: pd.Series, y_close: pd.Series, lookback: int, z_entry: float, z_exit: float
) -> Tuple[pd.Series, pd.Series]:
    """
    Market-neutral pairs trading strategy using spread z-score.
    Returns daily returns and turnover.
    """
    idx = x_close.index.intersection(y_close.index)
    x, y = x_close.reindex(idx).ffill(), y_close.reindex(idx).ffill()
    if len(x) < lookback + 5 or len(y) < lookback + 5:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    lx, ly = np.log(x), np.log(y)
    spread = lx - ly
    m = spread.rolling(lookback).mean()
    s = spread.rolling(lookback).std()
    z = (spread - m) / s

    pos = np.zeros(len(idx))
    state = 0
    for i, zi in enumerate(z):
        if state == 0:
            if zi <= -z_entry:
                state = +1
            elif zi >= z_entry:
                state = -1
        else:
            if abs(zi) <= z_exit:
                state = 0
        pos[i] = float(state)
    pos_series = pd.Series(pos, index=idx)
    rx, ry = x.pct_change().fillna(0.0), y.pct_change().fillna(0.0)
    pnl = pos_series.shift(1).fillna(0.0) * (rx - ry)
    turnover = pos_series.diff().abs().fillna(pos_series.abs()) * 2.0  # two legs
    return pnl, turnover


def equal_weight_portfolio(returns_dict: Dict[str, pd.Series]) -> pd.Series:
    """
    Combine multiple return series into an equal-weight portfolio.
    """
    df = pd.DataFrame(returns_dict).dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float, index=returns_dict[next(iter(returns_dict))].index if returns_dict else None)
    return df.mean(axis=1).fillna(0.0)


# -------------------- Plain English --------------------
def explain_sharpe(v: float) -> str:
    """Return a plain-English explanation for Sharpe ratio value."""
    if v < 0:
        return "This strategy loses value relative to its risk. Consider avoiding."
    if v < 0.5:
        return "Risk-adjusted return is weak and may be unstable."
    if v < 1.0:
        return "Modest risk-adjusted return. Consider smaller position sizes."
    if v < 1.5:
        return "Good risk-adjusted return. A reasonable candidate."
    if v < 2.0:
        return "Very good risk-adjusted return."
    return "Exceptional risk-adjusted return — double-check it’s not overfit."


def explain_cagr(v: float) -> str:
    """Return a plain-English explanation for CAGR value."""
    if v < 0:
        return "Loses money over the period."
    if v < 0.05:
        return "Low growth. Might not beat passive alternatives."
    if v < 0.15:
        return "Moderate growth."
    return "High growth. Ensure the risks are acceptable."


def explain_maxdd(v: float) -> str:
    """Return a plain-English explanation for max drawdown value."""
    if v > -0.10:
        return "Shallow worst loss — easier to live with."
    if v > -0.25:
        return "Manageable worst loss for many investors."
    if v > -0.40:
        return "Deep loss — expect uncomfortable periods."
    return "Very deep loss — only for high risk tolerance."


def explain_dd_days(d: int) -> str:
    """Return a plain-English explanation for drawdown duration."""
    if d < 200:
        return "Recoveries are relatively quick."
    if d < 500:
        return "Recoveries take a while; patience required."
    if d < 1000:
        return "Recoveries are slow; long patience needed."
    return "Very slow recoveries; this will test patience."


def build_summary_block(name: str, c: float, s: float, dd: float, ddd: int) -> str:
    """Build a plain-English summary block for a strategy's metrics."""
    lines = [
        f"Strategy: {name}",
        f"- CAGR (growth rate): {pct(c)}. {explain_cagr(c)}",
        f"- Sharpe (risk-adjusted return): {s:.2f}. {explain_sharpe(s)}",
        f"- Maximum Drawdown (worst loss): {pct(dd)}. {explain_maxdd(dd)}",
        f"- Drawdown Days (time to recover): {ddd} days. {explain_dd_days(ddd)}"
    ]
    recs = []
    if s >= 1.0 and dd > -0.30:
        recs.append("Good candidate for live use with sensible position sizing.")
    if s >= 1.5 and c >= 0.10:
        recs.append("Consider making this a core strategy in the portfolio.")
    if c >= 0.15 and dd <= -0.30:
        recs.append("High returns but deep losses — use smaller sizes or add stop-loss/vol targeting.")
    if s < 0.5:
        recs.append("Treat as experimental. Results are weak as-is.")
    if recs:
        lines.append("Recommendations:")
        for r in recs:
            lines.append(f"  • {r}")
    return "\n".join(lines)


# -------------------- UI --------------------
st.set_page_config(page_title="JSE Strategy Comparison", layout="wide")
st.title("📈 JSE Strategy Comparison (Point-and-Click)")
st.caption("Pick tickers, dates, strategies, and costs. Get clear English explanations and downloads.")

with st.sidebar:
    st.header("1) Universe")
    default_choice = st.radio("Choose tickers from:", ["JSE Top 40 (built-in)", "Upload my list"])
    uploaded = None
    tickers: List[str] = []
    if default_choice == "Upload my list":
        uploaded = st.file_uploader("Upload TXT/CSV of tickers (e.g., NPN.JO)", type=["txt", "csv"])
        if uploaded:
            tickers = parse_uploaded_tickers(uploaded)
            st.success(f"Loaded {len(tickers)} tickers.")
    else:
        tickers = DEFAULT_TOP40.copy()

    tickers_text = st.text_area("Tickers (comma-separated)", value=", ".join(tickers))
    tickers = [t.strip() for t in tickers_text.split(",") if t.strip()]

    st.header("2) Date Range")
    start_date = st.date_input("Start", value=date(2015, 1, 1))
    end_date = st.date_input("End", value=date.today(), min_value=start_date)

    st.header("3) Costs (per turnover)")
    fees_bps = st.slider("Broker & taxes (bps)", 0, 50, 5)
    slippage_bps = st.slider("Slippage (bps)", 0, 50, 5)
    total_bps = fees_bps + slippage_bps

    st.header("4) Strategies")
    use_mr = st.checkbox("Mean Reversion (Bollinger z-score)", True)
    use_tf = st.checkbox("Trend Following (EMA cross)", True)
    use_bo = st.checkbox("Breakout (Donchian)", False)
    use_pairs = st.checkbox("Pairs (market-neutral)", False)

    st.caption("Tip: Start with Mean Reversion + Trend; add Breakout/Pairs later.")

    st.header("5) Parameters")
    st.subheader("Mean Reversion")
    mr_lb = st.slider("MR lookback (days)", 10, 60, 20, step=1)
    mr_z_entry = st.slider("MR entry z-score (buy below)", 0.5, 3.0, 1.5, step=0.1)
    mr_z_exit = st.slider("MR exit z-score (flat above)", 0.1, 2.0, 0.5, step=0.1)

    st.subheader("Trend Following")
    tf_short = st.slider("EMA short", 10, 150, 50, step=5)
    tf_long = st.slider("EMA long", 100, 300, 200, step=5)

    st.subheader("Breakout")
    bo_entry = st.slider("Entry lookback (N-day high)", 10, 120, 55, step=1)
    bo_exit = st.slider("Exit lookback (M-day low)", 5, 60, 20, step=1)

    st.subheader("Pairs")
    pair_x = st.selectbox("Leg X", options=tickers, index=0 if tickers else 0)
    pair_y = st.selectbox("Leg Y", options=tickers, index=1 if len(tickers) > 1 else 0)
    pairs_lb = st.slider("Pairs lookback", 20, 120, 60, step=5)
    pairs_z_entry = st.slider("Pairs entry z", 0.5, 3.0, 2.0, step=0.1)
    pairs_z_exit = st.slider("Pairs exit z", 0.1, 2.0, 0.5, step=0.1)

    run_btn = st.button("▶ Run Backtest")

if run_btn:
    if not tickers:
        st.error("Please provide at least one ticker.")
        st.stop()

    st.info("Downloading price data…")
    px = download_prices(tickers, start_date.isoformat(), end_date.isoformat())
    if px.empty:
        st.error("No price data found. Check tickers or date range. Try fewer tickers or a different date range.")
        st.stop()

    px = px.dropna(how="all")
    dates = px.index

    results: Dict[str, pd.Series] = {}
    english_blocks: Dict[str, str] = {}

    if use_mr:
        per_ticker = {}
        for t in px.columns:
            s = px[t].dropna()
            r, p = strat_mean_reversion_bbands(s, mr_lb, mr_z_entry, mr_z_exit)
            if r.empty:
                continue
            r_cost = apply_turnover_costs(r, p, total_bps)
            per_ticker[t] = r_cost.reindex(dates).fillna(0.0)
        if per_ticker:
            port = equal_weight_portfolio(per_ticker)
            results["Mean Reversion"] = port

    if use_tf:
        per_ticker = {}
        for t in px.columns:
            s = px[t].dropna()
            r, p = strat_trend_ema_crossover(s, tf_short, tf_long)
            if r.empty:
                continue
            r_cost = apply_turnover_costs(r, p, total_bps)
            per_ticker[t] = r_cost.reindex(dates).fillna(0.0)
        if per_ticker:
            port = equal_weight_portfolio(per_ticker)
            results["Trend EMA"] = port

    if use_bo:
        per_ticker = {}
        for t in px.columns:
            s = px[t].dropna()
            r, p = strat_breakout_donchian(s, bo_entry, bo_exit)
            if r.empty:
                continue
            r_cost = apply_turnover_costs(r, p, total_bps)
            per_ticker[t] = r_cost.reindex(dates).fillna(0.0)
        if per_ticker:
            port = equal_weight_portfolio(per_ticker)
            results["Breakout Donchian"] = port

    if use_pairs and pair_x in px.columns and pair_y in px.columns and pair_x != pair_y:
        x = px[pair_x].dropna()
        y = px[pair_y].dropna()
        r_pairs, turn_pairs = strat_pairs_spread_z(x, y, pairs_lb, pairs_z_entry, pairs_z_exit)
        if r_pairs.empty:
            st.warning("Pairs strategy: insufficient data for selected tickers or lookback.")
        else:
            r_pairs = r_pairs.reindex(dates).fillna(0.0)
            cost_per_day = (total_bps / 10000.0) * turn_pairs.reindex(dates).fillna(0.0)
            port_pairs = r_pairs - cost_per_day
            results[f"Pairs {pair_x}/{pair_y}"] = port_pairs

    if not results:
        st.warning("No strategies produced results (insufficient data after filters). Try different dates, tickers, or parameters.")
        st.stop()

    rows = []
    for name, rets in results.items():
        eq = equity_curve(rets)
        c = cagr(eq)
        s = sharpe_ratio(rets)
        dd, ddd = max_drawdown(eq)
        rows.append([name, c, s, dd, ddd])
        english_blocks[name] = build_summary_block(name, c, s, dd, ddd)

    summary_df = pd.DataFrame(
        rows,
        columns=["Strategy", "CAGR", "Sharpe", "MaxDD", "DD_Days"]
    ).set_index("Strategy")
    st.subheader("Results")
    st.dataframe(
        summary_df.assign(
            CAGR=lambda d: d["CAGR"].map(pct),
            MaxDD=lambda d: d["MaxDD"].map(pct)
        )[["CAGR", "Sharpe", "MaxDD", "DD_Days"]]
    )

    st.subheader("Plain-English Explanations")
    for name in summary_df.index:
        with st.expander(name, expanded=True):
            st.text(english_blocks[name])

    st.subheader("Equity Curves")
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, rets in results.items():
        eq = equity_curve(rets)
        ax.plot(eq, label=name)
    ax.set_title("Strategy Equity Curves")
    ax.set_yscale("log")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Downloads")
    csv_buf = io.StringIO()
    out_csv = summary_df.copy()
    out_csv.to_csv(csv_buf)
    st.download_button(
        "Download Summary CSV",
        data=csv_buf.getvalue(),
        file_name="strategy_summary.csv",
        mime="text/csv"
    )

    txt_report = io.StringIO()
    txt_report.write("JSE Strategy Report (Plain English)\n\n")
    txt_report.write(f"Date range: {start_date} to {end_date}\n")
    txt_report.write(f"Costs: fees {fees_bps} bps, slippage {slippage_bps} bps (total {total_bps} bps per turnover)\n\n")
    txt_report.write("Metrics guide:\n- CAGR = average yearly growth (bigger is better)\n")
    txt_report.write("- Sharpe = return per unit of risk (>1 good, >2 excellent)\n")
    txt_report.write("- Max Drawdown = worst loss from prior peak (less negative better)\n")
    txt_report.write("- Drawdown Days = time to recover a loss (smaller better)\n\n")
    for name in summary_df.index:
        txt_report.write(english_blocks[name] + "\n\n")
    st.download_button(
        "Download TXT Report",
        data=txt_report.getvalue(),
        file_name="strategy_report.txt",
        mime="text/plain"
    )

    if REPORTLAB_AVAILABLE:
        pdf_buf = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buf,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm
        )
        styles = getSampleStyleSheet()
        story = []
        story.append(Paragraph("JSE Strategy Report (Plain English)", styles["Title"]))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(f"Date range: {start_date} to {end_date}", styles["Normal"]))
        story.append(Paragraph(
            f"Costs: fees {fees_bps} bps, slippage {slippage_bps} bps (total {total_bps} bps per turnover)",
            styles["Normal"]
        ))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Metrics guide:", styles["Heading3"]))
        story.append(Paragraph("• CAGR = average yearly growth (bigger is better)", styles["Normal"]))
        story.append(Paragraph("• Sharpe = return per unit of risk (>1 good, >2 excellent)", styles["Normal"]))
        story.append(Paragraph("• Max Drawdown = worst loss from prior peak (less negative better)", styles["Normal"]))
        story.append(Paragraph("• Drawdown Days = time to recover a loss (smaller better)", styles["Normal"]))
        story.append(Spacer(1, 0.4 * cm))
        for name in summary_df.index:
            story.append(Paragraph(name, styles["Heading3"]))
            for line in english_blocks[name].split("\n"):
                story.append(Paragraph(line.replace("•", "-"), styles["Normal"]))
            story.append(Spacer(1, 0.3 * cm))
        doc.build(story)
        st.download_button(
            "Download PDF Report",
            data=pdf_buf.getvalue(),
            file_name="strategy_report.pdf",
            mime="application/pdf"
        )
    else:
        st.caption("Install `reportlab` for PDF export (optional).")