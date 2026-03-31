"""
Backtest of Father's Stock Rules (52 years of experience)
Using S&P 500 (^GSPC) as proxy via yfinance
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

try:
    # ── Page config ──────────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Father's Rules Backtest",
        page_icon="📈",
        layout="wide",
    )
except:
    pass

def main():
    st.title("📈 Father's Stock Rules — Backtest")
    st.caption(
        "52 years of wisdom: _Discipline + Patience = Stable Long-Term Growth_"
    )

    # ── Sidebar: parameters ───────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")

        start_date = st.date_input("Start date", value=date(2000, 1, 1),min_value=date(1900,1,1), max_value=date.today())
        end_date   = st.date_input("End date",   value=date.today())

        initial_cash = st.number_input(
            "Initial investment ($)", min_value=1_000, max_value=10_000_000,
            value=10_000, step=1_000,
        )

        st.subheader("📋 The Rules")
        st.markdown(
            """
            | Change from ref | Action |
            |---|---|
            | Falls 5% | Hold |
            | Falls 15% | Buy 10% of cash |
            | Falls 25% | Buy 25% of cash |
            | Rises 5% | Hold |
            | Rises 15% | Hold |
            | Rises 25% | Sell 10% of shares |
            | Rises 35% | Sell 20% of shares |
            | Rises 45% | Sell 30% of shares |
            | Rises 60% | Sell 40% of shares |
            | Rises 100% | Sell everything |
            """
        )

        st.subheader("📌 Benchmark")
        ticker_choice = st.radio(
            "Index / proxy",
            options=["^SP500TR", "SPY", "^GSPC"],
            captions=[
                "Total Return (dividends reinvested) — limited history",
                "SPY ETF (dividends in price) — from 1993",
                "Price only — no dividends",
            ],
            index=1,
        )

        st.subheader("🔧 Advanced")
        ref_reset_on_trade = st.checkbox(
            "Reset reference price after each trade", value=True,
            help="When ON: reference resets to current price after a buy/sell. "
                "When OFF: reference is always the initial buy price."
        )
        commission = st.number_input(
            "Commission per trade ($)", min_value=0.0, max_value=50.0,
            value=0.0, step=0.5,
        )

    # ── Rules definition ──────────────────────────────────────────────────────────
    # Each rule: (threshold_pct, action, fraction)
    # action: "buy" uses fraction of available cash; "sell" uses fraction of shares held
    RULES = [
        (-25, "buy",  0.25),
        (-15, "buy",  0.10),
        ( 25, "sell", 0.10),
        ( 35, "sell", 0.20),
        ( 45, "sell", 0.30),
        ( 60, "sell", 0.40),
        (100, "sell", 1.00),
    ]

    def apply_rules(
        prices: pd.Series,
        initial_cash: float,
        ref_reset: bool,
        commission: float,
    ) -> pd.DataFrame:
        """
        Simulate the strategy day by day.
        Returns a DataFrame with portfolio state per day.
        """
        ref_price   = prices.iloc[0]  # reference price for % calculation
        shares      = initial_cash / ref_price  # buy 100% on day 1
        cash        = 0.0

        # Track which thresholds have been triggered since last reference reset
        triggered: set[int] = set()

        records = []

        for dt, price in prices.items():
            if price <= 0 or np.isnan(price):
                continue

            pct_change = (price - ref_price) / ref_price * 100
            trade_type = "hold"
            trade_qty  = 0.0
            trade_val  = 0.0

            # Evaluate rules in order of threshold severity
            # Buys: most negative first (buy more on bigger dip)
            buy_rules  = sorted([r for r in RULES if r[1] == "buy"],  key=lambda x: x[0])
            sell_rules = sorted([r for r in RULES if r[1] == "sell"], key=lambda x: x[0])

            # Check buy triggers (price fell enough)
            for threshold, action, fraction in buy_rules:
                if pct_change <= threshold and threshold not in triggered:
                    amount = cash * fraction
                    if amount > commission and cash > commission:
                        amount = min(amount, cash)
                        qty    = (amount - commission) / price
                        if qty > 0:
                            shares    += qty
                            cash      -= amount
                            trade_type = f"BUY {fraction*100:.0f}%cash"
                            trade_qty  = qty
                            trade_val  = amount
                            triggered.add(threshold)
                            if ref_reset:
                                ref_price = price
                                triggered = set()
                    break  # one rule per day

            # Check sell triggers (price rose enough) — only if no buy happened
            if trade_type == "hold":
                for threshold, action, fraction in sell_rules:
                    if pct_change >= threshold and threshold not in triggered:
                        qty = shares * fraction
                        if qty > 0:
                            proceeds   = qty * price - commission
                            shares    -= qty
                            cash      += max(proceeds, 0)
                            trade_type = f"SELL {fraction*100:.0f}%pos"
                            trade_qty  = qty
                            trade_val  = proceeds
                            triggered.add(threshold)
                            if ref_reset:
                                ref_price = price
                                triggered = set()
                        break  # one rule per day

            portfolio_value = cash + shares * price

            records.append({
                "date":            dt,
                "price":           price,
                "pct_from_ref":    pct_change,
                "ref_price":       ref_price,
                "cash":            cash,
                "shares":          shares,
                "portfolio_value": portfolio_value,
                "trade":           trade_type,
                "trade_qty":       trade_qty,
                "trade_value":     trade_val,
            })

        return pd.DataFrame(records).set_index("date")


    # ── Load data ─────────────────────────────────────────────────────────────────
    @st.cache_data(ttl=3600)
    def load_data(start: date, end: date, ticker: str) -> pd.Series:
        t  = yf.Ticker(ticker)
        df = t.history(start=str(start), end=str(end))
        return df["Close"].dropna()


    with st.spinner(f"Fetching {ticker_choice} data…"):
        try:
            prices = load_data(start_date, end_date, ticker_choice)
        except Exception as e:
            st.error(f"Could not load data: {e}")
            st.stop()

    if prices.empty:
        st.error("No price data returned. Check your date range.")
        st.stop()

    # Normalise index to date (remove tz)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    # ── Run strategy ──────────────────────────────────────────────────────────────
    result = apply_rules(prices, float(initial_cash), ref_reset_on_trade, commission)

    # Buy-and-hold benchmark
    bah_shares = initial_cash / prices.iloc[0]
    bah_values = bah_shares * prices
    bah_values.name = "buy_and_hold"

    # ── KPI metrics ───────────────────────────────────────────────────────────────
    final_val   = result["portfolio_value"].iloc[-1]
    bah_final   = bah_values.iloc[-1]
    total_ret   = (final_val - initial_cash) / initial_cash * 100
    bah_ret     = (bah_final - initial_cash) / initial_cash * 100
    n_years     = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr        = ((final_val / initial_cash) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    bah_cagr    = ((bah_final / initial_cash) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    # Max drawdown for strategy
    roll_max    = result["portfolio_value"].cummax()
    drawdown    = (result["portfolio_value"] - roll_max) / roll_max * 100
    max_dd      = drawdown.min()

    roll_max_bah = bah_values.cummax()
    dd_bah       = (bah_values - roll_max_bah) / roll_max_bah * 100
    max_dd_bah   = dd_bah.min()

    trades      = result[result["trade"] != "hold"]
    n_trades    = len(trades)

    # ── Layout: KPIs ─────────────────────────────────────────────────────────────
    st.subheader("📊 Results")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Final Value",        f"${final_val:,.0f}",   f"{total_ret:+.1f}%")
    c2.metric("B&H Final Value",    f"${bah_final:,.0f}",   f"{bah_ret:+.1f}%")
    c3.metric("CAGR (Strategy)",    f"{cagr:.2f}%")
    c4.metric("CAGR (B&H)",         f"{bah_cagr:.2f}%")
    c5.metric("# Trades",           str(n_trades))

    c6, c7, c8 = st.columns(3)
    c6.metric("Max Drawdown (Strategy)", f"{max_dd:.1f}%")
    c7.metric("Max Drawdown (B&H)",      f"{max_dd_bah:.1f}%")
    c8.metric("Years simulated",         f"{n_years:.1f}")

    # ── Portfolio value chart ─────────────────────────────────────────────────────
    st.subheader("📈 Portfolio Value Over Time")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Portfolio Value", "Drawdown (%)"),
    )

    fig.add_trace(
        go.Scatter(
            x=result.index, y=result["portfolio_value"],
            name="Father's Strategy",
            line=dict(color="#00b4d8", width=2),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bah_values.index, y=bah_values,
            name="Buy & Hold",
            line=dict(color="#f77f00", width=2, dash="dash"),
        ),
        row=1, col=1,
    )

    # Mark trade points
    buys  = result[result["trade"].str.startswith("BUY",  na=False)]
    sells = result[result["trade"].str.startswith("SELL", na=False)]

    fig.add_trace(
        go.Scatter(
            x=buys.index, y=buys["portfolio_value"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=9, color="#2dc653"),
            customdata=np.stack([buys["trade"], buys["trade_value"], buys["trade_qty"], buys["price"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Action: %{customdata[0]}<br>"
                "Spent: $%{customdata[1]:,.0f}<br>"
                "Shares bought: %{customdata[2]:,.4f}<br>"
                "Price: $%{customdata[3]:,.2f}<br>"
                "Portfolio: $%{y:,.0f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sells.index, y=sells["portfolio_value"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=9, color="#e63946"),
            customdata=np.stack([sells["trade"], sells["trade_value"], sells["trade_qty"], sells["price"]], axis=1),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Action: %{customdata[0]}<br>"
                "Proceeds: $%{customdata[1]:,.0f}<br>"
                "Shares sold: %{customdata[2]:,.4f}<br>"
                "Price: $%{customdata[3]:,.2f}<br>"
                "Portfolio: $%{y:,.0f}<extra></extra>"
            ),
        ),
        row=1, col=1,
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown,
            name="Strategy DD",
            fill="tozeroy",
            line=dict(color="#00b4d8", width=1),
            fillcolor="rgba(0,180,216,0.15)",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dd_bah.index, y=dd_bah,
            name="B&H DD",
            line=dict(color="#f77f00", width=1, dash="dash"),
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # ── Cash vs Equity allocation ─────────────────────────────────────────────────
    st.subheader("💰 Cash vs Equity Allocation Over Time")

    equity = result["shares"] * result["price"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=result.index, y=equity,
        name="Equity", stackgroup="one",
        line=dict(width=0), fillcolor="rgba(0,180,216,0.6)",
    ))
    fig2.add_trace(go.Scatter(
        x=result.index, y=result["cash"],
        name="Cash", stackgroup="one",
        line=dict(width=0), fillcolor="rgba(247,127,0,0.6)",
    ))
    fig2.update_layout(
        height=300, template="plotly_dark",
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
        yaxis_title="Value ($)",
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── Trade log ─────────────────────────────────────────────────────────────────
    with st.expander(f"📋 Trade Log ({n_trades} trades)"):
        trade_log = trades[["price", "pct_from_ref", "trade", "trade_qty", "trade_value", "cash", "portfolio_value"]].copy()
        trade_log.index = trade_log.index.strftime("%Y-%m-%d")
        trade_log.columns = ["Price", "% from Ref", "Action", "Qty", "Value ($)", "Cash ($)", "Portfolio ($)"]
        trade_log = trade_log.round(2)
        st.dataframe(trade_log, use_container_width=True)

    # ── Raw data download ─────────────────────────────────────────────────────────
    csv = result.reset_index().to_csv(index=False).encode()
    st.download_button(
        "⬇️ Download full simulation data (CSV)",
        data=csv,
        file_name="fathers_rules_backtest.csv",
        mime="text/csv",
    )

    st.info("Inspired by https://x.com/gudanglifehack/status/2038396420018040973")

if __name__ =="__main__":
    main()