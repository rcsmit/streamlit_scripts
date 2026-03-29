# https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a

# CHANGES vs original:
# 1. LOESS center removed → replaced by causal SMA (center=False always)
#    LOESS used the entire dataset to fit the smoothing curve, giving it
#    knowledge of future prices. A causal SMA only looks back.
# 2. center_boll parameter removed from rolling std as well (always False)
# 3. implement_bb_strategy now trades at next day's Open price, not same-day Close
#    (you cannot actually buy at the close price that triggered the signal)
# 4. calculate_portfolio_value updated to accept and use open_prices

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from typing import Tuple, List

from utils import get_data_yfinance


def interface() -> Tuple[str, str, str, float, int, float, int, float]:
    """
    Create the Streamlit sidebar interface for user input.

    Returns:
        Tuple containing user inputs for ticker choice, period, interval,
        z1, window, initial_investment, and transaction_fee.
    """
    choice_ = st.sidebar.selectbox(
        "Which ticker", ["BTC-USD", "ETH-USD", "AMZN", "OTHER"]
    )

    if choice_ == "OTHER":
        choice = st.sidebar.text_input("Ticker", "AAPL")
    else:
        choice = choice_

    period = st.sidebar.selectbox(
        "Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        5,
    )
    interval = st.sidebar.selectbox(
        "Interval", ["1d", "5d", "1wk", "1mo", "3mo"], 0
    )

    st.sidebar.markdown("## Bollinger Bands")
    z1 = st.sidebar.number_input("Z-value (band width)", 0.0, 3.0, 2.0)
    wdw = int(st.sidebar.number_input("Window for Bollinger", 2, 60, 20))

    initial_investment = st.sidebar.number_input(
        "Initial investment", 0, 1_000_000_000, 1000
    )
    transaction_fee = (
        st.sidebar.number_input("Transaction fee (%)", 0.0, 100.0, 0.25) / 100
    )
    return choice, period, interval, z1, wdw, initial_investment, transaction_fee


def calculate_various_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column structure and add a row number column.

    Args:
        df: Raw DataFrame from yfinance.

    Returns:
        Cleaned DataFrame with rownumber column and reset index.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    if "rownumber" not in df.columns:
        df.insert(0, "rownumber", range(1, len(df) + 1))
    df = df.reset_index()
    return df


def do_bollinger(
    df: pd.DataFrame, z1: float, wdw: int
) -> pd.DataFrame:
    """
    Calculate causal Bollinger Bands using a look-back SMA as center.

    The original code used LOESS smoothing fitted on the full dataset,
    which introduced look-ahead bias. This version uses a simple causal
    SMA (center=False) so only past data informs each band value.

    Args:
        df: Input DataFrame with a 'Close' column.
        z1: Number of standard deviations for the bands.
        wdw: Rolling window size (look-back only).

    Returns:
        DataFrame with added columns: boll_center, boll_low_1, boll_high_1,
        boll_low_2, boll_high_2.
    """
    close = df["Close"]

    # Causal SMA: only looks at the past `wdw` candles
    sma = close.rolling(window=wdw, center=False).mean()
    std = close.rolling(window=wdw, center=False).std()

    df["boll_center"] = sma
    df["boll_low_1"] = sma - std * z1
    df["boll_high_1"] = sma + std * z1
    # z2 kept as z1 * 2 for the outer shading (informational only, not traded)
    df["boll_low_2"] = sma - std * z1 * 2
    df["boll_high_2"] = sma + std * z1 * 2

    return df


def implement_bb_strategy(
    close: pd.Series,
    open_: pd.Series,
    dates: pd.Series,
    boll_low_1: pd.Series,
    boll_high_1: pd.Series,
) -> Tuple[List[float], List[float], List[int], pd.Series]:
    """
    Implement Bollinger Bands strategy, executing trades at next-bar Open.

    Signal is detected when close crosses a band on bar i.
    The trade is executed at Open of bar i+1.
    All output lists and the returned dates are aligned to the EXECUTION bar
    (i+1), so that plot markers and portfolio values appear on the correct date.

    Args:
        close: Series of closing prices.
        open_: Series of opening prices (used for trade execution).
        dates: Series of dates aligned to close/open.
        boll_low_1: Lower Bollinger Band.
        boll_high_1: Upper Bollinger Band.

    Returns:
        Tuple of:
          buy_price  – execution price or NaN, one entry per bar from index 2..n
          sell_price – execution price or NaN, same alignment
          bb_signal  – 1 / -1 / 0, same alignment
          exec_dates – dates of the execution bars (i+1)
    """
    buy_price: List[float] = []
    sell_price: List[float] = []
    bb_signal: List[int] = []
    exec_date_indices: List[int] = []

    close_arr = close.to_numpy()
    open_arr = open_.to_numpy()
    low_arr = boll_low_1.to_numpy()
    high_arr = boll_high_1.to_numpy()
    dates_arr = dates.to_numpy()

    # Always buy on day 1 (bar 0) at open — enter the market immediately
    signal = 1
    buy_price.append(open_arr[0])
    sell_price.append(np.nan)
    bb_signal.append(1)
    exec_date_indices.append(0)

    # i   = signal bar  (close crossed band)
    # i+1 = execution bar (open used for trade)
    for i in range(1, len(close_arr) - 1):
        crossed_below = (
            close_arr[i - 1] > low_arr[i - 1] and close_arr[i] < low_arr[i]
        )
        crossed_above = (
            close_arr[i - 1] < high_arr[i - 1] and close_arr[i] > high_arr[i]
        )

        exec_price = open_arr[i + 1]
        exec_date_indices.append(i + 1)

        if crossed_below and signal != 1:
            buy_price.append(exec_price)
            sell_price.append(np.nan)
            signal = 1
            bb_signal.append(signal)
        elif crossed_above and signal != -1:
            buy_price.append(np.nan)
            sell_price.append(exec_price)
            signal = -1
            bb_signal.append(signal)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)

    exec_dates = pd.Series(dates_arr[exec_date_indices])
    return buy_price, sell_price, bb_signal, exec_dates, exec_date_indices


def calculate_portfolio_value(
    dates: pd.Series,
    buy_price: List[float],
    sell_price: List[float],
    bb_signal: List[int],
    close: pd.Series,
    initial_investment: float,
    transaction_fee: float,
) -> Tuple[List[float], List[float]]:
    """
    Calculate portfolio value over time based on buy/sell signals.

    Trades are executed at the prices in buy_price / sell_price
    (which are next-bar opens in the fixed version).

    Args:
        dates: Series of dates (aligned to the signal bars).
        buy_price: Execution price for buy signals (NaN otherwise).
        sell_price: Execution price for sell signals (NaN otherwise).
        bb_signal: Signal list (1=buy, -1=sell, 0=hold).
        close: Closing prices for mark-to-market valuation.
        initial_investment: Starting capital in euros.
        transaction_fee: Fee fraction per trade (e.g. 0.0025 for 0.25%).

    Returns:
        Tuple of portfolio_values (mark-to-market) and portfolio_values_sell
        (as-if-liquidated-now).
    """
    cash = float(initial_investment)
    shares = 0.0
    portfolio_values: List[float] = []
    portfolio_values_sell: List[float] = []

    for buy, sell, close_price, signal in zip(
        buy_price, sell_price, close, bb_signal
    ):
        if signal == 1 and not np.isnan(buy):
            shares = (cash * (1 - transaction_fee)) / buy
            cash = 0.0
        elif signal == -1 and shares > 0 and not np.isnan(sell):
            cash = shares * sell * (1 - transaction_fee)
            shares = 0.0

        portfolio_value = cash + shares * close_price if shares > 0 else cash
        # Liquidation value: what you'd get if you sold everything now at close price
        portfolio_value_sell = cash + shares * close_price * (1 - transaction_fee) if shares > 0 else cash

        portfolio_values.append(portfolio_value)
        portfolio_values_sell.append(portfolio_value_sell)

    return portfolio_values, portfolio_values_sell


def plot_boll(
    df: pd.DataFrame,
    choice: str,
    buy_price: List[float],
    sell_price: List[float],
    signal_dates: pd.Series,
) -> None:
    """
    Plot Bollinger Bands with buy/sell markers.

    Args:
        df: Full DataFrame with band columns.
        choice: Ticker symbol for the chart title.
        buy_price: Buy execution prices (NaN where no buy).
        sell_price: Sell execution prices (NaN where no sell).
        signal_dates: Dates aligned to the signal list.
    """
    traces = [
        go.Scatter(
            name="boll high 2",
            x=df["Date"], y=df["boll_high_2"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.8)"),
            fillcolor="rgba(255,255,0,0.0)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll high 1",
            x=df["Date"], y=df["boll_high_1"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.0)"),
            fillcolor="rgba(255,255,0,0.2)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll_center (SMA)",
            x=df["Date"], y=df["boll_center"],
            mode="lines",
            line=dict(width=0.9, color="rgba(255,165,0,1)"),
            fillcolor="rgba(255,255,0,0.4)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll low 1",
            x=df["Date"], y=df["boll_low_1"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.0)"),
            fillcolor="rgba(255,255,0,0.4)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll low 2",
            x=df["Date"], y=df["boll_low_2"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.8)"),
            fillcolor="rgba(255,255,0,0.2)",
            fill="tonexty",
        ),
        go.Scatter(
            name="Close",
            x=df["Date"], y=df["Close"],
            mode="lines",
            line=dict(width=1, color="rgba(0,0,0,1)"),
        ),
        go.Scatter(
            name="BUY",
            x=signal_dates, y=buy_price,
            mode="markers",
            marker_symbol="triangle-up",
            opacity=0.6,
            marker_color="green",
            marker_size=11,
        ),
        go.Scatter(
            name="SELL",
            x=signal_dates, y=sell_price,
            mode="markers",
            marker_symbol="triangle-down",
            opacity=0.6,
            marker_color="red",
            marker_size=11,
        ),
    ]

    fig = go.Figure(
        data=traces,
        layout=go.Layout(
            title=f"Bollinger Bands — {choice}",
            yaxis=dict(title="USD"),
        ),
    )
    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, width='stretch')


def plot_value_portfolio(
    dates: pd.Series,
    portfolio_values: List[float],
) -> None:
    """
    Plot portfolio value over time.

    Args:
        dates: Execution bar dates.
        portfolio_values: Mark-to-market portfolio values.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates, y=portfolio_values,
            mode="lines",
            name="Portfolio Value (€)",
            line=dict(color="blue"),
        )
    )
    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (€)",
        xaxis=dict(tickformat="%d-%m-%Y"),
        template="plotly_white",
    )
    st.plotly_chart(fig, width='stretch')


def plot_combined(
    df: pd.DataFrame,
    choice: str,
    exec_dates: pd.Series,
    buy_price: List[float],
    sell_price: List[float],
    bb_signal: List[int],
    portfolio_values: List[float],
) -> None:
    """
    Combined chart: Bollinger Bands + close + buy/sell markers (left y-axis, USD)
    and portfolio value (right y-axis, €), with vertical lines at trade moments.

    Args:
        df: Full DataFrame with band columns.
        choice: Ticker symbol for title.
        exec_dates: Execution bar dates aligned to signal lists.
        buy_price: Buy prices (NaN where no buy).
        sell_price: Sell prices (NaN where no sell).
        bb_signal: Signal list (1=buy, -1=sell, 0=hold).
        portfolio_values: Mark-to-market portfolio values.
    """
    traces = [
        go.Scatter(
            name="boll high 2",
            x=df["Date"], y=df["boll_high_2"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.8)"),
            fillcolor="rgba(255,255,0,0.0)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll high 1",
            x=df["Date"], y=df["boll_high_1"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.0)"),
            fillcolor="rgba(255,255,0,0.2)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll_center (SMA)",
            x=df["Date"], y=df["boll_center"],
            mode="lines",
            line=dict(width=0.9, color="rgba(255,165,0,1)"),
            fillcolor="rgba(255,255,0,0.4)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll low 1",
            x=df["Date"], y=df["boll_low_1"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.0)"),
            fillcolor="rgba(255,255,0,0.4)",
            fill="tonexty",
        ),
        go.Scatter(
            name="boll low 2",
            x=df["Date"], y=df["boll_low_2"],
            mode="lines",
            line=dict(width=0.5, color="rgba(255,255,0,0.8)"),
            fillcolor="rgba(255,255,0,0.2)",
            fill="tonexty",
        ),
        go.Scatter(
            name="Close",
            x=df["Date"], y=df["Close"],
            mode="lines",
            line=dict(width=1, color="rgba(0,0,0,1)"),
        ),
        go.Scatter(
            name="BUY",
            x=exec_dates, y=buy_price,
            mode="markers",
            marker_symbol="triangle-up",
            opacity=0.6,
            marker_color="green",
            marker_size=11,
        ),
        go.Scatter(
            name="SELL",
            x=exec_dates, y=sell_price,
            mode="markers",
            marker_symbol="triangle-down",
            opacity=0.6,
            marker_color="red",
            marker_size=11,
        ),
        go.Scatter(
            name="Portfolio Value (€)",
            x=exec_dates,
            y=portfolio_values,
            mode="lines",
            line=dict(color="blue", width=1.5),
            yaxis="y2",
        ),
    ]

    # Vertical dotted lines at every trade
    shapes = []
    for signal, date in zip(bb_signal, list(exec_dates)):
        if signal == 1:
            shapes.append(dict(
                type="line", x0=date, x1=date, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="green", width=1, dash="dot"),
            ))
        elif signal == -1:
            shapes.append(dict(
                type="line", x0=date, x1=date, y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color="red", width=1, dash="dot"),
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Bollinger Bands & Portfolio Value — {choice}",
        xaxis=dict(title="Date", tickformat="%d-%m-%Y"),
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(
            title="Portfolio Value (€)",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_white",
        shapes=shapes,
    )
    st.plotly_chart(fig, width='stretch')





def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.header("YFinance — Bollinger Band Strategy ")


    choice, period, interval, z1, wdw, initial_investment, transaction_fee = interface()

    df = get_data_yfinance(choice, interval, period, None)
    if len(df) == 0:
        st.error("No data or wrong ticker.")
        st.stop()

    df["rownumber"] = np.arange(len(df))
    df = calculate_various_columns_df(df)
    df = do_bollinger(df, z1, wdw)

    # Signal detected at bar i (close crosses band).
    # Trade executed at bar i+1 open. Bar 0 always triggers a buy.
    # exec_date_indices lets us fetch the exact close for mark-to-market.
    buy_price, sell_price, bb_signal, exec_dates, exec_date_indices = implement_bb_strategy(
        df["Close"], df["Open"], df["Date"], df["boll_low_1"], df["boll_high_1"]
    )

    # Close prices at the exact execution bars for mark-to-market valuation.
    close_aligned = df["Close"].iloc[exec_date_indices].reset_index(drop=True)

    portfolio_values, portfolio_values_sell = calculate_portfolio_value(
        exec_dates,
        buy_price,
        sell_price,
        bb_signal,
        close_aligned,
        initial_investment,
        transaction_fee,
    )

    if portfolio_values:
        df["Date"] = pd.to_datetime(df["Date"])
        start_date = df["Date"].iloc[0]
        end_date = df["Date"].iloc[-1]
        number_of_years = round((end_date - start_date).days / 365.25, 2)

        # Strategy metrics
        strat_value_end = portfolio_values[-1]
        strat_rendement = round(strat_value_end / initial_investment * 100, 2)
        strat_cagr = round(
            ((strat_value_end / initial_investment) ** (1 / number_of_years) - 1) * 100, 2
        )

        # Buy-and-hold metrics: buy at first close, sell at last close
        price_start = df["Close"].iloc[0]
        price_end = df["Close"].iloc[-1]
        bah_value_end = initial_investment * (price_end / price_start)
        bah_rendement = round(bah_value_end / initial_investment * 100, 2)
        bah_cagr = round(
            ((bah_value_end / initial_investment) ** (1 / number_of_years) - 1) * 100, 2
        )

        col1, col2 = st.columns(2)
        col1.metric(
            label=f"Strategy — {number_of_years}y",
            value=f"€ {round(strat_value_end, 2)}",
            delta=f"{strat_rendement}% total | CAGR {strat_cagr}%",
        )
        col2.metric(
            label=f"Buy & Hold {choice} — {number_of_years}y",
            value=f"€ {round(bah_value_end, 2)}",
            delta=f"{bah_rendement}% total | CAGR {bah_cagr}%",
        )

    plot_boll(df, choice, buy_price, sell_price, exec_dates)
    plot_value_portfolio(exec_dates, portfolio_values)
    plot_combined(df, choice, exec_dates, buy_price, sell_price, bb_signal, portfolio_values)

    tekst = (
        "<style>.infobox{background-color:lightblue;padding:5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit "
        "(<a href='http://www.twitter.com/rcsmit' target='_blank'>@rcsmit</a>)<br>"
        "Inspired by: <a href='https://medium.com/the-financial-journal/"
        "the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a'>"
        "I Needed Money, So I Wrote An Algorithm</a><br>"
        "Sourcecode: <a href='https://github.com/rcsmit/streamlit_scripts/blob/main/"
        "yfinance_info.py' target='_blank'>github.com/rcsmit</a></div>"
    )
    disclaimer = (
        "**Disclaimer:** For educational purposes only. Not financial advice. "
        "Always consult a qualified advisor before investing. "
        "See also [yfinance disclaimer](https://pypi.org/project/yfinance/)."
    )

    st.markdown(tekst, unsafe_allow_html=True)
    st.markdown(disclaimer)

if __name__ == "__main__":
    main()