# https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from skmisc.loess import loess
from typing import Tuple, List

pd.options.mode.chained_assignment = None


def interface() -> Tuple[str, str, str, float, float, int, bool, int, float]:
    """
    Create the Streamlit sidebar interface for user input.

    Returns:
        Tuple containing user inputs for ticker choice, period, interval, z1, z2, window, center_boll, initial investment, and transaction fee.
    """
    choice_ = st.sidebar.selectbox(
        "Which ticker", ["BTC-USD", "ETH-USD", "AMZN", "OTHER"]
    )

    if choice_ == "OTHER":
        choice = st.sidebar.text_input("Ticker", "AAPL")
    else:
        choice = choice_
    period = st.sidebar.selectbox(
        "Period", ["3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], 2
    )  # "1d", "5d", "1mo",
    interval = st.sidebar.selectbox("Interval", ["1d", "5d", "1wk", "1mo", "3mo"], 0)

    st.sidebar.markdown("## Bollinger Bands")
    z1 = st.sidebar.number_input("Z-value 1 (used for strategy)", 0.0, 3.0, 1.0)
    z2 = z1
    if z1 > z2:
        st.warning("Z1 has to be smaller than Z2")
        st.stop()
    wdw = 20  #  int(st.sidebar.number_input("Window for bollinger", 2, 60, 20))
    center_boll = (
        True  # st.sidebar.selectbox("Center bollinger", [True, False], index=0)
    )
    counter_under_treshold =  st.sidebar.number_input("Counter under treshold", 0, 100, 1,help="How many times the close value has to be under the band before a buy action")	
    counter_over_treshold =   st.sidebar.number_input("Counter over treshold", 0, 100, 1, help="How many times the close value has to be above the band before a sell action")
    initial_investment = st.sidebar.number_input(
        "Initial investment", 0, 1000000000, 1000
    )
    transaction_fee = st.sidebar.number_input("Transaction fee", 0.0, 100.0, 0.25) / 100
    return (
        choice,
        period,
        interval,
        z1,
        z2,
        wdw,
        center_boll,
        initial_investment,
        transaction_fee,counter_under_treshold, counter_over_treshold
    )


def calculate_various_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate various columns for the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Modified DataFrame with additional columns.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    if "rownumber" not in df.columns:
        df.insert(0, "rownumber", range(1, len(df) + 1))

    df = df.reset_index()
    return df


def do_bollinger(
    df: pd.DataFrame, z1: float, z2: float, wdw: int, center_boll: bool
) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        z1 (float): Z-value for the first Bollinger Band.
        z2 (float): Z-value for the second Bollinger Band.
        wdw (int): Window size for the Bollinger Bands.
        center_boll (bool): Whether to center the Bollinger Bands.

    Returns:
        pd.DataFrame: Modified DataFrame with Bollinger Bands columns.
    """

    def sma(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate the Simple Moving Average (SMA) for a given data series.

        Args:
            data (pd.Series): The input data series.
            window (int): The window size for calculating the SMA.

        Returns:
            pd.Series: The SMA of the input data series.
        """
        return data.rolling(window=window, center=center_boll).mean()

    def do_lowess(df: pd.DataFrame) -> np.ndarray:
        """
        Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the 'Close' column of the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing the 'Close' column.

        Returns:
            np.ndarray: The smoothed values from the LOWESS algorithm.
        """
        x = np.asarray(df["rownumber"], dtype=np.float64)
        y = np.asarray(df["Close"], dtype=np.float64)
        span = 32 / len(y)  # 32 corresponds to a window of 20 when using SMA
        loess_ = loess(x, y)
        loess_.model.span = span
        loess_.model.degree = 1
        loess_.control.iterations = 1
        loess_.control.surface = "direct"
        loess_.control.statistics = "approximate"
        loess_.fit()
        pred = loess_.predict(x, stderror=True)
        return pred.values

    def calculate_bollinger_bands(
        data: pd.Series, sma: pd.Series, window: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate the Bollinger Bands for a given data series.

        Args:
            data (pd.Series): The input data series.
            sma (pd.Series): The Simple Moving Average (SMA) of the data series.
            window (int): The window size for calculating the Bollinger Bands.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series, pd.Series]: The lower and upper Bollinger Bands for z1 and z2.
        """
        # std = data.rolling(window=window, center=False).std()
        std = data.ewm(span=window, adjust=False).std()
        upper_bb_1 = sma + std * z1
        lower_bb_1 = sma - std * z1
        upper_bb_2 = sma + std * z2
        lower_bb_2 = sma - std * z2
        return lower_bb_1, lower_bb_2, upper_bb_1, upper_bb_2

    df.loc[:, "boll_center_sma"] = sma(df.loc[:, "Close"], wdw)
    df.loc[:, "boll_center"] = do_lowess(df)

    # Calculate all bands at once
    low1, low2, high1, high2 = calculate_bollinger_bands(
        df["Close"], df["boll_center"], wdw
    )

    # Assign directly to original DataFrame using .loc
    df.loc[:, "boll_low_1"] = low1
    df.loc[:, "boll_low_2"] = low2
    df.loc[:, "boll_high_1"] = high1
    df.loc[:, "boll_high_2"] = high2
    return df


def implement_bb_strategy(
    close: pd.Series,
    bol_low_1: pd.Series,
    bol_high_1: pd.Series,
    buy_price: pd.Series,
    buy_price_history: list,
    sell_price: pd.Series,
    status: pd.Series,
    bb_signal: pd.Series,
    counter_under: int,
    counter_over: int,
    counter_under_treshold:int,
    counter_over_treshold:int,
) -> Tuple[float, float, int]:
    """
    Imp lement Bollinger Bands strategy.

    Args:
        close (pd.Series): Input series of closing prices.
        bol_low_1 (pd.Series): Lower Bollinger Band.
        bol_high_1 (pd.Series): Upper Bollinger Band.
        status : list in possession (1) or not (0)
        bb_signal : list moments of which you buy or sell
        counter_under (int) : how many times the close value is under the under band
        counter_over (int) : how many times the close value is under the under band
        counter_under_treshold (int):how many times the close value has to be under the band before a buy action
        counter_over_treshold (int) : how many times the close value has to be above the band before a sell action
    Returns:
        Tuple containing lists of buy price, sell price, and signal.
    """
    buy_price_old = buy_price_history[-1]

    buy_price_history_ = buy_price_old
   

    # Check
    #   if the price crosses below the lower Bollinger Band
    #   the status is not in possession
    #   and that it is already [counter_under_treshold] times below the lower band

    i = len(close) - 1
    if close[i - 1] > bol_low_1[i - 1] and close[i] < bol_low_1[i]:

        if status[-1] == 0:
            if counter_under >= counter_under_treshold:
                buy_price_ = close[i]
                buy_price_history_ = close[i]

                sell_price_ = np.nan
                status_ = 1
                bb_signal_ = 1
                counter_under = 0
            else:
                buy_price_ = np.nan
                sell_price_ = np.nan
                status_ = status[-1]
                bb_signal_ = 0
                counter_under += 1
        else:
            buy_price_ = np.nan
            sell_price_ = np.nan
            status_ = status[-1]
            bb_signal_ = 0

    # Check
    #   if the price crosses above the higher Bollinger Band
    #   the status is in possession
    #   the price is above the buying price
    #   and that it is already [counter_over_treshold] times above the higher band

    elif close[i - 1] < bol_high_1[i - 1] and close[i] > bol_high_1[i]:

        if (status[-1] == 1) and (close[i] > buy_price_old):
            if counter_over >= counter_over_treshold:

                buy_price_ = np.nan
                sell_price_ = close[i]
                status_ = 0
                bb_signal_ = -1
                counter_over = 0
            else:
                buy_price_ = np.nan
                sell_price_ = np.nan
                bb_signal_ = 0
                status_ = status[-1]
                counter_over += 1
        else:
            buy_price_ = np.nan
            sell_price_ = np.nan
            bb_signal_ = 0
            status_ = status[-1]
    else:
        buy_price_ = np.nan
        sell_price_ = np.nan
        bb_signal_ = 0
        status_ = status[-1]
    buy_price.append(buy_price_)
    buy_price_history.append(buy_price_history_)
    sell_price.append(sell_price_)
    bb_signal.append(bb_signal_)
    status.append(status_)

    return (
        buy_price,
        buy_price_history,
        sell_price,
        status,
        bb_signal,
        counter_under,
        counter_over,counter_under_treshold, 
        counter_over_treshold
    )


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
    Calculate the portfolio value over time based on buy and sell signals.

    Args:
        dates (pd.Series): Series of dates.
        buy_price (List[float]): List of buy prices.
        sell_price (List[float]): List of sell prices.
        bb_signal (List[int]): List of buy/sell signals.
        close (pd.Series): Series of closing prices.
        initial_investment (float): Initial investment amount.
        transaction_fee (float): Transaction fee as a fraction.

    Returns:
        Tuple containing lists of portfolio values when holding and when sold.
    """
    portfolio_value = 0
    cash = initial_investment
    shares = 0
    portfolio_values = []
    portfolio_values_sell = []
    shares_list = []
    for date, buy, sell, close, signal in zip(
        dates, buy_price, sell_price, close, bb_signal
    ):

        if signal == 1:  # Buy signal
            shares = (cash * (1 - transaction_fee)) / buy
            cash = 0
        elif signal == -1:  # Sell signal
            cash = shares * sell * (1 - transaction_fee)
            shares = 0

        portfolio_value = cash + (shares * close)
        portfolio_value_sell = cash if shares == 0 else None

        portfolio_values.append(portfolio_value)
        portfolio_values_sell.append(portfolio_value_sell)
        shares_list.append(shares)

    return portfolio_values, portfolio_values_sell


def plot_boll(
    df: pd.DataFrame,
    choice: str,
    buy_price: List[float],
    sell_price: List[float],
    bb_signal: List[int],
    base: int,
    min,
    max,
) -> None:
    """
    Plot Bollinger Bands and buy/sell signals.

    Args:
        df (pd.DataFrame): Input DataFrame.
        choice (str): Ticker choice.
        buy_price (List[float]): List of buy prices.
        sell_price (List[float]): List of sell prices.
        bb_signal (List[int]): List of buy/sell signals.
        base (int) : the base
    """
    buy = go.Scatter(
        name="BUY",
        x=df["Date"],
        y=buy_price,
        mode="markers",
        marker_symbol="triangle-up",
        opacity=0.4,
        marker_line_color="midnightblue",
        marker_color="green",
        marker_line_width=0,
        marker_size=11,
    )

    sell = go.Scatter(
        name="SELL",
        x=df["Date"],
        y=sell_price,
        mode="markers",
        marker_symbol="triangle-down",
        opacity=0.4,
        marker_line_color="midnightblue",
        marker_color="red",
        marker_line_width=0,
        marker_size=11,
    )

    boll_low_1 = go.Scatter(
        name="boll low 1",
        x=df["Date"],
        y=df["boll_low_1"],
        mode="lines",
        line=dict(width=0.5, color="rgba(255, 255, 0, 0.0)"),
        fillcolor="rgba(255,255,0, 0.4)",
        fill="tonexty",
    )

    boll = go.Scatter(
        name="boll_loess",
        x=df["Date"],
        y=df["boll_center"],
        mode="lines",
        line=dict(width=0.9, color="rgba(255,165,0,1)"),
        fillcolor="rgba(255,255,0,0.4)",
        fill="tonexty",
    )

    boll_high_1 = go.Scatter(
        name="boll high 1",
        x=df["Date"],
        y=df["boll_high_1"],
        mode="lines",
        line=dict(width=0.5, color="rgba(255, 255, 0, 0.0)"),
        fillcolor="rgba(255,255,0, 0.2)",
        # fill='tonexty'
    )

    close = go.Scatter(
        name="Close",
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        line=dict(width=1, color="rgba(0,0,0, 1)"),
        fillcolor="rgba(68, 68, 68, 0.2)",
    )
    close_start = go.Scatter(
        x=df["Date"][:base],
        y=df["Close"][:base],
        mode="lines",
        line=dict(width=1, color="rgba(255,0,0, 1)"),
        fillcolor="rgba(68, 68, 68, 0.2)",
    )

    data = [boll_high_1, boll, boll_low_1, close, close_start, buy, sell]

    layout = go.Layout(
        yaxis=dict(title="USD", range=[min, max]),
        title=f"Bollinger bands - {choice}",
    )

    fig1 = go.Figure(data=data, layout=layout)
    fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    # st.plotly_chart(fig1, use_container_width=True)

    return fig1


def plot_value_portfolio(
    dates: pd.Series, portfolio_values: List[float], portfolio_values_sell: List[float]
) -> None:
    """
    Plot the portfolio value over time.

    Args:
        dates (pd.Series): Series of dates.
        portfolio_values (List[float]): List of portfolio values when holding.
        portfolio_values_sell (List[float]): List of portfolio values when selling.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates, y=portfolio_values, mode="lines", name="Portfolio Value Hold"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=portfolio_values_sell,
            mode="lines",
            name="Portfolio Value all sold",
        )
    )

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value (€)",
        template="plotly_white",
    )

    # st.plotly_chart(fig)
    return fig


def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.header("Y Finance charts / strategy using Bollinger bands")
    (
        choice,
        period,
        interval,
        z1,
        z2,
        wdw,
        center_boll,
        initial_investment,
        transaction_fee,counter_under_treshold, counter_over_treshold
    ) = interface()

    data = yf.download(
        tickers=(choice),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
    )
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error("No data or wrong input")
        st.stop()
    df["rownumber"] = np.arange(len(df))

    df = calculate_various_columns_df(df)
    base = 45
    buy_price = [np.nan] * (base - 1)
    buy_price_history = [np.nan] * (base - 1)
    sell_price = [np.nan] * (base - 1)
    bb_signal = [np.nan] * (base - 1)

    status = [0] * (base - 1)

    df["Date"] = pd.to_datetime(df["Date"])
    start_date = df["Date"].iloc[0]
    end_date = df["Date"].iloc[-1]

    # Create a complete date range from the start to the end date
    full_date_range = pd.date_range(start=df["Date"].iloc[0], end=df["Date"].iloc[-1])

    # https://medium.com/data-storytelling-corner/better-un-refugee-storytelling-with-animated-streamlit-data-visuals-879e0254ddbf
    # if 'current_year' not in st.session_state:
    #     st.session_state.current_year = base
    #     start = base
    # else:
    #     start = st.session_state.current_year
    # if 'is_playing' not in st.session_state:
    #     st.session_state.is_playing = False
    # # Play and Pause buttons side-by-side
    # col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("Play"):
    #         st.session_state.is_playing = True
    # with col2:
    #     if st.button("Pause"):
    #         st.session_state.is_playing =
    st.session_state.is_playing = True
    start = base
    placehholder = st.empty()
    placeholder_value = st.empty()
    min, max = df["Close"].min(), df["Close"].max()
    counter_under, counter_over = 0, 0
    while st.session_state.is_playing:
        for i in range(start, len(df)):
            buy_price, sell_price, bb_signal, fig, counter_under, counter_over,counter_under_treshold, counter_over_treshold = (
                generate_bollinger_graph_wrapper(
                    choice,
                    z1,
                    z2,
                    wdw,
                    center_boll,
                    df,
                    base,
                    buy_price,
                    buy_price_history,
                    sell_price,
                    bb_signal,
                    status,
                    full_date_range,
                    min,
                    max,
                    i,
                    counter_under,
                    counter_over,
                    counter_under_treshold,
                    counter_over_treshold
                )
            )
            placehholder.plotly_chart(fig, use_container_width=True)
            st.session_state.current_year = i
        st.session_state.is_playing = False
    dates = df["Date"]
    close = df["Close"]

    # # Display current year's visuals when not animating
    # i = st.session_state.current_year
    # buy_price, sell_price, bb_signal, fig = generate_bollinger_graph_wrapper(choice, z1, z2, wdw, center_boll, df, base, buy_price, buy_price_history, sell_price, bb_signal, status, full_date_range, i)
    # placehholder.plotly_chart(fig, use_container_width=True)

    portfolio_values, portfolio_values_sell = calculate_portfolio_value(
        dates,
        buy_price,
        sell_price,
        bb_signal,
        close,
        initial_investment,
        transaction_fee,
    )
    fig_values = plot_value_portfolio(dates, portfolio_values, portfolio_values_sell)
    placeholder_value.plotly_chart(fig_values, use_container_width=True)

    rendement = portfolio_values[-1] / portfolio_values[0] * 100

    rendement_coin = df["Close"].iloc[-1] / df["Close"].iloc[0] * 100

    number_of_years = round((end_date - start_date).days / 365.25, 2)
    cagr = round(
        ((portfolio_values[-1] / portfolio_values[0]) ** (1 / number_of_years) - 1)
        * 100,
        2,
    )

    cagr_coin = round(
        ((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (1 / number_of_years) - 1)
        * 100,
        2,
    )

    st.info(
        f"Value: {round(rendement, 2)}% in {number_of_years} year(s). Compound ROI: {cagr}%"
    )
    st.info(
        f"Value coin: {round(rendement_coin, 2)}% in {number_of_years} year(s). Compound ROI coin: {cagr_coin}%"
    )

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        "Inspired by : <a href='https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a'>I Needed Money, So I Wrote An Algorithm</a> <br>"
        "Also used : <a href='https://medium.com/codex/algorithmic-trading-with-bollinger-bands-in-python-1b0a00c9ef99'>Algorithmic Trading with Bollinger Bands in Python</a> <br>"
        "Sourcecode : <a href='https://github.com/rcsmit/streamlit_scripts/blob/main/yfinance_info.py' target='_blank'>github.com/rcsmit</a><br>"
        "How-to tutorial : <a href='https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d/' target='_blank'>rcsmit.medium.com</a><br>"
        "Read <a href='https://pypi.org/project/yfinance/'>disclaimer</a> at of Yfinance"
    )

    disclaimer_text = """
    **Disclaimer:** The information provided in this application is for educational purposes only and does not constitute financial advice. 
    Investing in financial markets involves risk, and you should consult with a qualified financial advisor before making any investment decisions. 
    The author of this application is not responsible for any financial losses that may occur as a result of using this information.
    """

    st.markdown(tekst, unsafe_allow_html=True)
    st.markdown(disclaimer_text, unsafe_allow_html=True)


def generate_bollinger_graph_wrapper(
    choice,
    z1,
    z2,
    wdw,
    center_boll,
    df,
    base,
    buy_price,
    buy_price_history,
    sell_price,
    bb_signal,
    status,
    full_date_range,
    min,
    max,
    i,
    counter_under,
    counter_over,
    counter_under_treshold,
    counter_over_treshold
):
    df_ = df[:i].copy()

    df_ = do_bollinger(df_, z1, z2, wdw, center_boll)
    (
        buy_price,
        buy_price_history,
        sell_price,
        status,
        bb_signal,
        counter_under,
        counter_over,counter_under_treshold, 
        counter_over_treshold
    ) = implement_bb_strategy(
        df_["Close"].to_list(),
        df_["boll_low_1"],
        df_["boll_high_1"],
        buy_price,
        buy_price_history,
        sell_price,
        status,
        bb_signal,
        counter_under,
        counter_over,
        counter_under_treshold, 
        counter_over_treshold
    )

    # Reindex the DataFrame to include all dates in the full date range
    df_ = (
        df_.set_index("Date").reindex(full_date_range).rename_axis("Date").reset_index()
    )

    # Fill missing values only for the 'Date' column
    # df_['Date'].fillna(method='ffill', inplace=True)
    # df_['Date'] = df_['Date'].fillna(method='ffill')
    df_["Date"].ffill()

    fig = plot_boll(df_, choice, buy_price, sell_price, bb_signal, base, min, max)
    return buy_price, sell_price, bb_signal, fig, counter_under, counter_over,counter_under_treshold, counter_over_treshold


if __name__ == "__main__":
    main()
