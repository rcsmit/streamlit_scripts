#https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a

# script to find out which window size for lowess corresponds to the SMA

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from skmisc.loess import loess
from typing import Tuple, List

def interface() -> Tuple[str, str, str, float, float, int, bool, int, float]:
    """
    Create the Streamlit sidebar interface for user input.

    Returns:
        Tuple containing user inputs for ticker choice, period, interval, z1, z2, window, center_boll, initial investment, and transaction fee.
    """
    choice_ = st.sidebar.selectbox("Which ticker", ["BTC-USD", "ETH-USD", "AMZN", "OTHER"])

    if choice_ == "OTHER":
        choice = st.sidebar.text_input("Ticker", "AAPL")
    else:
        choice = choice_
    period = st.sidebar.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"], 5)
    interval = st.sidebar.selectbox("Interval", ["1d", "5d", "1wk", "1mo", "3mo"], 0)


    return choice, period, interval

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
    if 'rownumber' not in df.columns:
        df.insert(0, 'rownumber', range(1, len(df) + 1))
    
    df = df.reset_index()
    return df

def do_sma(data: pd.Series, window: int) -> pd.Series:
        """
        Calculate the Simple Moving Average (SMA) for a given data series.

        Args:
            data (pd.Series): The input data series.
            window (int): The window size for calculating the SMA.

        Returns:
            pd.Series: The SMA of the input data series.
        """
        return data.rolling(window=window, center=True).mean()
    
def do_lowess(df: pd.DataFrame, window:int) -> np.ndarray:
    """
    Perform LOWESS (Locally Weighted Scatterplot Smoothing) on the 'Close' column of the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'Close' column.

    Returns:
        np.ndarray: The smoothed values from the LOWESS algorithm.
    """
    x = np.asarray(df['rownumber'], dtype=np.float64)
    y = np.asarray(df['Close'], dtype=np.float64)
    span = window / len(y)
    l = loess(x, y)
    l.model.span = span
    l.model.degree = 1
    l.control.iterations = 1
    l.control.surface = "direct"
    l.control.statistics = "exact"
    l.fit()
    pred = l.predict(x, stderror=True)
    return pred.values

def plot(df: pd.DataFrame, lowess: List[float], sma: List[float]) -> None:
    """
    Plot 
    Args:
        df (pd.DataFrame): Input DataFrame.
        
        lowess (List[float]): List of lowess.
        sma (List[float]): List of sma.
        
    """
    

    close = go.Scatter(
            name="boll_loess",
            x=df["rownumber"],
            y=df["Close"],
            mode='lines',
            line=dict(width=0.9, color='rgba(0,165,0,1)'),
            
        )
    boll = go.Scatter(
        name="boll_loess",
        x=df["rownumber"],
        y=lowess,
        mode='lines',
        line=dict(width=0.9, color='rgba(255,165,0,1)'),
        
    )

    boll_sma = go.Scatter(
        name="boll_sma",
        x=df["rownumber"],
        y=sma,
        mode='lines',
        line=dict(width=0.9, color='rgba(255,0,0,1)'),
        
    )

  

    data = [close, boll,boll_sma]

    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=f"Close, LOWESS, SMA "
    )

    fig1 = go.Figure(data=data, layout=layout)
    fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig1, use_container_width=True)

def calculate_correlation(df: pd.DataFrame, wdw_lowess: int, wdw_sma: int, verbose: bool = True) -> float:
    """
    Calculate the correlation between LOWESS and SMA for the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the 'Close' column.
        wdw_lowess (int): The window size for the LOWESS algorithm.
        wdw_sma (int): The window size for the SMA calculation.
        verbose (bool): If True, plot the LOWESS and SMA. Default is True.

    Returns:
        float: The correlation coefficient between LOWESS and SMA.
    """
    lowess= do_lowess(df, wdw_lowess)
    sma = do_sma(df["Close"],wdw_sma)
    # Convert SMA to list and slice LOWESS to match SMA length
    sma_list = sma.dropna().tolist()
    sma_list = sma.dropna().tolist()
    if len(sma_list) % 2 != 0:
        sma_list = sma_list[1:]
    len_diff = len(lowess) - len(sma_list)
    cut_off = len_diff // 2
    lowess_list = lowess[cut_off:len(lowess) - cut_off]
    

    # Calculate the correlation between the two lists
    correlation = np.corrcoef(lowess_list, sma_list)[0, 1]

    # Display the correlation
    if verbose:
        st.write(f"Correlation between LOWESS and SMA: {correlation:.8f}")
    
        plot(df,lowess,sma)
    return correlation

def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.header("Find the correlation between LOWESS and SMA for a given stock")

    choice, period, interval= interface()

    data = yf.download(tickers=(choice), period=period, interval=interval, group_by='ticker', auto_adjust=True, prepost=False)
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error("No data or wrong input")
        st.stop()
    df['rownumber'] = np.arange(len(df))
    
    # Initialize an empty DataFrame to store the correlation values
    correlation_table = pd.DataFrame(columns=['wdw_lowess', 'wdw_sma', 'correlation'])

    # Iterate over the ranges for wdw_lowess and wdw_sma
    wdw_sma = 20
    print (wdw_sma)
    for wdw_lowess in range(10,70):
        print (wdw_lowess)
        correlation = calculate_correlation(df, wdw_lowess, wdw_sma, False)
            # Append the correlation value to the DataFrame
        new_row = pd.DataFrame({'wdw_lowess': [wdw_lowess], 'wdw_sma': [wdw_sma], 'correlation': [correlation]})
        correlation_table = pd.concat([correlation_table, new_row], ignore_index=True)

    # Display the correlation table
    st.write(correlation_table)
    import plotly.express as px

    import plotly.express as px

    # Create a line graph of the correlation table
    fig = px.line(
        correlation_table,
        x='wdw_lowess',
        y='correlation',
        color='wdw_sma',
        title='Correlation between LOWESS and SMA'
    )

    # Update layout
    fig.update_layout(
        xaxis_title='Window Size for LOWESS',
        yaxis_title='Correlation'
    )

    # Display the line graph in Streamlit
    st.plotly_chart(fig)
if __name__ == "__main__":
    main()
