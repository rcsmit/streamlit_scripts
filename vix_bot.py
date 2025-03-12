# https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from typing import Tuple, List
from utils import get_data_yfinance
pd.options.mode.chained_assignment = None


def last_actions(merged_df):
    """Perform the last actions on the DataFrame."""
    merged_df["total_posession"] = merged_df["value_spy"] + merged_df["value_qqq"]+merged_df["wallet"]
    # Delete the last row in the DataFrame because the QQQ and SPY doesn't have a Close value
    merged_df = merged_df.iloc[:-1]
    return merged_df

def make_dataframe():
    """Create a DataFrame with the financial data."""
    x=0
    start= st.sidebar.date_input("Start date", value=pd.to_datetime("2022-01-01"))
    for ticker_choice in ["%5EVIX", "SPY", "QQQ"]:
        df = get_data_yfinance(ticker_choice, "1d", period="5y", start=start)
        if len(df) == 0:
            st.error("No data or wrong input")
            st.stop()
        df["rownumber"] = np.arange(len(df))
        df["Ticker"] = ticker_choice  # Add a column to identify the ticker
       
        if x==1:
            merged_df = pd.merge(left=merged_df, right=df, on="Date",how="outer").copy(deep=True)
         
        else:
            merged_df=df
            x=1
        
    merged_df["buy_signal"] = 0
    merged_df["sell_signal"] = 0
    merged_df["in_posession"] = 0
    merged_df["number_spy"] = 0
    merged_df["value_spy"] = 0

    merged_df["number_qqq"] = 0
    merged_df["value_qqq"] = 0
    merged_df["wallet"] = 0
    start_worth_qqq = 10000
    start_worth_spy = 10000
   
    merged_df.at[0, "wallet"] = start_worth_qqq+start_worth_spy

    merged_df["%5EVIX_sma20"] = merged_df["%5EVIX_Close"].rolling(window=20).mean()
    merged_df["%5EVIX_std20"] = merged_df["%5EVIX_Close"].rolling(window=20).std()
    merged_df["%5EVIX_sma50"] = merged_df["%5EVIX_Close"].rolling(window=50).mean()
    
    merged_df["%5EVIX_bollinger_up"] = merged_df["%5EVIX_sma20"] + (merged_df["%5EVIX_std20"] * 2)
    merged_df["%5EVIX_bollinger_down"] = merged_df["%5EVIX_sma20"] - (merged_df["%5EVIX_std20"] * 2)

    return merged_df

def plot_data(df: pd.DataFrame,  method: str):
    """Wrapper to plot the financial data."""
    if method == "Omero":
        st.info("This strategy involves buying SPY and QQQ when the VIX is above 30 and selling when it is below 15.")
    elif method == "Moving Average Crossovers":
        st.info("This strategy calculates two moving averages of the VIX: a short-term (e.g., 20-day) and a long-term (e.g., 50-day) average. When the short-term average rises above the long-term, it is typically seen as an indicator of increasing volatility, leading traders to sell SPY and QQQ. Conversely, when the short-term average falls below the long-term, it may suggest declining volatility, signaling potential buying opportunities for SPY and QQQ.")
    elif method == "Bollinger Bands":
        st.info("A Bollinger Band is a volatility indicator that consists of three lines: a middle band (typically a simple moving average) and two outer bands set a certain number of standard deviations away from the middle band. If the VIX moves above its upper Bollinger Band, it may signal overbought conditions, suggesting a potential market downturn and prompting traders to sell SPY and QQQ. On the other hand, if the VIX drops below the lower Bollinger Band, it could indicate oversold conditions, pointing to a possible market rebound and a buying opportunity for SPY and QQQ.")
    else:
        st.error("Error in method")
        st.stop()

    plot_data_(df, "%5EVIX_Close", method)
    col1,col2 = st.columns(2)
    with col1:
        plot_data_(df, "QQQ_Close", method)
    with col2:
        plot_data_(df, "SPY_Close", method)


    # col3,col4,col5= st.columns(3)
    # with col3:
    #     plot_data_(df, "wallet", method)
    # with col4:
    #     plot_data_(df, "value_spy", method)
    # with col5:
    #     plot_data_(df, "value_qqq", method)
    
    plot_data_(df,"total_posession", method)
    

def plot_data_(df: pd.DataFrame, col:str, method: str) -> None:
    """Plot stock data with buy/sell signals."""
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[col], mode='lines', name=f'{col} Close', line=dict(width=1)))
    if col == "%5EVIX_Close":

        if method == "Omero":
            pass
        elif method == "Moving Average Crossovers": 
            fig.add_trace(go.Scatter(x=df["Date"], y=df["%5EVIX_sma20"], mode='lines', name='VIX sma20', line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["%5EVIX_sma50"], mode='lines', name='VIX sma50', line=dict(width=1)))
            
        elif method == "Bollinger Bands":
        

            fig.add_trace(go.Scatter(x=df["Date"], y=df["%5EVIX_bollinger_up"], mode='lines', name='Bollinger Up', line=dict(width=1)))
            fig.add_trace(go.Scatter(x=df["Date"], y=df["%5EVIX_bollinger_down"], mode='lines', name='Bollinger Down', line=dict(width=1)))
            
            # Add light yellow color to the area between the Bollinger bands
            fig.add_trace(go.Scatter(
                x=pd.concat([df["Date"], df["Date"][::-1]]),
                y=pd.concat([df["%5EVIX_bollinger_up"], df["%5EVIX_bollinger_down"][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 255, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name='Bollinger Bands'
            ))
        else:
            st.error("Error in method")
            st.stop()   
    fig.add_trace(go.Scatter(x=df[df["buy_signal"] == 1]["Date"], y=df[df["buy_signal"] == 1][col],
                                mode='markers', marker_symbol="triangle-up", marker_color="green",
                                marker_size=10, name='Buy Signal'))
    fig.add_trace(go.Scatter(x=df[df["sell_signal"] == 1]["Date"], y=df[df["sell_signal"] == 1][col],
                                mode='markers', marker_symbol="triangle-down", marker_color="red",
                                marker_size=10, name='Sell Signal'))
    fig.update_layout(title=f'{col}', xaxis_title='Date', yaxis_title=f'{col}')
    st.plotly_chart(fig)

def calculate_roi(df: pd.DataFrame) -> None:
    """Calculate and display ROI and CAGR."""
    
    for w in ["total_posession", "QQQ_Close", "SPY_Close"]:
        rendement = df[w].iloc[-1] / df[w].iloc[0] * 100
        number_of_years = round((df["Date"].iloc[-1] - df["Date"].iloc[0]).days / 365.25, 2)
        cagr = ((df[w].iloc[-1] / df[w].iloc[0]) ** (1 / number_of_years) - 1) * 100
        st.info(f"{w} | Value: {rendement:.2f}% in {number_of_years} year(s). Compound ROI: {cagr:.2f}%")

def keeping(merged_df, index, row):
    merged_df.at[index, "in_posession"] = 1
    merged_df.at[index,"number_spy"] = merged_df.at[index-1,"number_spy"]
    merged_df.at[index,"number_qqq"] = merged_df.at[index-1,"number_qqq"]
    merged_df.at[index,"value_spy"] = row["SPY_Close"]*merged_df.at[index,"number_spy"]
    merged_df.at[index,"value_qqq"] = row["QQQ_Close"]*merged_df.at[index,"number_qqq"]

def selling(merged_df, index):
    merged_df.at[index, "sell_signal"] = 1
    merged_df.at[index, "in_posession"] = 0
    merged_df.at[index, "wallet"] = merged_df.at[index-1, "value_spy"] + merged_df.at[index-1, "value_qqq"]
    merged_df.at[index,"number_spy"] = 0
    merged_df.at[index,"number_qqq"] = 0
    merged_df.at[index,"value_spy"] = 0
    merged_df.at[index,"value_qqq"] = 0

def buying(merged_df, index, row):
    merged_df.at[index, "buy_signal"] = 1
    merged_df.at[index, "in_posession"] = 1
    merged_df.at[index,"number_spy"] = (merged_df.at[index-1,"wallet"] /2)/ row["SPY_Close"]
    merged_df.at[index,"number_qqq"] = (merged_df.at[index-1,"wallet"] /2)/ row["QQQ_Close"]
    merged_df.at[index,"value_spy"] = row["SPY_Close"]*merged_df.at[index,"number_spy"]
    merged_df.at[index,"value_qqq"] = row["QQQ_Close"]*merged_df.at[index,"number_qqq"]
  
   

def apply_omero(merged_df):
    """Apply the Omero strategy."""
    for index, row in merged_df.iterrows():
        vix = row["%5EVIX_Close"] # VIX closing price
        if index>0:
            if vix:
                if merged_df.at[index-1, "in_posession"] == 0:
                    if vix > 30:
                        # I buy SPY and QQQ
                        buying(merged_df, index, row)
                    else:
                        merged_df.at[index, "wallet"] = merged_df.at[index-1, "wallet"] 
                else: # I have
                    if vix < 15:
                        # I sell
                        selling(merged_df, index)
                        
                    else:
                        # I KEEP
                        keeping(merged_df, index, row)

def apply_moving_average_crossvers(merged_df):
    for index, row in merged_df.iterrows():
        vix = row["%5EVIX_Close"] # VIX closing price
        sma20 = row["%5EVIX_sma20"]
        sma50 = row["%5EVIX_sma50"]
        
       
        if index>0:
            if vix:
                if merged_df.at[index-1, "in_posession"] == 0:
                    if sma50 > sma20:
                        # I buy SPY and QQQ
                        buying(merged_df, index, row)
                    else:
                        merged_df.at[index, "wallet"] = merged_df.at[index-1, "wallet"] 
                else:
                    if sma20 > sma50:  # I have
                        # I sell
                        selling(merged_df, index)   
                    else:
                        # I KEEP
                        keeping(merged_df, index, row)


def apply_bollinger(merged_df):
    """ Apply the Bollinger bands strategy.
        https://paperswithbacktest.com/wiki/using-vix-to-trade-spy-and-sp-500
    """
    for index, row in merged_df.iterrows():
        vix = row["%5EVIX_Close"] # VIX closing price
        up = row["%5EVIX_bollinger_up"]
        down = row["%5EVIX_bollinger_down"]
        if index>0:
            if vix:
                if merged_df.at[index-1, "in_posession"] == 0:
                    if vix > up:
                        # I buy SPY and QQQ
                        buying(merged_df, index, row)
                    else:
                        merged_df.at[index, "wallet"] = merged_df.at[index-1, "wallet"] 
                else:
                    if vix < down:  # I have
                        # I sell
                        selling(merged_df, index)   
                    else:
                        # I KEEP
                        keeping(merged_df, index, row)



def main() -> None:
    """
    Main function to run the Streamlit application.
    """
    st.header("Using VIX to trade spy and sp-500")

    merged_df = make_dataframe()
    method = st.sidebar.selectbox("Select a method", ["Omero", "Moving Average Crossovers", "Bollinger Bands"])
    if method == "Omero":
        apply_omero(merged_df)
    elif method == "Moving Average Crossovers": 
        apply_moving_average_crossvers(merged_df)       
    elif method == "Bollinger Bands":
        apply_bollinger(merged_df)
   
    merged_df = last_actions(merged_df)

    plot_data(merged_df, method)
    calculate_roi(merged_df)
    st.info("https://paperswithbacktest.com/wiki/using-vix-to-trade-spy-and-sp-500")

if __name__ == "__main__":
    main()
