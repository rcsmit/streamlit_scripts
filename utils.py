import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import platform

@st.cache_data()
def get_data_yfinance(choice, interval, period="3m",start="2015-01-01"):
    """
    Retrieve financial data from Yahoo Finance.

    Args:
        choice (str): Ticker symbol for the financial data.
        interval (str): Data interval (e.g., '1d', '1wk', '1mo').
        start (str): Start date for the financial data (default is '2015-01-01').
    Returns:
        pd.DataFrame: Preprocessed financial data with 'Date' as the index and 'Month' column added.
    """
    #data = yf.download(tickers=(choice), period=period, interval=interval, group_by='ticker', auto_adjust=True, prepost=False)
    data = yf.download(tickers=choice, period=period, start=start, interval=interval, group_by='ticker', auto_adjust=True, prepost=False, ignore_tz=True)
    
    df = pd.DataFrame(data)
  
    if df.empty:
        st.error(f"No data or wrong input - {choice}")
        return None
  
   
    if platform.processor() != "":
        # local
        # df[f"{choice}_Close"]   = df["Close"]
        df.columns = ['_'.join(col) for col in df.columns]
        df["Close"] = df[f"{choice}_Close"]
        
    else:
        df.columns = ['_'.join(col) for col in df.columns]
        df["Close"] = df[f"{choice}_Close"]
        
    
  
  
    df[f"close_{choice}"]   = df["Close"]

    df['rownumber'] = np.arange(len(df))
    df["Koers"] = df["Close"]
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df.get("Datetime", df["Date"]))
    
    return df