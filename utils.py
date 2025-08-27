import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import platform



#def get_data_old(choice, period, interval, window):
def get_data_yfinance(choice, interval, period="3m",start="2015-01-01", window=7):

    """Retreat the data from Yahoo Finance.

    Kept this one instead the one in utils.py because both functions are not compatible.
    """
    data = yf.download(
        tickers=(choice),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
    )

    if interval in ["1d","5d","1wk","1mo","3mo"]:
        index_field = "Date"
    elif interval in [ "1m","2m","5m","15m","30m","60m","90m","1h"]:
        index_field = "Datetime"
    df = pd.DataFrame(data)
    
    # if platform.processor() != "":
    #     # local
    #     df[f"{choice}_Close"]   = df["Close"]
        
    # else:
    df.columns = ['_'.join(col) for col in df.columns]
    df["Close"] = df[f"{choice}_Close"]
   
    if len(df) == 0:
        st.error(f"No data or wrong input - {choice}")
        
        st.stop()
    else:
        df['rownumber'] = np.arange(len(df))
    column_name = "close_" + choice
    df[column_name] = df["Close"]
    df = df.reset_index()
 
    try:
        df["Date"] = df[index_field]
    except:
        df["Date"] = df["index"]
    #df = df[["rownumber","Date", column_name, "Close"]]
  
   
     # Add a new column 'sma' with 3-period SMA
    df['sma'] = df[column_name].rolling(window=window, center=True).mean()
    return df


@st.cache_data()
def get_data_yfinance_non_working(choice, interval, period="3m",start="2015-01-01"):
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
    data = yf.download(tickers=choice, start=start, interval=interval)# group_by='ticker', auto_adjust=True, prepost=False, ignore_tz=True)
   
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