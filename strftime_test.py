import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt

def get_data(choice,  interval):
    data = yf.download(tickers=(choice), start="2021-11-28",interval=interval,group_by='ticker',auto_adjust=True,prepost=False)
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error(f"No data or wrong input - {choice}")
        df = None
    else:
        df['rownumber'] = np.arange(len(df))

    df = df.reset_index()

    df = df[["Date", "Close"]]
    df["Date_y_m_d"] = df["Date"].dt.strftime("%Y-%m-%d")

    return df




def main():
    df = get_data("BTC-USD", "1d")
    st.write(df)
    st.write(pd.show_versions())
    

if __name__ == "__main__":
    main()
