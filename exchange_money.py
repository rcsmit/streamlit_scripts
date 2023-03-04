# WHAT IS THE BEST WAY TO EXCHANGE MONEY IN THAILAND?
# Exchange cash or take it from ATM
#
# Calculation to find the Break Even Point


import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly


def find_rate_yfinance():
    choice = "EURTHB=X"  # st.sidebar.text_input("Which ticker", "BTC-USD")
    period = "3mo"  # st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 5)
    interval = "1d"  # st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],8)
    ticker = yf.Tickers(choice)

    data = yf.download(
        tickers=(choice),
        period=period,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        prepost=False,
    )
    # print (data)
    df = pd.DataFrame(data)
    # print (df.to_string())
    rate_yfinance = df.iloc[-1, -2]  # last available close-rate
    return rate_yfinance


cost_creditcard_fix = 4.5  # in euro
cost_creditcard_variable = 1.02  # in %

cost_debitcard_fix = 0  # in euro
cost_debitcard_variable = 1.012  # in %

cost_atm = 220  # in baht


rate_yfinance = find_rate_yfinance() #36.62453  # find_rate_yfinance()
rate_cc = rate_yfinance / cost_creditcard_variable
rate_dc = rate_yfinance / cost_debitcard_variable

rate_xe = 35.77
rate_street = rate_yfinance * 0.98 # 0.98637 # 35.11
rate_with_conversion = 33.66 / 36.90399 * rate_yfinance
print(
    f"rate_yfinance={rate_yfinance} / rate_street={rate_street} / / rate_dc={rate_dc} / rate_cc={rate_cc} / rate_with_conversion=rate_with_conversion={rate_with_conversion}"
)


def from_baht():
    y = []

    for i in range(0, 35000, 5000):
        cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_baht(i)
        y.append(
            [
                i,
                round(cc, 2),
                round(dc, 2),
                round(cc_with_conv, 2),
                round(dc_with_conv, 2),
                round(street, 2),
            ]
        )
        if round(street, 2) == round(cc, 2):
            print(f"Street is same as cc {i} - {cc}")
        if round(street, 2) == round(dc, 2):
            print(f"Street is same as dc {i} - {dc}")

    total_df_baht = pd.DataFrame(
        y,
        columns=[
            "baht",
            "creditcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
    )
    print(total_df_baht)
    fig = px.line(
        total_df_baht,
        x="baht",
        y=["creditcard", "debitcard", "street"],
        title="From Baht to Euro",
    )
    # plotly.offline.init_notebook_mode(connected=True)
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)

def from_euro():
    x=[]
    for i in range(0, 1000, 10):
        cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_euro(i)
        if cc<0: cc = 0
        if dc<0: dc = 0
        if cc_with_conv<0: cc_with_conv = 0
        if dc_with_conv<0: dc_with_conv = 0
        
        x.append(
            [
                i,
                round(cc, 2),
                round(dc, 2),
                cc_with_conv,
                dc_with_conv,
                round(street, 2),
            ]
        )
        if int(street) == int(cc):
            print(f"Street is same as cc {i} - {cc}")
        if int(street) == int(dc):
            print(f"Street is same as dc {i} - {dc}")

    total_df = pd.DataFrame(
        x,
        columns=[
            "euro",
            "creditcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
    )
    print(total_df)
    fig = px.line(
        total_df,
        x="euro",
        y=[
            "creditcard",
            "debitcard",
            "debitcard",
            "creditcard_with_conv",
            "debitcard_with_conv",
            "street",
        ],
        title="From Euro to Baht",
    )
    # plotly.offline.init_notebook_mode(connected=True)
    # plotly.offline.plot(fig)

    st.plotly_chart(fig)

def calculate_from_euro(i):
    cc = (i - cost_creditcard_fix - (cost_atm / rate_cc)) * rate_cc
    dc = (i - cost_debitcard_fix - (cost_atm / rate_dc)) * rate_dc
    cc_with_conv = (
        i - cost_creditcard_fix - (cost_atm / rate_with_conversion)
    ) * rate_with_conversion
    dc_with_conv = (
        i - cost_debitcard_fix - (cost_atm / rate_with_conversion)
    ) * rate_with_conversion

    street = i * rate_street
    return cc, dc, cc_with_conv, dc_with_conv, street


def calculate_from_baht(i):
    cc = (((i + cost_atm) / rate_cc)) + cost_creditcard_fix
    dc = (((i + cost_atm) / rate_dc)) + cost_debitcard_fix
    cc_with_conv = (i + cost_atm) / rate_with_conversion + cost_creditcard_fix
    dc_with_conv = (i + cost_atm) / rate_with_conversion * (
        1 + (cost_debitcard_variable / 100)
    ) + cost_debitcard_fix
    street = i / rate_street
    return cc, dc, cc_with_conv, dc_with_conv, street


def how_much_euro_do_i_get_for_x_baht(i):
    cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_baht(i  )
    print(f"BAHT: {i} -> EURO cc={cc} dc={dc} cc_with_conv={cc_with_conv} dc_with_conv={dc_with_conv} street={street}")


def how_much_baht_do_i_get_for_x_euro(i):
    cc, dc, cc_with_conv, dc_with_conv, street = calculate_from_euro(i)
    print(f"EURO {i} -> BAHT : cc={cc} dc={dc} cc_with_conv={cc_with_conv} dc_with_conv={dc_with_conv} street={street}")


from_baht()
from_euro()
how_much_euro_do_i_get_for_x_baht(30000)
how_much_baht_do_i_get_for_x_euro(846.13)


# 56 dec 1646 exchange rate Rabobank 36.62453   835,03
# 26 decv exchange rate XE 36.9278
# 26 dec excange with conversion 33.66 = factor 0.9099
# factor 0.9917875

# 18 dec exchagne rate rabobank 36.88591 25220  36.88591  691.93 = 36.449
# 19 dec exchange rate 36.915156
# factor 0,9992077
