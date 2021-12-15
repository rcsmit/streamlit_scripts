#https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a

import yfinance as yf
import pandas as pd

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

def get_data(choice, period, interval):
    ticker = yf.Tickers(choice)

    data = yf.download(tickers=(choice),period=period,interval=interval,group_by='ticker',auto_adjust=True,prepost=False)
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error(f"No data or wrong input - {choice}")
        df = None
    else:
        df['rownumber'] = np.arange(len(df))

    return df

def print_dataframe(df):
    print(df)

def find_slope_scipy(x_,y_):
    m, b, r_value, p_value, std_err = stats.linregress(x_, y_)
    r_sq = r_value**2
    return m,b,r_sq

def calculate_various_columns_df(df, wdw, center_boll, z1,z2):
    df = df.reset_index()
    std = np.std(df['Close'])
    mean = df['Close'].mean()
    x = list(range(0,len(df)))
    y = df["Close"].to_list()
    m,b,r_sq = find_slope_scipy(x,y)

    if z1 >= z2:
        st.warning("Z1 has to be smaller than Z2")
        st.stop()
    df['trendline'] = (df['rownumber'] *m +b)
    df['trendline_low_1'] = (df['rownumber'] *m +b) - z1 * std
    df['trendline_high_1'] = (df['rownumber'] *m +b) + z1 * std
    df['trendline_low_2'] = (df['rownumber'] *m +b) - z2 * std
    df['trendline_high_2'] = (df['rownumber'] *m +b) + z2 * std
    df['z_from_mean'] = (df['Close'] - mean) / std
    df["z_from_trendline"] =  (df['Close'] - df['trendline']) / std
    df = do_bollinger(df, z1, z2, wdw, center_boll)
    return df, std, mean, m, b



def do_bollinger(df, z1, z2, wdw, center_boll):
    #    # https://medium.com/codex/algorithmic-trading-with-bollinger-bands-in-python-1b0a00c9ef99
    def sma(data, window):
        sma = data.rolling(window = window, center=center_boll).mean()
        return sma
    def bb(data, sma, window):
        std = data.rolling(window = window, center=center_boll).std()
        upper_bb_1 = sma + std * z1
        lower_bb_1 = sma - std * z1

        upper_bb_2 = sma + std * z2
        lower_bb_2 = sma - std * z2

        return lower_bb_1, lower_bb_2, upper_bb_1, upper_bb_2

    df['boll_center'] = sma(df['Close'], wdw)
    df['boll_low_1'], df['boll_low_2'], df['boll_high_1'], df['boll_high_2'] = bb(df['Close'], df['boll_center'], wdw)
    return df
def implement_bb_strategy(df, bol_low_1, bol_high_1):
    #https://medium.com/codex/how-to-calculate-bollinger-bands-of-a-stock-with-python-f9f7d1184fc3
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(1,len(df)):
        if df[i-1] > bol_low_1[i-1] and df[i] < bol_low_1[i]:
            # buy
            if signal != 1:

                buy_price.append(df[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif df[i-1] < bol_high_1[i-1] and df[i] > bol_high_1[i]:
            # sell
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(df[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)


    return buy_price, sell_price, bb_signal, signal

def plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label):

    buy = go.Scatter(
        name='BUY',
        x=df[x_as_label],
        y=buy_price ,
            mode="markers",  marker_symbol='triangle-up', opacity=0.4,
                    marker_line_color="midnightblue", marker_color="green",
                    marker_line_width=0, marker_size=11,
                    )



    sell = go.Scatter(
        name='SELL',
        x=df[x_as_label],
        y=sell_price ,
        mode="markers",marker_symbol='triangle-down',opacity=0.4,
                    marker_line_color="midnightblue", marker_color="red",
                    marker_line_width=0, marker_size=11,
                    )




    boll_low_2 = go.Scatter(
        name='boll low 2',
        x=df[x_as_label],
        y=df['boll_low_2'] ,
        mode='lines',
        line=dict(width=0.5,
                color="rgba(255, 255, 0, 0.8)"),
        fillcolor='rgba(255,255,0,0.2)',
        fill='tonexty')

    boll_low_1 = go.Scatter(
        name='boll low 1',
        x=df[x_as_label],
        y=df['boll_low_1'] ,
        mode='lines',
        line=dict(width=0.5,
                color="rgba(255, 255, 0, 0.0)"),
        fillcolor='rgba(255,255,0, 0.4)',
        fill='tonexty')

    boll = go.Scatter(
        name="boll",
        x=df[x_as_label],
        y=df["boll_center"],
        mode='lines',
        line=dict(width=0.9,color='rgba(255,165,0,1)'),
        fillcolor='rgba(255,255,0,0.4)',
        fill='tonexty'
        )

    boll_high_1 = go.Scatter(
        name='boll high 1',
        x=df[x_as_label],
        y=df['boll_high_1'] ,
        mode='lines',
        line=dict(width=0.5,
                color="rgba(255, 255, 0, 0.0)"),
        fillcolor='rgba(255,255,0, 0.2)',
            fill='tonexty'
        )
    boll_high_2 = go.Scatter(
        name='boll high 2',
        x=df[x_as_label],
        y=df['boll_high_2'] ,
        mode='lines',
        line=dict(width=0.5,
                color="rgba(255, 255, 0, 0.8)"),
        fillcolor='rgba(255,255,0, 0.0)',
            fill='tonexty'

        )



    close = go.Scatter(
        name="Close",
        x=df[x_as_label],
        y=df["Close"],
        mode='lines',
        line=dict(width=1,color='rgba(0,0,0, 1)'),
        fillcolor='rgba(68, 68, 68, 0.2)',
        )

    data = [boll_high_2,boll_high_1, boll, boll_low_1,boll_low_2,close, buy, sell ]

    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=f"Bollinger bands - {choice}")
        #, xaxis=dict(tickformat="%d-%m")
    fig1 = go.Figure(data=data, layout=layout)
    min=df["Close"].min() * 0.9975
    max=df["Close"].max() *1.0025

    #fig1.update_yaxes(autorange=True)
    fig1.update_yaxes(range=[min,max])
    #fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

    #fig.show()
    st.plotly_chart(fig1, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.header("Crypto dashboard of René Smit")

    col1, col2 = st.columns(2)
    with col1:
        st.sidebar.write("LEFT")
        period_left = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 1)
        interval_left =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],5)
    with col2:
        st.sidebar.write ("RIGHT")
        period_right = st.sidebar.selectbox("Period ", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 0)
        interval_right =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],0)



    time_zone = st.sidebar.selectbox("Tijdzone", ["Europe/Amsterdam", "Asia/Bangkok"],1)
    wdw = int( st.sidebar.number_input("Window Moving Average",2,60,20))
    center_boll = st.sidebar.selectbox("Center Moving Average", [True, False], index=1)
    z1 = st.sidebar.number_input("Z-value 1", 0.0,3.0,1.0)
    z2 = st.sidebar.number_input("Z-value 2", 0.0,3.0,1.96)
    # https://finance.yahoo.com/lookup?s=-usd&.tsrc=fin-srch
    choicelist = ["BTC-USD", "ETH-USD", "XRP-USD", "LUNA1-USD", "SOL1-USD", "DOT1-USD", "DOGE-USD", "ADA-USD", "SHIB-USD", "LTC-USD", "LRC-USD", "CRO-USD"]



    for choice in choicelist:
        col1, col2 = st.columns(2)

        with col1:
            show_graph_in_column(time_zone, wdw, center_boll, z1, z2, choice, period_left, interval_left)

        with col2:
            show_graph_in_column(time_zone, wdw, center_boll, z1, z2,  choice, period_right, interval_right)
    show_info()

def show_info():
    st.write()

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        "Inspired by : <a href='https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a'>#</a> and"

        "<a href='https://medium.com/codex/algorithmic-trading-with-bollinger-bands-in-python-1b0a00c9ef99'>#</a> <br>"
        "Sourcecode : <a href='https://github.com/rcsmit/streamlit_scripts/crypto_dashboard.py' target='_blank'>github.com/rcsmit</a><br>"
        "How-to tutorial : <a href='https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d/' target='_blank'>rcsmit.medium.com</a><br>"
         "Info given 'as is'. <b>Read <a href='https://pypi.org/project/yfinance/'>disclaimer</a> at of Yfinace</b>"
       )


    st.sidebar.markdown(tekst, unsafe_allow_html=True)

def show_graph_in_column(time_zone, wdw, center_boll, z1, z2,  choice, period, interval):
    interval_datetime = ["1m","2m","5m","15m","30m","60m","90m","1h"]
    if interval in interval_datetime:
        x_as_label = "Datetime"
    else:
        x_as_label = "Date"
    df = get_data(choice, period, interval)

    if df is not None:
        df, std, mean,m,b = calculate_various_columns_df(df, wdw, center_boll, z1,z2)

        if x_as_label == "Datetime":
            df['Datetime'] = df['Datetime'].dt.tz_convert(time_zone)

        buy_price, sell_price, bb_signal, signal = implement_bb_strategy(df['Close'], df['boll_low_1'], df['boll_high_1'])
        if signal == 1:
            plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label)
        else:
            with st.expander(f"{choice} not interesting" ):
                plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label)
                    #st.info(f"{choice} not interesting" )


main()
