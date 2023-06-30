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
    return df, m,b,std

def sma(data, window, center_boll):
        sma = data.rolling(window = window, center=center_boll).mean()
        return sma

def do_bollinger(df, z1, z2, wdw, center_boll):
    #    # https://medium.com/codex/algorithmic-trading-with-bollinger-bands-in-python-1b0a00c9ef99

    def bb(data, sma, window, center_boll):
        std = data.rolling(window = window, center=center_boll).std()
        upper_bb_1 = sma + std * z1
        lower_bb_1 = sma - std * z1

        upper_bb_2 = sma + std * z2
        lower_bb_2 = sma - std * z2

        return lower_bb_1, lower_bb_2, upper_bb_1, upper_bb_2

    df['boll_center'] = sma(df['Close'], wdw, center_boll)
    df['boll_low_1'], df['boll_low_2'], df['boll_high_1'], df['boll_high_2'] = bb(df['Close'], df['boll_center'], wdw, center_boll)
    return df

def implement_macd_strategie_1(close, ma_short, ma_long):

    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(1,len(close)):
        if ma_short[i-1] > ma_long[i-1] and ma_short[i] < ma_long[i]:

            # sell
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(close[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        elif  ma_short[i-1] < ma_long[i-1] and ma_short[i]> ma_long[i]:
            # buy
            if signal != 1:
                buy_price.append(close[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
    return buy_price, sell_price, macd_signal, signal


def implement_macd_strategie_2(close, macd_line, signal_line):

    buy_price = []
    sell_price = []
    macd_signal = []
    signal = 0

    for i in range(1,len(close)):
        if macd_line[i-1] > signal_line[i-1] and macd_line[i] < signal_line[i]:

            # sell
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(signal_line[i])
                signal = -1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        elif  macd_line[i-1] < signal_line[i-1] and macd_line[i]> signal_line[i]:
            # buy
            if signal != 1:
                buy_price.append(signal_line[i])
                sell_price.append(np.nan)
                signal = 1
                macd_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                macd_signal.append(0)

        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            macd_signal.append(0)
    return buy_price, sell_price, macd_signal, signal

def implement_bb_strategy(close, bol_low_1, bol_high_1):
    #https://medium.com/codex/how-to-calculate-bollinger-bands-of-a-stock-with-python-f9f7d1184fc3
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0

    for i in range(1,len(close)):
        if  close[i-1] > bol_low_1[i-1] and  close[i] < bol_low_1[i]:
            # buy
            if signal != 1:

                buy_price.append(close[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif  close[i-1] < bol_high_1[i-1] and  close[i] > bol_high_1[i]:
            # sell
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(close[i])
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


def draw_buy_sell_candlestick_close(df, buy_price, sell_price, x_as_label):

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

    candlestick = go.Candlestick(x=df[x_as_label],
                        name = "candlestick",
                                   open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'], opacity=0.2)

    close = go.Scatter(
        name="Close",
        x=df[x_as_label],
        y=df["Close"],
        mode='lines',
        line=dict(width=1,color='rgba(0,0,0, 0.3)'),
        fillcolor='rgba(68, 68, 68, 0.2)',
        )

    return buy,sell,candlestick,close

def plot_trendline(df, choice,m,b, std, x_as_label,  logarithmic):
    if logarithmic:
        close = go.Scatter(
            name="Close",
            x=df[x_as_label],
            y=df["Close"],
            mode='lines',
            line=dict(width=0.75,color='rgba(0,0,255, 1)'))
            
        data = [close]
        title = f"{choice} (y axis logarithmic)"
    
    else:
        low = go.Scatter(
            name='Low',
            x=df[x_as_label],
            y=df['Low'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(0,0,255, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty')

        close = go.Scatter(
            name="Close",
            x=df[x_as_label],
            y=df["Close"],
            mode='lines',
            line=dict(width=0.75,color='rgba(0,0,255, 0.8)'),
            fillcolor='rgba(68, 68, 68, 0.2)',
            fill='tonexty')
    
        high = go.Scatter(
            name='High',
            x=df[x_as_label],
            y=df['High'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(0,0,255, 0.5)"),
            fillcolor='rgba(68, 68, 68, 0.1)',
            )

        trendline_low_2 = go.Scatter(
            name='trendline low 2',
            x=df[x_as_label],
            y=df['trendline_low_2'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 255, 0, 0.8)"),
            fillcolor='rgba(255,255,0,0.2)',
            fill='tonexty')

        trendline_low_1 = go.Scatter(
            name='trendline low 2',
            x=df[x_as_label],
            y=df['trendline_low_1'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 255, 0, 0.0)"),
            fillcolor='rgba(255,255,0, 0.4)',
            fill='tonexty')

        trendline = go.Scatter(
            name="trendline",
            x=df[x_as_label],
            y=df["trendline"],
            mode='lines',
            line=dict(width=0.9,color='rgba(255,165,0,1)'),
            fillcolor='rgba(255,255,0,0.4)',
            fill='tonexty'
            )

        trendline_high_1 = go.Scatter(
            name='trendline high 1',
            x=df[x_as_label],
            y=df['trendline_high_1'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 255, 0, 0.0)"),
            fillcolor='rgba(255,255,0, 0.2)',
                fill='tonexty'
            )
        trendline_high_2 = go.Scatter(
            name='trendline high 2',
            x=df[x_as_label],
            y=df['trendline_high_2'] ,
            mode='lines',
            line=dict(width=0.5,
                    color="rgba(255, 255, 0, 0.8)"),
            fillcolor='rgba(255,255,0, 0.0)',
                fill='tonexty'

            )

        title = f"{choice} | trendline = {round(m,1)} * x + {round(b,1)} | std = {round(std,2)}"
        data = [high, close, low,trendline_high_2,trendline_high_1, trendline, trendline_low_1,trendline_low_2 ]
        
    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=title,)
        #, xaxis=dict(tickformat="%d-%m")
    
    fig1 = go.Figure(data=data, layout=layout)
    if logarithmic:
        fig1.update_layout(        yaxis_type='log'    )
    fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

    #fig.show()
    st.plotly_chart(fig1, use_container_width=True)
def plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label):

    buy, sell, candlestick, close = draw_buy_sell_candlestick_close(df, buy_price, sell_price, x_as_label)
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



    data = [boll_high_2,boll_high_1, boll, boll_low_1,boll_low_2,close, buy, sell, candlestick ]

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
def plot_macd(df, choice,  buy_price, sell_price, macd_signal, x_as_label):

    buy, sell, candlestick, close = draw_buy_sell_candlestick_close(df, buy_price, sell_price, x_as_label)

    ma_long = go.Scatter(
        name='ema_long',
        x=df[x_as_label],
        y=df['ma_long'] ,
        mode='lines',
        line=dict(width=0.75,
                color="rgba(255,0 ,0, 1)"),
        )

    ma_short = go.Scatter(
        name='ema_short',
        x=df[x_as_label],
        y=df['ma_short'] ,
        mode='lines',
        line=dict(width=0.75,
                color="rgba(0, 0, 255, 1)"),
       )


    data = [ma_long, ma_short ,close, buy, sell, candlestick ]

    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=f"Moving averages - {choice}")
        #, xaxis=dict(tickformat="%d-%m")
    fig1 = go.Figure(data=data, layout=layout)
    min=df["Close"].min() * 0.9975
    max=df["Close"].max() *1.0025

    #fig1.update_yaxes(autorange=True)
    fig1.update_yaxes(range=[min,max])
    #fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

    #fig.show()
    st.plotly_chart(fig1, use_container_width=True)



def plot_macd_2(df, choice,  buy_price, sell_price, macd_signal, x_as_label):

    buy, sell, candlestick, close = draw_buy_sell_candlestick_close(df, buy_price, sell_price, x_as_label)

    from plotly.subplots import make_subplots

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        name='BUY',
        x=df[x_as_label],
        y=buy_price ,
        mode="markers",marker_symbol='triangle-up',opacity=0.4,
                    marker_line_color="midnightblue", marker_color="green",
                    marker_line_width=0, marker_size=11,
                    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name='SELL',
        x=df[x_as_label],
        y=sell_price ,
        mode="markers",marker_symbol='triangle-down',opacity=0.4,
                    marker_line_color="midnightblue", marker_color="red",
                    marker_line_width=0, marker_size=11,
                    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name='MACD_Line',
        x=df[x_as_label],
        y=df['MACD_Line'] ,
        mode='lines',
        line=dict(width=0.75,
                color="rgba(255,0 ,0, 1)"),

        ), secondary_y=False)

    fig.add_trace(go.Scatter(
        name='9 day signal_Line',
        x=df[x_as_label],
        y=df['Signal_Line'] ,
        mode='lines',
        line=dict(width=0.75,
                color="rgba(0, 0, 255, 1)"),

       ), secondary_y= False)
    fig.add_trace( go.Bar(
        name='MACD_Histogram',
        x=df[x_as_label],
        y=df['MACD_Histogram'] ,
        marker=dict(
            color="rgba(86, 23, 255, 0.5)",

            line=dict(
                color='MediumPurple',
                width=0
            ))),

          secondary_y=True)

    fig.update_layout(
    title_text=f"MACD - {choice}")
    # Set y-axes titles
    fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
    fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)


    # layout = go.Layout(
    #     yaxis=dict(title="USD"),
    #     title=)
        #, xaxis=dict(tickformat="%d-%m")
    # fig1 = go.Figure(data=data, layout=layout)
    # fig = make_subplots(specs=[[{"secondary_y": True}]])
    # min=df["Close"].min() * 0.9975
    # max=df["Close"].max() *1.0025

    # #fig1.update_yaxes(autorange=True)
    # fig1.update_yaxes(range=[min,max])
    # #fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

    #fig.show()
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(layout="wide")
    st.header("Crypto dashboard / watchlist of René Smit")



    period_top, interval_top, period_left, interval_left, period_right, interval_right, which_to_show, time_zone, wdw, center_boll, z1, z2, ma1, ma2, choicelist,logarithmic = input_options()
    # choicelist = ["BTC-USD"]
    for choice in choicelist:
        show_graph_in_column(time_zone, wdw, center_boll, z1, z2, choice, period_top, interval_top,"top", ma1, ma2, logarithmic)

        col1, col2 = st.columns(2)
        with col1:
            show_graph_in_column(time_zone, wdw, center_boll, z1, z2, choice, period_left, interval_left, which_to_show, ma1, ma2, logarithmic)

        with col2:
            show_graph_in_column(time_zone, wdw, center_boll, z1, z2,  choice, period_right, interval_right, which_to_show, ma1, ma2, logarithmic)
        st.markdown("<hr", unsafe_allow_html=True)
    show_info()

def input_options():
    # https://finance.yahoo.com/lookup?s=-usd&.tsrc=fin-srch
    choicelist_ = ["BTC-USD", "ETH-USD", "XRP-USD", "LUNA1-USD", "SOL1-USD", "DOT1-USD", "DOGE-USD", "ADA-USD", "SHIB-USD", "LTC-USD", "LRC-USD", "CRO-USD"]
    choicelist = st.sidebar.multiselect("Which coins", choicelist_, ["BTC-USD", "ETH-USD"])
    #choicelist = ["BTC-USD"]

    st.sidebar.write("TOP")
    period_top = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 2)
    interval_top =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],8)
    st.sidebar.write("LEFT")
    period_left = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 1)
    interval_left =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],5)

    st.sidebar.write ("RIGHT")
    period_right = st.sidebar.selectbox("Period ", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 0)
    interval_right =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],0)
    which_to_show = st.sidebar.selectbox("Which to show", ["Bollinger", "MACD", "Both"],2)
    logarithmic = st.sidebar.selectbox("Y axis logarithmic", [True,False], index = 1)
    
    time_zone = st.sidebar.selectbox("Tijdzone", ["CET", "CEST", "ICT"],1)
    wdw = int( st.sidebar.number_input("Window Moving Average",2,60,20))
    center_boll = st.sidebar.selectbox("Center Moving Average", [True, False], index=1)
    z1 = st.sidebar.number_input("Z-value 1", 0.0,3.0,1.0)
    z2 = st.sidebar.number_input("Z-value 2", 0.0,3.0,1.96)
    ma1=st.sidebar.number_input("MA1 (short)", 1,100,12 )
    ma2=st.sidebar.number_input("MA1 (short)", 1,100,26)

    return period_top, interval_top, period_left,interval_left,period_right,interval_right,which_to_show,time_zone,wdw,center_boll,z1,z2,ma1,ma2,choicelist, logarithmic

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
def ema(prices, days, correctie ,smoothing=2):
    """Calculate exponential moving average
    https://python.plainenglish.io/how-to-calculate-the-ema-of-a-stock-with-python-5d6fb3a29f5

    Args:
        prices ([type]): [description]
        days (int): [description]
        correctie (int): Needed because the first values will be NaN
        smoothing (int, optional):[description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    #
    #ema = [sum(prices[:days]) / days]
    ema = []

    for price in prices[:days-1+correctie]:
        ema.append(None)
    ema.append(sum(prices[correctie:+correctie+days]) / days)
    for price in prices[days+correctie:]:
        if price != None and ema[-1]!=None:
            ema.append((price * (smoothing / (1 + days))) + ema[-1] * (1 - (smoothing / (1 + days))))
        else:
            ema.append(None)

    return ema

def show_plot_bollinger(choice, x_as_label, df,z1, z2, wdw, center_boll):
    df = do_bollinger(df, z1, z2, wdw, center_boll)
    buy_price, sell_price, bb_signal, signal_bb = implement_bb_strategy(df['Close'], df['boll_low_1'], df['boll_high_1'])

    if signal_bb == 1:
        plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label)
    else:
        with st.expander(f"Bollinger - {choice} not interesting" ):
            plot_boll(df, choice,  buy_price, sell_price, bb_signal, x_as_label)
                        #st.info(f"{choice} not interesting" )

def show_plot_macd(df,choice,x_as_label, ma1, ma2):
    df["ma_short"]= ema(df["Close"], ma1,0)
    df["ma_long"]= ema(df["Close"], ma2,0)
    df["MACD_Line"] = df["ma_short"] - df["ma_long"]
    df["Signal_Line"] =  ema(df["MACD_Line"], 9,26)
    df["MACD_Histogram"] =df["MACD_Line"] - df["Signal_Line"]


    buy_price, sell_price, macd_signal, signal_macd = implement_macd_strategie_1(df['Close'], df['ma_short'], df['ma_long'])

    if signal_macd == 1:
        plot_macd(df, choice,  buy_price, sell_price, macd_signal, x_as_label)

    else:
        with st.expander(f"Moving averages - {choice} not interesting" ):
            plot_macd(df, choice,  buy_price, sell_price, macd_signal, x_as_label)

    buy_price, sell_price, macd_signal, signal_macd = implement_macd_strategie_2(df['Close'], df['MACD_Line'], df['Signal_Line'])

    if signal_macd == 1:
        plot_macd_2(df, choice,  buy_price, sell_price, macd_signal, x_as_label)

    else:
        with st.expander(f"MACD 2 - {choice} not interesting" ):
            plot_macd_2(df, choice,  buy_price, sell_price, macd_signal, x_as_label)
                        #st.info(f"{choice} not interesting" )


def show_graph_in_column(time_zone, wdw, center_boll, z1, z2,  choice, period, interval, which_to_show, ma1, ma2, logarithmic):
    ma1,ma2 = int(ma1), int(ma2)
    interval_datetime = ["1m","2m","5m","15m","30m","60m","90m","1h"]
    if interval in interval_datetime:
        x_as_label = "Datetime"
    else:
        x_as_label = "Date"
    df = get_data(choice, period, interval)

    if df is not None:
        df, m,b,std= calculate_various_columns_df(df, wdw, center_boll, z1,z2)

        # if x_as_label == "Datetime":
        #     #df['Datetime'] = df['Datetime'].dt.tz_convert(time_zone)
        #     df['Datetime'] = pd.to_datetime(df['Datetime']).dt.tz_localize('time_zone')
        #     df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize('UTC')
        if which_to_show=="top":
            plot_trendline(df, choice,m,b,std, x_as_label, logarithmic)

        if which_to_show =="Bollinger" or which_to_show =="Both":

            show_plot_bollinger(choice, x_as_label, df, z1, z2, wdw, center_boll)

        if which_to_show == "MACD" or which_to_show =="Both":
            show_plot_macd(df, choice, x_as_label, ma1, ma2)

main()

# https://towardsdatascience.com/detection-of-price-support-and-resistance-levels-in-python-baedc44c34c9

# De berekening van de MACD verloopt als volgt:

# MA1 = Exponentiële Moving Average op basis van slotkoersen, over periode X, is standaard 12 dagen
# MA2 = Exponentiële Moving Average op basis van slotkoersen, over periode X, is standaard 26 dagen
# Bereken het verschil van deze Moving Averages:

# MACD1 = MA1 - MA2
# Neem nu opnieuw een Moving Average van deze waarde:

# MACD2 = Exponentiële Moving Average berekend uit (MACD1) op basis van de laatste X-waardes, standaard 9.