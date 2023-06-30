import yfinance as yf
import pandas as pd

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import datetime as dt
# from scipy import stats
# import datetime as dt



def get_data(choice,  interval):
    """Retreat the data from Yahoo Finance
    """

    data = yf.download(tickers=(choice), start="2021-11-27",interval=interval,group_by='ticker',auto_adjust=True,prepost=False)
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error(f"No data or wrong input - {choice}")
        df = None
    else:
        df['rownumber'] = np.arange(len(df))
    column_name = "close_" + choice
    df[column_name] = df["Close"]
    df = df.reset_index()
    print (df)
    try:
        df["Date"] = df["Datetime"]
    except:
        pass
    df = df[["Date", column_name]]

    return df

def calculate_assets(df, choice, transactions):
    """Calculate the worth of the portfolio
    Args:
        df ([type]): [description]
        choice ([type]): [description]
        transactions ([type]): [description]

    Returns:
        [type]: [description]
    """
    close_column = "close_" + choice
    quantity_column = "quantity_" + choice
    asset_column = "asset_" + choice

    for i in range(1, len(df)):
        for j in range(len(transactions)):
            #st.write(f".{df.loc[i, 'Date_y_m_d']}. == .{transactions[j][0]}. | .{choice}. == .{transactions[j][1]}.")
            if df.loc[i, "Date_y_m_d"] == transactions[j][0] and choice ==transactions[j][1]:
                # st.write ("HIT")
                # st.write(f".{df.loc[i, 'Date_y_m_d']}. == .{transactions[j][0]}. | .{choice}. == .{transactions[j][1]}.")

                df.loc[i, quantity_column] = df.loc[i-1, quantity_column] + transactions[j][2]
                break
            else:
                df.loc[i, quantity_column] = df.loc[i-1,quantity_column]
    df[asset_column] = df[quantity_column] * df[close_column]
    return df

def make_scatter(df, name, y, color):
    """Generate the line
    """
    df=df[1:]
    scat =  go.Scatter(
        name=name,
        x=df["Date"],
        y=df[y] ,
        mode='lines',
        line=dict(width=0.8,
                color=f"rgba({color}, 0.8)"))
    return scat
def plot(df, choicelist,logarithmic):
    """Makethe plot
    Args:
        df ([type]): [description]
        choicelist ([type]): [description]
    """

    # QUANTITY
    color = ["0,255,0", "255,0,0", "0,0,255", "255,255,0", "0,255,255", "255,0,255"]

    what = [[ "quantity_"+c, "quantity_"+c,color[i]] for  i,c  in enumerate(choicelist)]

    data = [  make_scatter(df, w[0], w[1], w[2] )   for w in what   ]

    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=f"Quantity")
        #, xaxis=dict(tickformat="%d-%m")
    fig1 = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig1, use_container_width=True)

    # ASSETS


    df["Total"] = 0
    what2=[]
    for i,c  in enumerate(choicelist):
        print (i)
        c_ = "asset_"+c
        what2.append([c_,c_,color[i]])

    for c in choicelist:
        a_ = "asset_"+c
        df["Total"] = df["Total"] + df[a_]
    what2.append(          ["Total", "Total", "255, 128, 255"  ])

    data2 = []
    for w in what2:
        w_ = make_scatter(df, w[0], w[1], w[2])
        data2.append(w_)

    layout = go.Layout(
        yaxis=dict(title="USD"),
        title=f"Assets")
        #, xaxis=dict(tickformat="%d-%m")
    fig2 = go.Figure(data=data2, layout=layout)
    st.plotly_chart(fig2, use_container_width=True)

    # KOERS BTC/ETH
    fig3 = go.Figure()

    # add the BTC trace
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['close_BTC-USD'], name='BTC', yaxis='y'))

    # add the ETH trace
    fig3.add_trace(go.Scatter(x=df['Date'], y=df['close_ETH-USD'], name='ETH', yaxis='y2'))

    # set the layout
    if logarithmic:
        yaxis_type = 'log'
    else:
        yaxis_type = 'linear'

    fig3.update_layout(
        title='BTC/ETH rates',
        yaxis=dict(title='BTC', titlefont=dict(color='blue'), type=yaxis_type),
        yaxis2=dict(title='ETH', titlefont=dict(color='red'), overlaying='y', side='right', type=yaxis_type)
    )

    # Display the plot
    
    # show the figure
    #fig3.show()
    st.plotly_chart(fig3, use_container_width=True)

def make_database(choicelist, interval):
    first = True
    for choice in choicelist:
        if first == True:
            df_total = get_data(choice,  interval)
            first = False
        else:
            df_temp = get_data(choice,  interval)

            df_total = pd.merge(
                df_total, df_temp, how="inner", on = "Date"
                        )
        c_name = "quantity_"+choice
        df_total[c_name] = 0.0
    #df_total["Date_y_m_d"] = str(df_total["Date"].dt.strftime("%Y-%m-%d"))
    df_total["Date_y_m_d"] =(df_total["Date"].dt.strftime("%Y-%m-%d"))
    print (df_total)
    print (df_total.dtypes)
    return df_total
def get_transactions():
    """Get the transactions. Later these will be loaded from a .csv or googlesheet
    """
    # assumed that buy/sell is at close-price. Negative is sell, positive is buy
    # TODO: Transaction costs. Real buying price
    #               date           ticker      quantity
    transactions = [['2021-11-28', "BTC-USD",  0.00128752] ,
                    ['2021-11-28', "ETH-USD", 0.01774314]]

                    # ,
                    # ['2021-12-15', "BTC-USD", -4.0],
                    # ['2021-12-17', "ETH-USD", -5.0]]
    return transactions

def input_options():
    # https://finance.yahoo.com/lookup?s=-usd&.tsrc=fin-srch
    choicelist_ = ["BTC-USD", "ETH-USD", "XRP-USD", "LUNA1-USD", "SOL1-USD", "DOT1-USD", "DOGE-USD", "ADA-USD", "SHIB-USD", "LTC-USD", "LRC-USD", "CRO-USD"]
    choicelist = st.sidebar.multiselect("Which coins", choicelist_, ["BTC-USD", "ETH-USD"])
    interval_top = "1d"# st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],8)
    logarithmic = st.sidebar.selectbox("Y axis logarithmic", [True,False], index = 1)
    
    return  interval_top,choicelist, logarithmic


def main():
    st.header("Crypto dashboard / watchlist of René Smit")
    interval_top,choicelist, logarithmic= input_options()
    df =  make_database(choicelist, interval_top)
    transactions = get_transactions()

    for choice in choicelist:
        df = calculate_assets(df, choice, transactions)
    st.write(df)
    plot(df, choicelist, logarithmic)

if __name__ == "__main__":
    main()

    # https://pythonawesome.com/building-a-proper-portfolio-tracker-with-python/