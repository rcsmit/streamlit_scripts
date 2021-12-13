#https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a

import yfinance as yf
import pandas as pd

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

class QuantGaloreData:
    #Next, we will create a class that will store our data and perform the calculations.

    def __init__(self):
        choice= st.sidebar.text_input("Which ticker", "BTC-USD")
        period = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 5)
        interval =st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],8)
        self.ticker = yf.Tickers(choice)

        self.data = yf.download(tickers=(choice),period=period,interval=interval,group_by='ticker',auto_adjust=True,prepost=False)
        self.df = pd.DataFrame(self.data)
        self.df['rownumber'] = np.arange(len(self.df))
        self.choice = choice

    def print_dataframe(self):
        print(self.df)

    def plt_dataframe(self):
        def find_slope_scipy(x_,y_):

            m, b, r_value, p_value, std_err = stats.linregress(x_, y_)
            r_sq = r_value**2
            return m,b,r_sq

        def manipulate_df(df):
            df = df.reset_index()
            x = list(range(0,len(df)))
            std = np.std(self.df['Close'])
            mean = df['Close'].mean()
            y = df["Close"].to_list()
            m,b,r_sq = find_slope_scipy(x,y)
            df['trendline'] = (df['rownumber'] *m +b)
            z = st.sidebar.number_input("Z-value", 0.0,3.0,1.96)
            df['trendline_low'] = (df['rownumber'] *m +b) - z * std
            df['trendline_high'] = (df['rownumber'] *m +b) + z * std
            df['z_from_mean'] = (df['Close'] - mean) / std
            return df

        def plot_figure1(df, choice):

            low = go.Scatter(
                name='Low',
                x=df["Date"],
                y=df['Low'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(0,0,255, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty')

            close = go.Scatter(
                name="Close",
                x=df["Date"],
                y=df["Close"],
                mode='lines',
                line=dict(width=0.75,color='rgba(0,0,255, 0.8)'),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty')

            high = go.Scatter(
                name='High',
                x=df["Date"],
                y=df['High'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(0,0,255, 0.5)"),
                fillcolor='rgba(68, 68, 68, 0.1)',
                )


            trendline_low = go.Scatter(
                name='trendline low',
                x=df["Date"],
                y=df['trendline_low'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,0 0.05)',
                fill='tonexty')

            trendline = go.Scatter(
                name="trendline",
                x=df["Date"],
                y=df["trendline"],
                mode='lines',
                line=dict(width=0.9,color='rgba(255,255,0,1)'),
                fillcolor='rgba(255,255,0 0.05)',
                fill='tonexty'
                )

            trendline_high = go.Scatter(
                name='trendline high',
                x=df["Date"],
                y=df['trendline_high'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,255 0.05)',
                )



            data = [high, close, low,trendline_high, trendline, trendline_low ]

            layout = go.Layout(
                yaxis=dict(title="USD"),
                title=f"{choice}",)
                #, xaxis=dict(tickformat="%d-%m")
            fig1 = go.Figure(data=data, layout=layout)
            fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            #fig.show()
            st.plotly_chart(fig1, use_container_width=True)


        def plot_figure_z_scores(df,choice):
            z_score = go.Scatter(
                name="Close",
                x=df["Date"],
                y=df["z_from_mean"],
                mode='lines',
                line=dict(width=0.75,color='rgba(0,0,255, 0.8)'),
                fillcolor='rgba(68, 68, 68, 0.2)',
                fill='tonexty')

            data = [z_score ]

            layout = go.Layout(
                yaxis=dict(title=f"Z-score {choice}"),
                title="Z from mean",)
            fig1 = go.Figure(data=data, layout=layout)
            fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            st.plotly_chart(fig1, use_container_width=True)


        st.header("Y Finance charts")
        df = manipulate_df(self.df)
        choice = self.choice
        plot_figure1(df, choice)
        plot_figure_z_scores(df, choice)

        print (df)


    def find_z(self):
        # This will call the yahoo finance API and store Bitcoin’s OHLC data on the time interval we set.
        # You can enter in any ticker from all asset classes (futures, stocks, crypto), but for this example,
        #  we’ll use Bitcoin. Next, we will create a function to calculate the z-score values.

        mean = self.df['Close'].mean()
        z_from_mean = (self.df['Close'].tail(1) - mean) / np.std(self.df['Close'])
        print("BTC-USD",self.df['Close'].tail(1),z_from_mean)
        # This will calculate the mean value and then it will pull the last minute’s trading price
        # and divide that by the standard deviation of the dataset. Finally, this will return a calculated z-score.
def main():

    QGD = QuantGaloreData()
    #QGD.print_dataframe()
    QGD.plt_dataframe()

    st.write("Inspired by : https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a")
    st.write ("Read disclaimer at https://pypi.org/project/yfinance/")
main()
