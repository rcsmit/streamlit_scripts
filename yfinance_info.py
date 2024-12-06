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
        if len(self.df) == 0:
            st.error("No data or wrong input")
            st.stop()
        self.df['rownumber'] = np.arange(len(self.df))
        self.choice = choice

    def print_dataframe(self):
        print(self.df)

    def plt_dataframe(self):
        def find_slope_scipy(x_,y_):
            m, b, r_value, p_value, std_err = stats.linregress(x_, y_)
            r_sq = r_value**2
            return m,b,r_sq

        def calculate_various_columns_df(df):
            df.columns = [''.join(col).strip() for col in df.columns.values]
       
            df = df.reset_index()
            st.write(df)
            std = np.std(self.df['Close'])
            mean = df['Close'].mean()
            x = list(range(0,len(df)))
            y = df["Close"].to_list()
            m,b,r_sq = find_slope_scipy(x,y)
            z1 = st.sidebar.number_input("Z-value 1", 0.0,3.0,1.0)
            z2 = st.sidebar.number_input("Z-value 2", 0.0,3.0,1.96)
            wdw = int( st.sidebar.number_input("Window for bollinger",2,60,20))
            center_boll = st.sidebar.selectbox("Center bollinger", [True, False], index=0)

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
            return df, std, mean, m, b,z1, z2

        def do_bollinger_oud(df, z1,z2):
            # https://medium.com/codex/algorithmic-trading-with-bollinger-bands-in-python-1b0a00c9ef99
            def create_coll_boll(df, df_temp):
                std = np.std(df_temp['Close'])
                df.loc[i, "boll_center"] =  df_temp['Close'].mean()
                df.loc[i, "boll_high_1"] = df.loc[i, "boll_center"] + z1 * std
                df.loc[i, "boll_low_1"] = df.loc[i, "boll_center"] - z1* std
                df.loc[i, "boll_high_2"] = df.loc[i, "boll_center"] + z2 * std
                df.loc[i, "boll_low_2"] = df.loc[i, "boll_center"] - z2* std
                return df

            # df["boll_low"],df["boll_center"], df["boll_high"] = None, None, None


            if (wdw % 2) != 0:
                st.error("Please enter an even number for window")
                st.stop()
            if center_boll:
                for i in range(int(wdw/2),int((len(df)-wdw/2))):
                    df_temp = df.iloc[i-int(wdw/2):i+int(wdw/2), :]
                    df = create_coll_boll(df, df_temp)

            else:
                for i in range(20,len(df)):
                    df_temp = df.iloc[i-20:i, :]
                    df = create_coll_boll(df, df_temp)


            return df

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

            return buy_price, sell_price, bb_signal




        def plot_boll(df, choice,  buy_price, sell_price, bb_signal):

            buy = go.Scatter(
                name='BUY',
                x=df["Date"],
                y=buy_price ,
                  mode="markers",  marker_symbol='triangle-up', opacity=0.4,
                           marker_line_color="midnightblue", marker_color="green",
                           marker_line_width=0, marker_size=11,
                         )



            sell = go.Scatter(
                name='SELL',
                x=df["Date"],
                y=sell_price ,
                mode="markers",marker_symbol='triangle-down',opacity=0.4,
                           marker_line_color="midnightblue", marker_color="red",
                           marker_line_width=0, marker_size=11,
                         )




            boll_low_2 = go.Scatter(
                name='boll low 2',
                x=df["Date"],
                y=df['boll_low_2'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,0,0.2)',
                fill='tonexty')

            boll_low_1 = go.Scatter(
                name='boll low 1',
                x=df["Date"],
                y=df['boll_low_1'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.0)"),
                fillcolor='rgba(255,255,0, 0.4)',
                fill='tonexty')

            boll = go.Scatter(
                name="boll",
                x=df["Date"],
                y=df["boll_center"],
                mode='lines',
                line=dict(width=0.9,color='rgba(255,165,0,1)'),
                fillcolor='rgba(255,255,0,0.4)',
                fill='tonexty'
                )

            boll_high_1 = go.Scatter(
                name='boll high 1',
                x=df["Date"],
                y=df['boll_high_1'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.0)"),
                fillcolor='rgba(255,255,0, 0.2)',
                  fill='tonexty'
                )
            boll_high_2 = go.Scatter(
                name='boll high 2',
                x=df["Date"],
                y=df['boll_high_2'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,0, 0.0)',
                   fill='tonexty'

                )



            close = go.Scatter(
                name="Close",
                x=df["Date"],
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


            fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

            #fig.show()
            st.plotly_chart(fig1, use_container_width=True)

        def plot_figure1(df, choice,m,b, std):

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

            trendline_low_2 = go.Scatter(
                name='trendline low 2',
                x=df["Date"],
                y=df['trendline_low_2'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,0,0.2)',
                fill='tonexty')

            trendline_low_1 = go.Scatter(
                name='trendline low 2',
                x=df["Date"],
                y=df['trendline_low_1'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.0)"),
                fillcolor='rgba(255,255,0, 0.4)',
                fill='tonexty')

            trendline = go.Scatter(
                name="trendline",
                x=df["Date"],
                y=df["trendline"],
                mode='lines',
                line=dict(width=0.9,color='rgba(255,165,0,1)'),
                fillcolor='rgba(255,255,0,0.4)',
                fill='tonexty'
                )

            trendline_high_1 = go.Scatter(
                name='trendline high 1',
                x=df["Date"],
                y=df['trendline_high_1'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.0)"),
                fillcolor='rgba(255,255,0, 0.2)',
                  fill='tonexty'
                )
            trendline_high_2 = go.Scatter(
                name='trendline high 2',
                x=df["Date"],
                y=df['trendline_high_2'] ,
                mode='lines',
                line=dict(width=0.5,
                        color="rgba(255, 255, 0, 0.8)"),
                fillcolor='rgba(255,255,0, 0.0)',
                   fill='tonexty'

                )



            data = [high, close, low,trendline_high_2,trendline_high_1, trendline, trendline_low_1,trendline_low_2 ]

            layout = go.Layout(
                yaxis=dict(title="USD"),
                title=f"{choice} | trendline = {round(m,1)} * x + {round(b,1)} | std = {round(std,2)}",)
                #, xaxis=dict(tickformat="%d-%m")
            fig1 = go.Figure(data=data, layout=layout)


            fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))

            #fig.show()
            st.plotly_chart(fig1, use_container_width=True)

        def plot_figure_z_scores(df,choice,  std, mean, from_what):
            z_score = go.Scatter(
                name="Close",
                x=df["Date"],
                y=df[from_what],
                mode='lines',
                line=dict(width=0.75,color='rgba(0,0,255, 0.8)'),
                #fillcolor='rgba(68, 68, 68, 0.2)',
                #fill='tonexty'
                )

            data = [z_score ]
            if from_what =="z_from_mean":
                title = f"Z from mean | mean = {round(mean,2)} | std = {round(std,2)}"
            else:
                title = f"Z from Trendline  | std = {round(std,2)}"
            layout = go.Layout(
                yaxis=dict(title=f"Z-score {choice}"),
                title=title)
            fig1 = go.Figure(data=data, layout=layout)
            fig1.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
            st.plotly_chart(fig1, use_container_width=True)

        def buy_or_sell(df, std, z1,z2):
            # z_from_trendline = (df['trendline'].tail(1) - df['Close'].tail(1)) / std
            z_from_trendline =  round(df.z_from_trendline.iat[-1],2) #  (  df.Close.iat[-1] - df.trendline.iat[-1] ) / std
            st.write (f"Z-score value from trendline = {z_from_trendline}")

            if z_from_trendline <= -1*z2:
                st.write ("REALLY BUY !")
            elif (z_from_trendline > -1*z2) and (z_from_trendline <= -1*z1):
                st.write (" BUY !")
            elif (z_from_trendline > -1*z1) and (z_from_trendline <= 1*z1):
                st.write ("HOLD !")
            elif (z_from_trendline > 1*z1) and (z_from_trendline <= 2*z2):
                st.write ("Maybe SEL !")
            elif z_from_trendline > z2:
                st.write ("REALLY BUY !")
            st.write("This advice is just a joke ofcourse. Just use your own knowledge and insights.")

        def find_z_mean(self):
            # This will call the yahoo finance API and store Bitcoin’s OHLC data on the time interval we set.
            # You can enter in any ticker from all asset classes (futures, stocks, crypto), but for this example,
            #  we’ll use Bitcoin. Next, we will create a function to calculate the z-score values.

            mean = self.df['Close'].mean()
            z_from_mean = (self.df['Close'].tail(1) - mean) / np.std(self.df['Close'])
            print("BTC-USD",self.df['Close'].tail(1),z_from_mean)
            # This will calculate the mean value and then it will pull the last minute’s trading price
            # and divide that by the standard deviation of the dataset. Finally, this will return a calculated z-score.


        st.header("Y Finance charts")
        df, std, mean,m,b, z1,z2 = calculate_various_columns_df(self.df)
        buy_price, sell_price, bb_signal = implement_bb_strategy(df['Close'], df['boll_low_1'], df['boll_high_1'])

        choice = self.choice
        plot_figure1(df, choice,m,b, std)
        buy_or_sell(df, std, z1,z2)
        plot_boll(df, choice,  buy_price, sell_price, bb_signal)
        plot_figure_z_scores(df, choice,  std, mean, "z_from_trendline")
        plot_figure_z_scores(df, choice,  std, mean, "z_from_mean")

        print (df)




def main():

    QGD = QuantGaloreData()
    QGD.print_dataframe()
    QGD.plt_dataframe()

    st.write()

    tekst = (
        "<style> .infobox {  background-color: lightblue; padding: 5px;}</style>"
        "<hr><div class='infobox'>Made by Rene Smit. (<a href='http://www.twitter.com/rcsmit' target=\"_blank\">@rcsmit</a>) <br>"
        "Inspired by : <a href='https://medium.com/the-financial-journal/the-million-dollar-algorithm-straight-from-wall-street-3f88a62e3e0a'>I Needed Money, So I Wrote An Algorithm</a> <br>"
        "Sourcecode : <a href='https://github.com/rcsmit/COVIDcases/blob/main/covid_dashboard_rcsmit.py' target='_blank'>github.com/rcsmit</a><br>"
        "How-to tutorial : <a href='https://rcsmit.medium.com/making-interactive-webbased-graphs-with-python-and-streamlit-a9fecf58dd4d/' target='_blank'>rcsmit.medium.com</a><br>"
         "Read <a href=<'https://pypi.org/project/yfinance/'>disclaimer</a> at of Yfinace"
       )


    st.sidebar.markdown(tekst, unsafe_allow_html=True)
if __name__ == "__main__":
    main()


