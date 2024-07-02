import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# https://quantpedia.com/strategies/currency-momentum-factor/

def get_data(choice, period, interval, window):
    """Retreat the data from Yahoo Finance
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
    df = df[["Date", column_name]]
  
    # Add a new column 'sma' with 3-period SMA
    df['sma'] = df[column_name].rolling(window=window, center=True).mean()

    return df

def show_plot(df, rate_column):
    """_summary_

    Args:
        df (_type_): _description_
        rate_column (_type_): _description_
    """    
    # Add a line trace for the close_EURTHB=X column
    #fig.add_trace(go.Scatter(x=df['Date'], y=df[rate_column], mode='lines+markers', name=rate_column))

    # Create traces for scatter plot and SMA line
    trace_scatter = go.Scatter(x=df['Date'], y=df[rate_column], mode='markers', name='Original Data', marker=dict(size=2))
    trace_sma = go.Scatter(x=df['Date'], y=df['sma'], mode='lines', name='SMA')

    # Create layout
    layout = go.Layout(title='Original Data with SMA',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Rate'))

    # Create figure
    fig = go.Figure(data=[trace_scatter, trace_sma], layout=layout)


    # Update layout
    fig.update_layout(title=f'Close Price of {rate_column}',
                    xaxis_title='Date',
                    yaxis_title=f'Close Price ({rate_column})',
                    hovermode='x unified')

    # Show the plot
    st.plotly_chart(fig)

def input_options():
    """_summary_

    Returns:
        _type_: _description_
    """    
    
    choice = st.sidebar.selectbox("Which ticker", ["EURTHB=X", "BTC-USD", "BTC-EUR"],0)
    period = st.sidebar.selectbox("Period", ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"], 5)
    interval = st.sidebar.selectbox("Interval", [ "1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"],8)
    window = st.sidebar.number_input("Window SMA", 1,None,15)
    return  choice, period, interval, window




def determine_rate_change(prev_rate, curr_rate):
    """ Function to determine rate change

    Args:
        prev_rate (_type_): _description_
        curr_rate (_type_): _description_

    Returns:
        _type_: _description_
    """    
    if curr_rate < prev_rate:
        return 'decreasing'
    elif curr_rate == prev_rate:
        return 'same'
    else:
        return 'increasing'


def create_transition_matrix(df, rate_column):
    """Function to create transition matrix

    Args:
        df (_type_): _description_
        rate_column (_type_): _description_

    Returns:
        _type_: _description_
    """    
    transitions = {
        'decreasing': {'decreasing': 0, 'same': 0, 'increasing': 0},
        'same': {'decreasing': 0, 'same': 0, 'increasing': 0},
        'increasing': {'decreasing': 0, 'same': 0, 'increasing': 0}
    }
    
    for i in range(len(df) - 2):
        prev_rate_day1 = df.loc[i, rate_column]
        prev_rate_day2 = df.loc[i + 1, rate_column]
        prev_rate_day3 = df.loc[i + 2, rate_column]
        
        state_day1_day2 = determine_rate_change(prev_rate_day1, prev_rate_day2)
        state_day2_day3 = determine_rate_change(prev_rate_day2, prev_rate_day3)
        
        transitions[state_day1_day2][state_day2_day3] += 1
    
    
    return pd.DataFrame(transitions)


def main():
    st.header("Momentum koersen")   


    choice, period, interval, window = input_options()
    st.info("""Analyzes and displays transition matrices to identify patterns in rate changes.
               This app can be used for momentum analysis of currencies and cryptocurrencies, 
               providing insights into trends and potential market movements.""")
    df = get_data(choice, period, interval, window)
    st.write (df)
    rate_column = f'close_{choice}'
    show_plot(df, rate_column)
    
    st.write("original values")
    transition_matrix = create_transition_matrix(df,  rate_column)

    st.write(transition_matrix)

    st.write("SMA")
    transition_matrix = create_transition_matrix(df, "sma")
    st.write(transition_matrix)
    
if __name__ == "__main__":
    main()

