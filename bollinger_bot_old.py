import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# this script calculates the bollinger bands day by day, crreating a more realistic simulation
# than by hindsight

# Fetch historical data
def fetch_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    if len(data)==0 :
        st.error("No data. Is the symbol valid?")
        st.stop()
    return data

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window, no_of_std):
    rolling_mean = data['Close'].rolling(window).mean()
    rolling_std = data['Close'].rolling(window).std()
    
    data['Bollinger High'] = rolling_mean + (rolling_std * no_of_std)
    data['Bollinger Low'] = rolling_mean - (rolling_std * no_of_std)
    return data

def simulate_trading(data, cash, position, transaction_cost, sell_only_at_profit):
    buying_price = 0  # Price at which the stock was bought
    data['Signal'] = 0  # Column to store trading signals
    data['Portfolio Value'] = cash  # Column to store portfolio value

    # Initial buy for hold strategy
    initial_cash = 10000
    initial_position = initial_cash / (data['Close'][0] * (1 + transaction_cost))
    hold_value = initial_position * data['Close'][-1] * (1 - transaction_cost)

    for i in range(1, len(data)):
        if data['Close'][i] < data['Bollinger Low'][i] and position == 0:
            # Buy signal with transaction cost
            shares_to_buy = cash / (data['Close'][i] * (1 + transaction_cost))
            position = shares_to_buy
            buying_price = data['Close'][i]
            cash -= shares_to_buy * data['Close'][i] * (1 + transaction_cost)
            data.at[data.index[i], 'Signal'] = 1  # Buy
            st.write(f"Buying at {data['Close'][i]} with {shares_to_buy} shares")
        elif data['Close'][i] > data['Bollinger High'][i] and position > 0:
            if not sell_only_at_profit or data['Close'][i] > buying_price:
                # Sell signal with transaction cost
                cash += position * data['Close'][i] * (1 - transaction_cost)
                position = 0
                data.at[data.index[i], 'Signal'] = -1  # Sell
                st.write(f"Selling at {data['Close'][i]}")
        
        # Update portfolio value
        data.at[data.index[i], 'Portfolio Value'] = cash if position == 0 else cash + position * data['Close'][i]

    final_value = cash if position == 0 else cash + position * data['Close'][-1]
    st.write(f"Final portfolio value: {final_value}")
    st.write(f"Buy and hold strategy final value: {hold_value}")
    return data, final_value, hold_value



def plot_portfolio_value(data):
    fig = go.Figure()

    # Add portfolio value line
    fig.add_trace(go.Scatter(x=data.index, y=data['Portfolio Value'], mode='lines', name='Portfolio Value'))

    # Add buy signals
    buy_signals = data[data['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Portfolio Value'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))

    # Add sell signals
    sell_signals = data[data['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Portfolio Value'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

    # Customize layout
    fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Portfolio Value', template='plotly_dark')

    

    st.plotly_chart(fig)
def plot_data(data):
    fig = go.Figure()

    # Add closing price line
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger High'], mode='lines', name='Bollinger High', line=dict(dash='dash', width=0.7)))
    fig.add_trace(go.Scatter(x=data.index, y=data['Bollinger Low'], mode='lines', name='Bollinger Low', line=dict(dash='dash',width=0.7)))

    # Add buy signals
    buy_signals = data[data['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))

    # Add sell signals
    sell_signals = data[data['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

    # Customize layout
    fig.update_layout(title='Bollinger Bands Trading Strategy', xaxis_title='Date', yaxis_title='Price', template='plotly_dark')

    st.plotly_chart(fig)


def calculate_years_between_dates(start_date, end_date):
    try:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
    except:
        st.error("Make sure the start and enddate are in yyyy-mm-dd")
        st.stop()

    years = end.year - start.year

    delta = end - start
    years = delta.days / 365.25  # Using 365.25 to account for leap years
    return round(years, 1)
def calculate_cagr(cash, final_value, years):
    cagr = (final_value / cash) ** (1 / years) - 1
    return cagr

def main():
    symbol = st.sidebar.selectbox("symbol", (['AAPL', 'BTC-USD', 'ETH-USD']))
    start_date = st.sidebar.text_input("start date", '2015-01-01')
    end_date = st.sidebar.text_input("end date", '2022-07-01')
    sell_only_at_profit = st.sidebar.selectbox("sell_only_at_profit", [True, False], True)
    window = 20
    no_of_std = 2

    cash = 10000  # Initial cash
    position = 0  # Initial position

    transaction_cost = st.sidebar.number_input("Transaction costs (%)", 0.0,1.0,0.5)/100
    years = calculate_years_between_dates(start_date, end_date)
    
    data = fetch_data(symbol, start=start_date, end=end_date)
    data = calculate_bollinger_bands(data, window, no_of_std)
    with st.expander("Transactions"):
        data_sim, final_value, hold_value = simulate_trading(data, cash, position, transaction_cost,sell_only_at_profit)
    plot_data(data)
    plot_portfolio_value(data_sim)
    
    cagr = round(calculate_cagr(cash, final_value, years)*100,2)
    cagr_hold = round(calculate_cagr(cash, hold_value, years)*100,2)

    st.write(f"ROI bolinger strategy: {cagr} % over {years} years")
    st.write(f"ROI buy and hold: {cagr_hold} % over {years} years")

    

if __name__ == "__main__":
    main()