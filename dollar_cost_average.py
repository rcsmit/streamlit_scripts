import yfinance as yf
import pandas as pd

import numpy as np
import streamlit as st
import plotly.express as px
import datetime

# Script to calculate Dollar Cost Averageing
# https://en.wikipedia.org/wiki/Dollar_cost_averaging

#  It retrieves the data from Yahoo Finance, calculates the investment values, and displays the results and line graphs using Streamlit and Plotly Express.
def get_data(choice,  interval, date_to_check_from):
    """Retrieves historical data for the specified choice (ticker symbol) from Yahoo Finance.
    
    Args:
        choice (str): The ticker symbol for the cryptocurrency (e.g., 'BTC-USD' for Bitcoin).
        interval (str): The interval for the historical data (e.g., '1d' for daily, '1h' for hourly, etc.).
        date_to_check_from (str): The starting date to retrieve the data from in the format 'YYYY-MM-DD'.
    
    Returns:
        pd.DataFrame: A dataframe containing the historical data with columns 'Date' and 'Close'.
    """

    data = yf.download(tickers=(choice), start=date_to_check_from,interval=interval,group_by='ticker',auto_adjust=True,prepost=False)
    df = pd.DataFrame(data)
    if len(df) == 0:
        st.error(f"No data or wrong input - {choice}. Maybe put a date in the past?")
        df = None
        st.stop()
    else:
        df['rownumber'] = np.arange(len(df))
    column_name = "close_" + choice
    df[column_name] = df["Close"]
    df = df.reset_index()
    try:
        df["Date"] = df["Datetime"]
    except:
        pass
    df = df[["Date", column_name]]

    return df



def calculate_investment_value(df, interval, periodical_investment_usd):
    """Calculates the investment values based on dollar-cost averaging strategy.
    
    Args:
        df (pd.DataFrame): A dataframe containing the historical data with columns 'Date' and 'Close'.
        interval (int): The investment interval in days.
        investment_amount_usd (float): The amount to invest at each interval in USD.
    
    Returns:
        pd.DataFrame: A dataframe containing the investment values and metrics.
    """
    results = []
    total_investments_btc, total_investment_usd,investment_amount_btc = 0,0,0

    for i in range(len(df)):
        if i % interval == 0:
            current_date =  df["Date"].iloc[i]
            current_rate = df['close_BTC-USD'].iloc[i]
            
            investment_amount_btc = periodical_investment_usd / current_rate
            total_investments_btc += investment_amount_btc
            total_investment_usd += periodical_investment_usd
            current_value_usd = total_investments_btc * current_rate

            result = {
               
                'Date': current_date,
                'Invested Amount (USD)': periodical_investment_usd,
                'Bitcoin Rate': current_rate,
                'Investment Amount (BTC)': investment_amount_btc,
                'Total Investments (BTC)': total_investments_btc,
                'Total Investments (USD)': total_investment_usd,
                'Total Portefeuille Value (USD)': current_value_usd
            }
            results.append(result)

    results_df = pd.DataFrame(results)
    results_df["rendement (%)"] = round(results_df["Total Portefeuille Value (USD)"] / results_df["Total Investments (USD)"]*100,1)
  
    return results_df


def make_plots(results_df, investment_interval, initial_investment):
    """Creates line plots for the investment values using Plotly Express.
    
    Args:
        results_df (pd.DataFrame): A dataframe containing the investment values and metrics.
         interval (int): The investment interval in days.
        investment_amount_usd (float): The amount to invest at each interval in USD.
    """
    
    fig = px.line(results_df, x='Date', y=['Total Investments (USD)', 'Total Portefeuille Value (USD)'],
                title=f'Total Investments and Portefeuille Value - investment: USD {initial_investment}, every {investment_interval} days')
    fig.update_layout(yaxis_title='USD')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    

    st.plotly_chart(fig)
    columns_to_plot = ["Bitcoin Rate","Investment Amount (BTC)", "Total Investments (BTC)","rendement (%)"]                
    # Create line graphs for each column
    for column in columns_to_plot:
        fig = px.line(results_df, x='Date', y=column, title=column)
        st.plotly_chart(fig)

   


def main():
    """Main function to run the dollar-cost averaging tool using Streamlit."""
    st.title("Dollar-cost averaging tool")

    # Set the parameters 
    date_to_check_from = st.sidebar.date_input("Date to check from", datetime.date(2020, 1, 1)).strftime("%Y-%m-%d")
    investment_interval = st.sidebar.number_input("Investment interval (in days)", 0,None,30)  # in days
    initial_investment = st.sidebar.number_input("Investment amount (dollars)", 0,None,100)  # in dollars
   
    df = get_data ("BTC-USD","1d", date_to_check_from)
    # Calculate the investment values
    results_df = calculate_investment_value(df, investment_interval, initial_investment, )
    

  
    make_plots(results_df, investment_interval, initial_investment)
    st.write (results_df)

    
if __name__ == "__main__":
    print(f"_________________________________")
    main()