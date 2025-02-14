import yfinance as yf
import pandas as pd

import numpy as np
import streamlit as st
import plotly.express as px
import datetime
from utils import get_data_yfinance

# Script to calculate Dollar Cost Averageing
# https://en.wikipedia.org/wiki/Dollar_cost_averaging

#  It retrieves the data from Yahoo Finance, calculates the investment values, and displays the results and line graphs using Streamlit and Plotly Express.
# TODO: Yahoo Finnance BTC rate begins only in 2017.

@st.cache_data()
def get_data_old(choice,  interval, date_to_check_from):
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
   
    # ["rendement (%)"] = round(results_df["Total Portefeuille Value (USD)"] / results_df["Total Investments (USD)"]*100,1)
    results_df["rendement (%)"] = round((results_df["Total Portefeuille Value (USD)"]-results_df["Total Investments (USD)"]) / results_df["Total Investments (USD)"]*100,1)
  
    return results_df


def make_plots_one_starting_date(results_df, investment_interval, initial_investment):
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

    columns_to_plot = ["rendement (%)","Bitcoin Rate","Investment Amount (BTC)", "Total Investments (BTC)"]                
    # Create line graphs for each column
    for column in columns_to_plot:
        fig = px.line(results_df, x='Date', y=column, title=column)
        if column == "rendement (%)":
        # Add horizontal line at y = 100
            fig.add_shape(type="line", x0=results_df['Date'].min(), x1=results_df['Date'].max(),
                        y0=0, y1=0, line=dict(color="red", dash="dash"))
        st.plotly_chart(fig)


def rendement_various_starting_dates(investment_interval, initial_investment):
    """
    Calculate rendement with various starting dates (1st of each month since 1/1/2017) and plot the results.

    Parameters:
        investment_interval (int): The interval between investments in days.
        initial_investment (float): The initial investment amount in USD.

    Returns:
        None
    """
    df = get_data_yfinance ("BTC-USD","1d", "2017-01-01")
    df['Date'] = pd.to_datetime(df['Date'])

    start_date_ = st.sidebar.text_input("Start date", '2017-01-01')
    end_date_ = st.sidebar.text_input("End date", '2099-12-31')
    
    rendement_data = []
    try:
        start_date = pd.Timestamp(start_date_)
        if end_date_ == '2099-12-31':
            end_date = pd.Timestamp.today()
        else:
            end_date = pd.Timestamp(end_date_)
    except:
        st.error("Error. Is the date in the right format? (yyyy-mm-dd) ?")
        st.stop()

    df=df[(df['Date'] >= start_date) & (df['Date'] <= end_date ) ] 

    if len(df)==0:
        st.error("No values found. Is the start date before the end date?")
        st.stop()
    
    st.subheader(f"Rendement with various starting dates (1st of month since {start_date_})")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    for i,date in enumerate(date_range):
        print (f"{i+1}/{len(date_range)}")
       
        # Specify the date after which you want to include rows
        filter_date = pd.Timestamp(date)

        # Create a boolean mask to filter rows
        mask = df['Date'] >= filter_date

        # Apply the mask to filter the DataFrame
        filtered_df = df[mask]


        # Calculate the investment values
        results_df = calculate_investment_value(filtered_df, investment_interval, initial_investment)
        last_rendement = results_df['rendement (%)'].iloc[-1]
        total_investments_USD = results_df['Total Investments (USD)'].iloc[-1]
        total_portefeuille_value_USD = results_df['Total Portefeuille Value (USD)'].iloc[-1]
        bitcoin_rate = results_df['Bitcoin Rate'].iloc[-1]

        lumpsum_result = total_investments_USD / results_df['Bitcoin Rate'].iloc[0] * bitcoin_rate

        rendement_data.append({'Date': date, 'Rendement_DCA': last_rendement, 'Bitcoin rate': bitcoin_rate,
                               'total_investments_USD':total_investments_USD,
                               'total_portefeuille_value_USD':total_portefeuille_value_USD,
                               'lumpsum_result':lumpsum_result })


    rendement_df = pd.DataFrame(rendement_data)

    # Assuming you have a DataFrame called 'rendement_df' with columns 'date', 'money invested', and 'worth portfolio'

    # Convert the 'date' column to datetime if it's not already
    rendement_df['Date'] = pd.to_datetime(rendement_df['Date'])

    # Calculate the number of years for each row
    today = pd.to_datetime(date.today())
    rendement_df['years'] = (today - rendement_df['Date']).dt.days / 365

    # Calculate the rendement per year using Pandas
    rendement_df['rendement per year_DCA'] = ((rendement_df['total_portefeuille_value_USD'] / 
                                           rendement_df['total_investments_USD']) ** 
                                           (1 / rendement_df['years']) - 1) * 100
    
    rendement_df['rendement per year_lumpsum'] = ((rendement_df['lumpsum_result'] / 
                                           rendement_df['total_investments_USD']) ** 
                                           (1 / rendement_df['years']) - 1) * 100

    # deleting last row (always 0% rendement since you just bought it)
    rendement_df = rendement_df.iloc[:-1]
    
    # Plotting with Plotly
    plot_rendement_DCA_and_rendement_per_year_DCA(rendement_df)
    plot_rendement_per_year_DCA_vs_lumpsum(rendement_df)
    plot_invested_DCA_vs_lumpsum(rendement_df)
    plot_btc_rate(df)

    with st.expander("Rendement DF"):
        st.write(rendement_df)

def plot_rendement_DCA_and_rendement_per_year_DCA(rendement_df):
    """Plot total rendement and rendement per year while using DCA
    Args:
        
        rendement_df (_type_): _description_
    """    
    for y_ in ['rendement per year_DCA', 'Rendement_DCA']: #'rendement per year_DCA', 'rendement per year_lumpsum' 
        fig = px.line(rendement_df, x='Date', y=y_, markers=False)
     
        fig.add_shape(type="line", x0=rendement_df['Date'].min(), x1=rendement_df['Date'].max(),
                    y0=0, y1=0, line=dict(color="red", dash="dash"))
        fig.update_layout(title=y_, xaxis_title='Date', yaxis_title=f"{y_} (%)")
        st.plotly_chart(fig)

def plot_btc_rate(df):
    """Make a simple plot of BTC rate in time
    """    
    fig = px.line(df, x='Date', y='close_BTC-USD', markers=False)
    fig.update_layout(title='BTC-USD', xaxis_title='Date', yaxis_title='BTC-USD')
    st.plotly_chart(fig)

def plot_rendement_per_year_DCA_vs_lumpsum(rendement_df):
    """plot rendement per year DCA vs  rendement per year lumpsum
    """    
    fig = px.line(rendement_df, x='Date', y=['rendement per year_DCA', 'rendement per year_lumpsum'], markers=False)
  
    fig.add_shape(type="line", x0=rendement_df['Date'].min(), x1=rendement_df['Date'].max(),
                y0=0, y1=0, line=dict(color="red", dash="dash"))
    fig.update_layout(title='DCA vs lumpsum', xaxis_title='Date', yaxis_title='Rendement (%)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig)

def plot_invested_DCA_vs_lumpsum(rendement_df):
    """plot rendement per year DCA vs  rendement per year lumpsum
    """    
    fig = px.line(rendement_df, x='Date', y=['total_investments_USD', 'total_portefeuille_value_USD', 'lumpsum_result'], markers=False)
  
    
    fig.update_layout(title='Total invested vs DCA vs lumpsum', xaxis_title='Date', yaxis_title='Amount (USD)')
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig)
    
def rendement_one_starting_date(investment_interval, initial_investment):
    date_to_check_from = st.sidebar.date_input("Date to check from", datetime.date(2020, 1, 1)).strftime("%Y-%m-%d")
    
    df = get_data ("BTC-USD","1d", date_to_check_from)
    #Calculate the investment values
    results_df = calculate_investment_value(df, investment_interval, initial_investment, )
    

  
    make_plots_one_starting_date(results_df, investment_interval, initial_investment)
    st.write (results_df)


def main():
    """Main function to run the dollar-cost averaging tool using Streamlit."""
    st.title("Dollar-cost averaging tool")
    st.info("https://rene-smit.com/dollar-cost-averaging-or-all-at-once/")
    # Set the parameters 
    
    what = st.sidebar.selectbox("What to do", ["one date", "various dates"], 1)
    investment_interval = st.sidebar.number_input("Investment interval (in days)", 0,None,30)  # in days
    initial_investment = st.sidebar.number_input("Investment amount (dollars)", 0,None,100)  # in dollars
    if what == "one date":
        rendement_one_starting_date(investment_interval, initial_investment)
    elif what == "various dates":
        rendement_various_starting_dates(investment_interval, initial_investment)
    else:
        st.error("Error in WHAT")
        st.stop()
    
if __name__ == "__main__":
    print(f"_________________________________")
    
    main()
    