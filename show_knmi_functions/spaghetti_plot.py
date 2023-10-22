import pandas as pd
import streamlit as st
from utils import get_data
import plotly.graph_objects as go
import datetime as dt


def spaghetti_plot(df, what):
    """Spaghetti plot,
       inspired by https://towardsdatascience.com/make-beautiful-and-useful-spaghetti-plots-with-python-ec4269d7e8c9
       but with a upper-and lowerbound per day (later smoothed)

    Args:
        df (df): dataframe with info. Date is in 'YYYYMMDD'
        what (str): which column to use
    """    
    df['date'] = df['YYYYMMDD']
    df['day_of_year'] = df['date'].dt.strftime('%j')
    date_str = df['DD'].astype(str).str.zfill(2) + '-' + df['MM'].astype(str).str.zfill(2) + '-1900'
    #filter out rows with February 29 dates (gives error with converting to datetime)
    df = df[~((df['date'].dt.month == 2) & (df['date'].dt.day == 29))]
    df['date_1900'] = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
    pivot_df = df.pivot(index='date_1900', columns='YYYY', values=what)
    pivot_df['mean'] = pivot_df.mean(axis=1) 
    pivot_df['std'] = pivot_df.std(axis=1) 
    pivot_df['upper_bound'] = pivot_df['mean'] + 2 * pivot_df['std']
    pivot_df['lower_bound'] = pivot_df['mean'] - 2 * pivot_df['std']

    # smooth the upper and lowerbound. Otherwise it's very ugly/jagged
    for b in ['upper_bound', 'lower_bound']:
        pivot_df[b] = pivot_df[b].rolling(9, center=True).mean()
    lw = pivot_df["lower_bound"]
    up = pivot_df["upper_bound"]
    pivot_df=pivot_df.reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
                        name=f"low",
                        x=pivot_df["date_1900"],
                        #y = pd.concat([lw,up[::-1]]),
                        y=pivot_df["lower_bound"], #+pivot_df["upper_bound"][::-1],
                        mode='lines',
                        fill='tozeroy',
                        fillcolor='rgba(255, 255, 255, 0.0)',
                        line=dict(width=0,
                        color='rgba(0, 0, 0, 1.0)'
                        ),
                        ))
    
    fig.add_trace(go.Scatter(
                        name=f"high",
                        x=pivot_df["date_1900"],
                        y=pivot_df["upper_bound"],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(211, 211, 211, 0.5)',
                        line=dict(width=0,
                        color='rgba(0, 0, 0, 0.0)'
                        ),
                        ))
    
    
    for column in pivot_df.columns[1:-4]:
    
        if column == pivot_df.columns[-5]:
            line = dict(width=1,
                        color='rgba(255, 0, 0, 1)'
                        )
        else:
            line = dict(width=.5,
                        color='rgba(255, 0, 255, 0.5)'
                        )
        fig.add_trace(go.Scatter(
                        name=column,
                        x=pivot_df["date_1900"],
                        y=pivot_df[column],
                        mode='lines',
                        line=line,
                        ))
       

    
 
    fig.update_layout(
            xaxis=dict(title="date",tickformat="%d-%m"),
            yaxis=dict(title=what),
            title=what,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    # Create a spaghetti line plot
   
    #fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig)


       
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    spaghetti_plot(df, 'temp_avg')

if __name__ == "__main__":
    main()
