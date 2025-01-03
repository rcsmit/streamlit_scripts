import pandas as pd
import numpy as np
import streamlit as st
from plotly_calplot import calplot
import plotly.express as px
import plotly.graph_objects as go
try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
    
def show_year_heatmap(df, datefield, what_to_show_):
    """Show a year heatmap. The x-values are the days (but normalized so
    every month has the same width), the y values are the months,
    and the z-values are the value to show 

    Args:
        df (_type_): _description_
        datefield (_type_): _description_
        what_to_show_ (_type_): _description_
    """
    for what_to_show in what_to_show_:
        st.subheader(what_to_show)

        df[datefield] = pd.to_datetime(df[datefield])
        years = df[datefield].dt.year.unique()
        months =  df[datefield].dt.month.unique()
                
        df["day"] = df[datefield].dt.day
        df["days_in_month"] = df[datefield].dt.days_in_month
        df["Month_progress"] = round((df["day"]-1) / (df["days_in_month"]-1) * 100,1)
         # Fill missing values with interpolation
        
        # Ensure all unique Month_progress values are present
        unique_month_progress = df["Month_progress"].unique()
        unique_month_progress.sort()
  
        # Loop through each year and each what_to_show value
        for year in df['YYYY'].unique():
            df_year_list = []

            # Loop through each month
            for month in df['MM'].unique():
                df_month = df[(df['YYYY'] == year) & (df['MM'] == month)]
                # Reindex to ensure all Month_progress values are present
                df_month = df_month.set_index('Month_progress').reindex(unique_month_progress).reset_index()

                # Fill missing values with the value from the row below
                df_month[what_to_show] = df_month[what_to_show].ffill().bfill()
                df_month["MM"] = df_month["MM"].ffill()
            
                # Append the df_month to the list
                df_year_list.append(df_month)

            # Combine all df_months into a new df_year
            df_year = pd.concat(df_year_list, ignore_index=True)

            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=df_year[what_to_show],
                y=df_year["MM"],
                x=df_year["Month_progress"],
                colorscale=[
                    [0, 'darkgreen'],      # Good
                    [0.2, 'green'],        # Moderate
                    [0.4, 'yellow'],       # Unhealthy for Sensitive Groups
                    [0.6, 'orange'],       # Unhealthy
                    [0.8, 'red'],          # Very Unhealthy
                    [1.0, 'purple']        # Hazardous
                ],
                showscale=True,
                colorbar=dict(
                    title=what_to_show,
                    titleside='right'
                ),
                hovertemplate=f'{what_to_show}: %{{z}}<extra></extra>'
            ))

            # Update layout
            fig.update_layout(
                title=f"{what_to_show} - {year}",
                xaxis_title='',
                yaxis_title='',
                yaxis_autorange='reversed',
                height=600,
                width=800,
                xaxis_visible=False,
                plot_bgcolor='white',
                yaxis=dict(
                    tickmode='linear',
                    dtick=1,
                    ticklen=10  # Adjust this value to increase/decrease the space between y-values
                )
            )

            # Show the plot
                    # Show figure
            st.plotly_chart(fig)

def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\result.csv"
    df = get_data(url)
    what_to_show_=["temp_max"]
    datefield = "YYYYMMDD"
    show_year_heatmap(df, datefield, what_to_show_)

if __name__ == "__main__":
    main()
