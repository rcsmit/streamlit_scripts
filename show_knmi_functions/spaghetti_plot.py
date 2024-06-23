import pandas as pd
import streamlit as st
try:
    from show_knmi_functions.utils import get_data, date_to_daynumber
except:
    from utils import get_data, date_to_daynumber

import plotly.graph_objects as go
import plotly.express as px  # For easy colormap generation
import numpy as np  # For linspace to distribute sampling

def spaghetti_plot(df, what, wdw, wdw_interval, sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient, cumulative, show_shift=False):
    """wrapper for spaghetti plot since show_knmi calles the function with what as list

    Args:
        df (df): dataframe with info. Date is in 'YYYYMMDD'
        what (list with strings): which column(s) to use
        wdw_interval (int) : window for smoothing the 95% interval
        wdw (int): sma window for the value
        wdw_interval (int): sma window for upper and lower bound
        sd_all (bool): calculate SD over all the values
        sd_day (bool): calculate SD per day
        spaghetti (bool): show the spaghetti
        mean_ (bool): show the mean
        last_year (bool): show the last year
        show_quantiles (bool): show the quantiles - calculated with pd.Series([row.quantile(0.025)])
        gradient (string): One of  "None" (as string), "Pubu", "Purd", "Greys" or "Plasma". 
                            See https://plotly.com/python/builtin-colorscales/
        cumulative (bool): Show the cumulative value
        show_shift : show possibility to shift xas (default:False)
    Returns:
        _type_: _description_
    """     
    df['month'] = df['YYYYMMDD'].dt.month
    
    if show_shift:

        #shift_days = st.sidebar.number_input("Shift days", 0,365,0)
        date_str = st.sidebar.text_input("X axis starts at (dd-mm)", "01-01")
        shift_days = date_to_daynumber(date_str) -1
        if shift_days ==0:
            min = st.sidebar.number_input("Maand minimaal (incl)", 1,12,1)
            max = st.sidebar.number_input("Maand maximaal (incl)", 1,12,12)
            df = df[(df['month'] >= min) & (df['month'] <= max)] 
    else:
        shift_days=0
    df["dayofyear"] = df['YYYYMMDD'].dt.dayofyear
    
    for w in what:
        spaghetti_plot_(df, w, wdw, wdw_interval, sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient, cumulative, shift_days)
        
def spaghetti_plot_(df, what, wdw, wdw_interval,  sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient, cumulative,shift_days):
    """Spaghetti plot,
       inspired by https://towardsdatascience.com/make-beautiful-and-useful-spaghetti-plots-with-python-ec4269d7e8c9
       but with a upper-and lowerbound per day (later smoothed)
    """    
       

    df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year
    
    # Function to check for leap year
    def is_leap_year(year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # Adjust date_of_year for leap years
    df['day_of_year'] = df.apply(
        lambda row: row['day_of_year'] - 1 if is_leap_year(row['year']) and row['day_of_year'] >= 61 else row['day_of_year'], 
        axis=1
    )
    df = df[~((df['date'].dt.month == 2) & (df['date'].dt.day == 29))]
    
    df["day_of_year_shifted"] = df["day_of_year"].shift(shift_days)
    df["year_shifted"] = df["year"].shift(shift_days)
    
    df[what] = pd.to_numeric(df[what], errors='coerce')  # Convert to numeric, handle errors
    df=df.fillna(0)
   
    # Filter and prepare the data more efficiently
    df_filtered = df.dropna(subset=what).copy()  # Avoid modifying original DataFrame
    df_filtered[what] = df_filtered[what].rolling(wdw, center=True).mean()
    
    # Exclude February 29th dates before pivot
    
    df_filtered = df_filtered[~(df_filtered['day_of_year_shifted'] == 0)]
    df_filtered['date_1900'] = pd.to_datetime(df_filtered['date'].dt.strftime('%d-%m-1900'), format='%d-%m-%Y')
    df_filtered=df_filtered.fillna(0)
    # Move 'neerslag_etmaalsom' to be the last column
    columns = list(df_filtered.columns)
    columns.append(columns.pop(columns.index('neerslag_etmaalsom')))
    df_filtered = df_filtered[columns]


    # Compute the cumulative value, starting each year at January 1st
    
    # df_filtered['cumulative'] = df_filtered.groupby('year')[what].cumsum()
    pivot_df = df_filtered.pivot(index=['day_of_year_shifted', 'date_1900'], columns='year_shifted', values=what)
    if cumulative:
        pivot_df = pivot_df.cumsum(axis=0)

    # Go through the specified columns and replace 0 with the value from the row above
      
    pivot_df['std_all'] = pivot_df.std().values.mean()
    try:
        pivot_df['mean'] =  pivot_df.iloc[:, :-1].mean(axis=1) 
        pivot_df['std'] = pivot_df.iloc[:, :-2].std(axis=1) 
    except:
        st.error(f"There are empty values in {what}")
        return
    
   
    # Assuming pivot_df is your DataFrame
    def calculate_quantiles(row):
        return pd.Series([row.quantile(0.025),row.quantile(0.25), row.quantile(0.5), row.quantile(0.75),row.quantile(0.975)])
   
  
    quantiles = pivot_df.iloc[:, 2:-3].apply(calculate_quantiles, axis=1)
    quantiles.columns = ['2_5_percentile', '25th Percentile', 'Median (50th Percentile)', '75th Percentile','97_5_percentile']

    # Add the quantiles back to your DataFrame
    pivot_df = pd.concat([pivot_df, quantiles], axis=1)

# Now, pivot_df contains the original columns along with the quantiles
    # print (pivot_df)
    pivot_df['upper_bound'] = pivot_df['mean'] + 2 * pivot_df['std']
    pivot_df['lower_bound'] = pivot_df['mean'] - 2 * pivot_df['std']

    pivot_df['upper_bound_all'] = pivot_df['mean'] + 2 * pivot_df['std_all']
    pivot_df['lower_bound_all'] = pivot_df['mean'] - 2 * pivot_df['std_all']

    # smooth the upper and lowerbound. Otherwise it's very ugly/jagged
    for b in ['upper_bound', 'lower_bound','upper_bound_all','lower_bound_all','2_5_percentile', '25th Percentile', 'Median (50th Percentile)', '75th Percentile','97_5_percentile']:
        pivot_df[b] = pivot_df[b].rolling(wdw_interval, center=True).mean()
    lw = pivot_df["lower_bound"]
    up = pivot_df["upper_bound"]
    pivot_df=pivot_df.reset_index()
    # for column in pivot_df.columns[2:-12]:
    #     for i in range(1, len(pivot_df)):
    #         if pivot_df.at[i, column] == 0:
    #             pivot_df.at[i, column] = pivot_df.at[i - 1, column]
                
   
   
    x_axis = pivot_df["day_of_year_shifted"]

    
    fig = go.Figure()
    if spaghetti:
        num_colors = len(pivot_df.columns[2:-12])
        
        if gradient != "None":
            if gradient =="Pubu": 
                colormap = px.colors.sequential.PuBu
            elif gradient =="Purd": 
                colormap = px.colors.sequential.PuRd
            elif gradient =="Greys": 
                colormap = px.colors.sequential.Greys
            elif gradient =="Plasma":
                colormap = px.colors.sequential.Plasma
            else:
                st.error("Error in colormap")
                st.stop()
        
            # colorscales
            color_indices = np.linspace(0, len(colormap) - 1, num_colors, dtype=int)
            colors = [colormap[i] for i in color_indices] 
        if last_year:
            end = -13
        else:
            end = -12 
        for i, column in enumerate(pivot_df.columns[2:end]):
            if cumulative & (shift_days!=0):
                ly = pivot_df.columns[-13]
       
                count_ly = pivot_df[ly].count()
                if count_ly  + shift_days <365 :
                
                    name = f"{int(column)} - {int(column)+1}"
                else:
                    name = f"{int(column)} - {int(column)+1}"
            else:
                name = f"{int(column)}"
            if gradient != "None":
                line_color = colors[i]  
                line = dict(width=0.5, color=line_color)
            else:
                line = dict(width=0.5)
                
            fig.add_trace(go.Scatter(
                            name=name,
                            x=x_axis,
                            y=pivot_df[column],
                            hovertext=pivot_df["date_1900"].dt.strftime('%d-%m'),
                            mode='lines',
                            line=line,
                            showlegend = True
                            ))
    
    if show_quantiles:
        fig.add_trace(go.Scatter(
                            name=f"low quantile per day",
                            x=x_axis,
                            #y = pd.concat([lw,up[::-1]]),
                            y=pivot_df["2_5_percentile"], #+pivot_df["upper_bound"][::-1],
                            mode='lines',
                            fill='tozeroy',
                            fillcolor='rgba(255, 255, 255, 0.0)',
                            
                            ))
        
        fig.add_trace(go.Scatter(
                            name=f"high quantile per day",
                            x=x_axis,
                            y=pivot_df["97_5_percentile"],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(211, 211, 211, 0.5)',
                            line=dict(width=1,
                            color='rgba(255, 128, 0, 1.0)'
                            ),
                            ))
    if sd_day:
        fig.add_trace(go.Scatter(
                            name=f"low CI per day",
                            x=x_axis,
                            #y = pd.concat([lw,up[::-1]]),
                            y=pivot_df["lower_bound"], #+pivot_df["upper_bound"][::-1],
                            mode='lines',
                            fill='tozeroy',
                            fillcolor='rgba(255, 255, 255, 0.0)',
                            line=dict(width=1,
                            color='rgba(0, 255, 0, 1.0)'
                            ),
                            ))
        
        fig.add_trace(go.Scatter(
                            name=f"high  Ci per day",
                            x=x_axis,
                            y=pivot_df["upper_bound"],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(211, 211, 211, 0.5)',
                            line=dict(width=1,
                            color='rgba(0, 255, 0, 1.0)'
                            ),
                            ))
    if sd_all:
        fig.add_trace(go.Scatter(
                            name=f"low CI ALL",
                            x=x_axis,
                            #y = pd.concat([lw,up[::-1]]),
                            y=pivot_df["lower_bound_all"], #+pivot_df["upper_bound"][::-1],
                            mode='lines',
                            fill='tozeroy',
                            fillcolor='rgba(255, 255, 255, 0.0)',
                            line=dict(width=1,
                            color='rgba(0, 0, 255, 1.0)'
                            ),
                            ))
        
        fig.add_trace(go.Scatter(
                            name=f"high  CI ALL",
                            x=x_axis,
                            y=pivot_df["upper_bound_all"],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(211, 211, 0, 0.5)',
                            line=dict(width=1,
                            color='rgba(0, 0, 255, 1.0)'
                            ),
                            ))
    if mean_:
        line = dict(width=.8,
                        color='rgba(0, 0, 255, 1)'
                        )
        fig.add_trace(go.Scatter(
                        name="MEAN",
                        x=x_axis,
                        y=pivot_df["mean"],
                        mode='lines',
                        line=line,
                        ))
    
    if last_year:
        ly = pivot_df.columns[-13]
       
        count_ly = pivot_df[ly].count()
       
        if count_ly  + shift_days <365 :
            
            name = int(ly)
        else:
            name = f"{int(column)+1} - {int(column)+2}"
        line = dict(width=1.7,
                    color='rgba(255, 0, 0, 1)'
                    )
          
        fig.add_trace(go.Scatter(
                        name=name,
                        x=x_axis,
                        y=pivot_df[ly],
                        hovertext=pivot_df["date_1900"].dt.strftime('%d-%m'),
                        mode='lines',
                        line=line,
                        ))
    
    # Filter for every 10th value
    every_10th_value = pivot_df.iloc[::10, :]
    x_axis = every_10th_value["day_of_year_shifted"]
    fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=x_axis,
                ticktext=[date.strftime('%d-%m') for date in every_10th_value['date_1900']]
                
                ),
            
            yaxis=dict(title=what),
            title=f"{what} - SMA{wdw}" ,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)


    # Customize the x-axis
    # fig.update_layout(
    #     xaxis_title='Custom Dates',
    #     xaxis=dict(
            #tickmode='array',
            #tickvals=pivot_df['date_1900'],
            #ticktext=[date.strftime('%d-%m') for date in pivot_df['date_1900']]
    #     )
    # )
    
    #fig.update_traces(hovertemplate=None)  # Disable hover info for faster rendering
    #fig.update_layout(showlegend=False)   # Disable legend for faster rendering

    # Create a spaghetti line plot
   
    #fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig)


       
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    df['neerslag_etmaalsom'] = df['neerslag_etmaalsom'].replace(-0.1, 0)

    spaghetti_plot(df, ['neerslag_etmaalsom'], 7, 7, False, False, True, False, True, False, "Plasma", True)

if __name__ == "__main__":
    main()
