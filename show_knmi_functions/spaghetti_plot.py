import pandas as pd
import streamlit as st
from show_knmi_functions.utils import get_data
import plotly.graph_objects as go
import plotly.express as px  # For easy colormap generation
import numpy as np  # For linspace to distribute sampling


def spaghetti_plot(df, what, wdw, wdw_interval, sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient):
    """wrapper for spaghetti plot since show_knmi calles the function with what as list

    Args:
        df (df): _description_
        what (list): _description_

        wdw (int) : window for smoothing the 95% interval
        sd_all :  Show CI calculated with a stdev overall
        sd_day : show CI calculated with a stdev per day
    """    
    for w in what:
        spaghetti_plot_(df, w, wdw, wdw_interval, sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient)

def spaghetti_plot_(df, what, wdw, wdw_interval,  sd_all, sd_day, spaghetti, mean_, last_year, show_quantiles, gradient):
    """Spaghetti plot,
       inspired by https://towardsdatascience.com/make-beautiful-and-useful-spaghetti-plots-with-python-ec4269d7e8c9
       but with a upper-and lowerbound per day (later smoothed)

    Args:
        df (df): dataframe with info. Date is in 'YYYYMMDD'
        what (str): which column to use
        wdw_interval (int) : window for smoothing the 95% interval
    """    
    gemini = True
    if gemini:
        df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
        df['day_of_year'] = df['date'].dt.dayofyear
        df[what] = pd.to_numeric(df[what], errors='coerce')  # Convert to numeric, handle errors

        # Filter and prepare the data more efficiently
        df_filtered = df.dropna(subset=what).copy()  # Avoid modifying original DataFrame
        df_filtered[what] = df_filtered[what].rolling(wdw, center=True).mean()

        # Exclude February 29th dates before pivot
        df_filtered = df_filtered[~((df_filtered['date'].dt.month == 2) & (df_filtered['date'].dt.day == 29))]
        df_filtered['date_1900'] = pd.to_datetime(df_filtered['date'].dt.strftime('%d-%m-1900'), format='%d-%m-%Y')

        # Pivot and calculate statistics
        pivot_df = df_filtered.pivot(index='date_1900', columns='YYYY', values=what)
    else:
            
        df['date'] = df['YYYYMMDD']
        df['day_of_year'] = df['date'].dt.strftime('%j')

        df = df[pd.notna(df[what])]
        df = df.replace('', None)
        df = df.replace('     ', None)
        date_str = df['DD'].astype(str).str.zfill(2) + '-' + df['MM'].astype(str).str.zfill(2) + '-1900'
        #filter out rows with February 29 dates (gives error with converting to datetime)
        df = df[~((df['date'].dt.month == 2) & (df['date'].dt.day == 29))]
        df['date_1900'] = pd.to_datetime(date_str, format='%d-%m-%Y', errors='coerce')
        df[what] = df[what].rolling(wdw, center=True).mean()
        pivot_df = df.pivot(index='date_1900', columns='YYYY', values=what)
    pivot_df = pivot_df.replace('', None)
    # print (pivot_df)
    # print (pivot_df.std().values)
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

    quantiles = pivot_df.iloc[:, :-3].apply(calculate_quantiles, axis=1)
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

    fig = go.Figure()
    if spaghetti:
        num_colors = len(pivot_df.columns[1:-12])
        
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
            
        for i, column in enumerate(pivot_df.columns[1:-12]):
            if column == pivot_df.columns[-13] and last_year:
                # line = dict(width=1,
                #             color='rgba(255, 0, 0, 1)'
                #             )
                pass
            else:
                # line = dict(width=.5,
                #             color='rgba(255, 0, 255, 0.5)'
                #             )
                if gradient != "None":
                    line_color = colors[i]  
                    line = dict(width=0.5, color=line_color)
                else:
                    line = dict(width=0.5)
                
                fig.add_trace(go.Scatter(
                                name=column,
                                x=pivot_df["date_1900"],
                                y=pivot_df[column],
                                
                                mode='lines',
                                line=line,
                                showlegend = True
                                ))
       
    if show_quantiles:
        fig.add_trace(go.Scatter(
                            name=f"low quantile per day",
                            x=pivot_df["date_1900"],
                            #y = pd.concat([lw,up[::-1]]),
                            y=pivot_df["2_5_percentile"], #+pivot_df["upper_bound"][::-1],
                            mode='lines',
                            fill='tozeroy',
                            fillcolor='rgba(255, 255, 255, 0.0)',
                            
                            ))
        
        fig.add_trace(go.Scatter(
                            name=f"high quantile per day",
                            x=pivot_df["date_1900"],
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
                            x=pivot_df["date_1900"],
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
                            x=pivot_df["date_1900"],
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
                            x=pivot_df["date_1900"],
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
                            x=pivot_df["date_1900"],
                            y=pivot_df["upper_bound_all"],
                            mode='lines',
                            fill='tonexty',
                            fillcolor='rgba(211, 211, 0, 0.5)',
                            line=dict(width=1,
                            color='rgba(0, 0, 255, 1.0)'
                            ),
                            ))
    if mean_:
        line = dict(width=.5,
                        color='rgba(0, 0, 255, 1)'
                        )
        fig.add_trace(go.Scatter(
                        name="MEAN",
                        x=pivot_df["date_1900"],
                        y=pivot_df["mean"],
                        mode='lines',
                        line=line,
                        ))
    
    if last_year:
        ly = pivot_df.columns[-13]
        line = dict(width=1,
                    color='rgba(255, 0, 0, 1)'
                    )
          
        fig.add_trace(go.Scatter(
                        name=ly,
                        x=pivot_df["date_1900"],
                        y=pivot_df[ly],
                        mode='lines',
                        line=line,
                        ))
    
    
    fig.update_layout(
            xaxis=dict(title="date",tickformat="%d-%m"),
            yaxis=dict(title=what),
            title=f"{what} - SMA{wdw}" ,)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    #fig.update_traces(hovertemplate=None)  # Disable hover info for faster rendering
    #fig.update_layout(showlegend=False)   # Disable legend for faster rendering

    # Create a spaghetti line plot
   
    #fig.update_layout(xaxis=dict(tickformat="%d-%m"))
    st.plotly_chart(fig)


       
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    spaghetti_plot(df, ['temp_avg'], 7, 7, False, False, True, False, True, False, "Greys")

if __name__ == "__main__":
    main()
