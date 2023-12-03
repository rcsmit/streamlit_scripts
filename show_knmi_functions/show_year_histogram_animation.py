import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.animation import FuncAnimation
try:
    from show_knmi_functions.utils import get_data, loess_skmisc
except:
    from utils import get_data, loess_skmisc

import plotly.express as px
import plotly.graph_objects as go

import imageio
import os
import webbrowser

from scipy.stats import norm 
import statistics 


def show_animation_histogram_matplotlib(df, what):
    

    # reproducing https://www.linkedin.com/feed/update/urn:li:activity:7134137565661556736/
    # https://svs.gsfc.nasa.gov/5065/

    df['year'] = df['YYYYMMDD'].dt.year
    df[what] = df[what].round().astype(int)
    # Create a pivot table with Year as index and temp_avg frequencies as columns
    pivot_table = df.pivot_table(index='year', columns=what, aggfunc='size', fill_value=0).reset_index()

    pivot_table.set_index('year', inplace=True)
    dfs = pivot_table.unstack().reset_index()
    dfs.columns = [what, 'year',  'value']
    
    fig = px.bar(dfs, x=what, y='value', animation_frame='year', range_y=[0,dfs['value'].max()])

    
    st.plotly_chart(fig)

def make_line_graph_in_time(df, what):

    def make_graph(x,y,low,high, std, title):
    
        # Create Plotly figure
        fig = go.Figure()

        # Add mean as a line plot
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name='Mean'
        ))
        fig.add_trace(go.Scatter(
            x=x,
            y=std,
            mode='lines',
            name='std'
        ))

        fig.add_trace(go.Scatter(
            name = "low",         
            x=x,
            y=low,
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(255, 255, 255, 0.0)',
            line=dict(width=1,
            color='rgba(0, 255, 0, 1.0)'
            ),
            ))
            
        fig.add_trace(go.Scatter(
            name = "high", 
            x=x,
            y=high,
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(211, 211, 211, 0.5)',
            line=dict(width=1,
            color='rgba(0, 255, 0, 1.0)'
            ),
            ))
                                
    
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Temperature',
            showlegend=True
        )

        # Show plot
        st.plotly_chart(fig)

        
    # # Calculate mean and standard deviation for each year
    summary_table = df.groupby('year')[what].agg(['mean', 'std']).reset_index()
        
    # Calculate confidence interval (2 times the standard deviation)
    summary_table['lower_bound'] =  summary_table['mean'] - ( 2 * summary_table['std'])
    summary_table['upper_bound'] =  summary_table['mean'] + ( 2 * summary_table['std'])

    x = summary_table['year'].to_list()
    mean = summary_table['mean'].to_list() 
    low = summary_table['lower_bound'].to_list() 
    high = summary_table['upper_bound'].to_list()
    std = summary_table['std'].to_list()

    #  span = 42/len(y), wat de 30 jarig doorlopend gemiddelde benadert
    # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/standaardmethode-voor-berekening-van-een-trend
    # KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)
    _, loess_mean, __,___ = loess_skmisc(x, mean,  ybounds=None, it=1)
    _, loess_low, __,___ = loess_skmisc(x, low,  ybounds=None, it=1)
    _, loess_high, __,___ = loess_skmisc(x, high,  ybounds=None, it=1)
    _, loess_std, __,___ = loess_skmisc(x, std,  ybounds=None, it=1)

    make_graph(x,mean,low,high, std, f"{what} and 95% CI")
    make_graph(x,loess_mean,loess_low,loess_high, loess_std, f"{what} and 95% CI LOESS - benadert 30 jarig doorlopend gemiddelde")

def make_gif_gaussian_distributions(df, what):
     
 
    unique_years = df['year'].unique()

    def make_graph(year):
        fig = plt.figure()
        values = df[df['year'] == year][what].tolist()
        print(year)
        x_axis = np.arange(0, 40, 1) 
        mean = statistics.mean(values) 
        sd = statistics.stdev(values) 
        plt.plot(x_axis, norm.pdf(x_axis, mean, sd)) 
        plt.axvline(mean, color='r', linestyle='--')

        filename= (f"histogram_{year}")
        plt.ylim(0, 0.1)
        plt.title(f"Distribution of {what} for year {year}")
        plt.savefig(filename, dpi=100,)
        plt.close()
        return filename

    filenames = []
    for i in unique_years:
        filename = make_graph(i)
        filenames.append(filename)

    # build gif
    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename_ in filenames:
            image = imageio.imread(f"{filename_}.png")
            writer.append_data(image)
    webbrowser.open('mygif.gif')

    # Remove files
    for filename__ in set(filenames):
        os.remove(f"{filename__}.png")


def show_year_histogram_animation(df, what):
    """MAIN FUNCTION

    Args:
        df (_type_): _description_
    """
    show_animation_histogram_matplotlib(df, what)
    make_line_graph_in_time(df, what)
    make_gif_gaussian_distributions(df, what)

       
def main():
    what= "temp_avg"
    url = "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=260&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX&start=19010101&end=20991231"
    df = get_data(url)
    show_year_histogram_animation(df, what)

if __name__ == "__main__":
    main()