import pandas as pd
import numpy as np
import streamlit as st
from plotly_calplot import calplot

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
    
def show_calender_heatmap(df, datefield, what_to_show_, percentile_colomap_max=95, log=False):
    # https://python.plainenglish.io/interactive-calendar-heatmaps-with-plotly-the-easieast-way-youll-find-5fc322125db7
    # creating the plot
    #colorscales = ["aggrnyl", "agsunset", "blackbody", "bluered", "blues", "blugrn", "bluyl", "brwnyl", "bugn", "bupu", "burg", "burgyl", "cividis", "darkmint", "electric", "emrld", "gnbu", "greens", "greys", "hot", "inferno", "jet", "magenta", "magma", "mint", "orrd", "oranges", "oryel", "peach", "pinkyl", "plasma", "plotly3", "pubu", "pubugn", "purd", "purp", "purples", "purpor", "rainbow", "rdbu", "rdpu", "redor", "reds", "sunset", "sunsetdark", "teal", "tealgrn", "turbo", "viridis", "ylgn", "ylgnbu", "ylorbr", "ylorrd", "algae", "amp", "deep", "dense", "gray", "haline", "ice", "matter", "solar", "speed", "tempo", "thermal", "turbid", "armyrose", "brbg", "earth", "fall", "geyser", "prgn", "piyg", "picnic", "portland", "puor", "rdgy", "rdylbu", "rdylgn", "spectral", "tealrose", "temps", "tropic", "balance", "curl", "delta", "oxy", "edge", "hsv", "icefire", "phase", "twilight", "mrybm", "mygbm"]
    # https://plotly.com/python/builtin-colorscales/
    for what_to_show in what_to_show_:
        st.subheader(what_to_show)
        df[datefield] = pd.to_datetime(df[datefield])
        years = df[datefield].dt.year.unique()

        # Loop through each year and each what_to_show value
        for year in years:
            df_year = df[df[datefield].dt.year == year]  
            
            # Assuming df[what_to_show] contains the values you want to process

            # colomap_max, defaults to max value of the data
            st.write(df_year[what_to_show])
            df_year[what_to_show]=df_year[what_to_show].fillna(0)
            
            colomap_max = np.percentile(df_year[what_to_show], percentile_colomap_max)
            st.write(colomap_max)
            # Log transform the data            
            if log:
                df_year[what_to_show] = np.log(df_year[what_to_show])
            # # Cap every value above the 95th percentile to the 95th percentile value
            # df_year[what_to_show] = np.where(df_year[what_to_show] > percentile_95, percentile_95, df_year[what_to_show])   
            fig = calplot(
                    df_year,
                    x=datefield,
                    y=what_to_show,
                    years_title=True,
                    name=what_to_show,
                    colorscale = "purples",
                    gap=2,
                    cmap_max  = colomap_max,
                    month_lines_width=2,
                    month_lines_color="black"
                    #space_between_plots=0.15
            )
            st.plotly_chart(fig)

def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    what_to_show_=["temp_max"]
    datefield = "YYYYMMDD"
    show_calender_heatmap(df, datefield, what_to_show_)

if __name__ == "__main__":
    main()
 