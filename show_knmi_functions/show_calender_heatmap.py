import pandas as pd
import numpy as np
import streamlit as st
from plotly_calplot import calplot

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
    
def show_calender_heatmap(df, datefield, what_to_show_):
    # https://python.plainenglish.io/interactive-calendar-heatmaps-with-plotly-the-easieast-way-youll-find-5fc322125db7
    # creating the plot
    for what_to_show in what_to_show_:
        st.subheader(what_to_show)
        df[datefield] = pd.to_datetime(df[datefield])
        years = df[datefield].dt.year.unique()

        # Loop through each year and each what_to_show value
        for year in years:
            df_year = df[df[datefield].dt.year == year]      
            fig = calplot(
                    df_year,
                    x=datefield,
                    y=what_to_show,
                    years_title=True,
                    name=what_to_show,
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
 