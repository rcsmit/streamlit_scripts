import streamlit as st
import pandas as pd
import polars as pl

from neerslaganalyse import extreme_claude_ai, extreme_claude_ai_seasonal
from show_knmi_functions.show_plot import show_scatter


def show_extremen(df,what_to_show_):
    mode = st.sidebar.selectbox("Mode [quantile|values]",["quantile","value"],0)
    if mode =="quantile":
        value = st.sidebar.number_input("Treshold",0.0,1.0,.995,format="%.3f")
    else:
        value = st.sidebar.number_input("Treshold",0.0,1000000.0,100.0)

        
    
    _df = df.copy(deep=True)
    df_pl = pl.from_pandas(_df)
    for what_to_show in what_to_show_:
        extreme_claude_ai(df_pl,mode, what_to_show, value)
        extreme_claude_ai_seasonal(df_pl, mode, what_to_show, value)
        if mode =="value":
            df_extreem = _df[_df[what_to_show]>value]
           
            show_scatter(df_extreem,"YYYYMMDD",what_to_show)
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    show_extremen(df)
if __name__ == "__main__":
    # main()
    print ("")