import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from show_knmi_functions.utils import get_data

def show_per_periode(df, gekozen_weerstation, what_to_show_, groeperen, graph_type):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    df.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    jaren = df["YYYY"].tolist()
    df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)]
    df['DD'] = df['DD'].astype(str).str.zfill(2)
    df['MM'] = df['MM'].astype(str).str.zfill(2)
    df["mmdd"] = df["MM"] +"-" + df["DD"]

    df["year"] = df["year"].astype(str)
    df['week'] = df['YYYYMMDD'].dt.isocalendar().week

    # df.to_csv("C:\\Users\\rcxsm\\Documents\\weekgemiddelden.csv")

    for what_to_show in what_to_show_:
        if groeperen == "maandgem":
            col_to_group = "MM"
        elif groeperen == "weekgem":
            # Group by year and week number, then calculate the mean
            col_to_group = "week"
        elif groeperen == "per_dag":
            #df["MD"] = df["month_day"]
            col_to_group = "mmdd"
            
        df_grouped = df.groupby(["year", col_to_group]).mean(numeric_only = True).reset_index()
        df_grouped ["year"] = df_grouped ["year"].astype(str)   
        # df_grouped.to_csv("C:\\Users\\rcxsm\\Documents\\weekgemiddelden.csv")
            
        df_pivoted = df_grouped.pivot(
                index=col_to_group, columns="year", values=what_to_show
            ).reset_index()
        
    
        if graph_type == "plotly":
            fig = go.Figure()

            if groeperen == "maandgem":
                sma = [go.Scatter(x=pd.Series(df_pivoted.MM), y=df_pivoted[c],  
                   mode='lines', name=f'{c}')
                   for c in df_pivoted.columns[1:]]
            elif groeperen == "weekgem":
                sma = [go.Scatter(x=pd.Series(df_pivoted.week), y=df_pivoted[c],  
                   mode='lines', name=f'{c}')
                   for c in df_pivoted.columns[1:]]
            else:

                sma = []
                list_years = df_pivoted.columns[1:]
                try:
                    highlight_year = st.sidebar.selectbox("Highlight year", list_years,len(list_years)-1)
                except:
                    pass
                for i, col in enumerate(df_pivoted.columns[1:]):
                    line_width = 0.7 if col != highlight_year  else 3  # make last column thicker
                    trace = go.Scatter(x=[df_pivoted.index, df_pivoted.mmdd], y=df_pivoted[col],
                                    mode='lines', line=dict(width=line_width), name=col)
                    sma.append(trace)

            data = sma
            title = (f"{ what_to_show} -  {gekozen_weerstation}")
            layout = go.Layout(
              
                yaxis=dict(title=what_to_show), 
                title=title,)
            
            fig = go.Figure(data=data, layout=layout)
        
            st.plotly_chart(fig, use_container_width=True)
        
        
        
        else:
            st.warning ("Under construction")


        
def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    show_per_periode(df, "De Bilt", ["temp_avg"], "maandgem", "plotly")
if __name__ == "__main__":
    main()
    