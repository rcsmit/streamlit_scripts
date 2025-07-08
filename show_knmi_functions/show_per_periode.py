import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go

try:
    from show_knmi_functions.utils import get_data, loess_skmisc
except:
    from utils import get_data, loess_skmisc



def show_per_periode(df, gekozen_weerstation, what_to_show_, groeperen, graph_type):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    df.set_index("YYYYMMDD")
    (month_min, month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1, 12))

    jaren = df["YYYY"].tolist()
    df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)]
    df['DD'] = df['DD'].astype(str).str.zfill(2)
    df['MM'] = df['MM'].astype(str).str.zfill(2)
    df["mmdd"] = df["MM"] + "-" + df["DD"]

    df["year"] = df["year"].astype(str)
    df['week'] = df['YYYYMMDD'].dt.isocalendar().week

    for what_to_show in what_to_show_:
        if groeperen == "maandgem":
            col_to_group = "MM"
            df_grouped = df.groupby(["year", col_to_group]).mean(numeric_only=True).reset_index()
            df_grouped["year"] = df_grouped["year"].astype(str)
            df_pivoted = df_grouped.pivot(index=col_to_group, columns="year", values=what_to_show).reset_index()
        elif groeperen == "weekgem":
            col_to_group = "week"
            df_grouped = df.groupby(["year", col_to_group]).mean(numeric_only=True).reset_index()
            df_grouped["year"] = df_grouped["year"].astype(str)
            df_pivoted = df_grouped.pivot(index=col_to_group, columns="year", values=what_to_show).reset_index()
        elif groeperen == "per_dag":
            col_to_group = "mmdd"
            df_grouped = df.groupby(["year", col_to_group]).mean(numeric_only=True).reset_index()
            df_grouped["year"] = df_grouped["year"].astype(str)
            df_pivoted = df_grouped.pivot(index=col_to_group, columns="year", values=what_to_show).reset_index()
        elif groeperen == "maand_per_jaar":
            df_grouped = df.groupby(["YYYY", "MM"]).mean(numeric_only=True).reset_index()
            df_grouped["YYYY"] = df_grouped["YYYY"].astype(str)
            df_grouped["MM"] = df_grouped["MM"].astype(int)
            df_pivoted = df_grouped.pivot(index="YYYY", columns="MM", values=what_to_show).reset_index()

        if graph_type == "plotly":
            fig = go.Figure()
            title = f"{what_to_show} - {gekozen_weerstation}"
            non_sma=None
            if groeperen == "maandgem":
                sma = [
                    go.Scatter(x=pd.Series(df_pivoted["MM"]), y=df_pivoted[col],
                               mode='lines', name=f'{col}')
                    for col in df_pivoted.columns[1:]
                ]
            elif groeperen == "weekgem":
                sma = [
                    go.Scatter(x=pd.Series(df_pivoted["week"]), y=df_pivoted[col],
                               mode='lines', name=f'{col}')
                    for col in df_pivoted.columns[1:]
                ]
            elif groeperen == "per_dag":
                sma = []
                list_years = df_pivoted.columns[1:]
                try:
                    highlight_year = st.sidebar.selectbox("Highlight year", list_years, len(list_years) - 1)
                except:
                    highlight_year = None
                for i, col in enumerate(df_pivoted.columns[1:]):
                    line_width = 0.7 if col != highlight_year else 3
                    trace = go.Scatter(x=df_pivoted["mmdd"], y=df_pivoted[col],
                                       mode='lines', line=dict(width=line_width), name=col)
                    sma.append(trace)
            elif groeperen == "maand_per_jaar":

                non_sma = [
                        go.Scatter(x=df_pivoted["YYYY"], y=df_pivoted[month],
                                mode="lines", name=f"Maand {month}")
                        for month in df_pivoted.columns[1:]
                    ]

                sma = []
                
                df_pivoted = df_pivoted[df_pivoted["YYYY"]!="2025"]  # exclude 2025 if present, gives an error with loess
                                                                    #smoothening (gives curves until 2003 for months that are
                                                                    # not there yet)
             
                for month in df_pivoted.columns[1:]:
                    _, y_hat, _, _ = loess_skmisc(df_pivoted["YYYY"], df_pivoted[month])
            
                    trace = go.Scatter(x=df_pivoted["YYYY"], y=y_hat, mode="lines", name=f"Maand {month}")
                    sma.append(trace)

            fig = go.Figure(data=sma, layout=go.Layout(yaxis=dict(title=what_to_show), title=title))
            st.plotly_chart(fig, use_container_width=True)

            if non_sma:
                fig = go.Figure(data=non_sma, layout=go.Layout(yaxis=dict(title=what_to_show), title=title))
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Under construction")

def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result_1901.csv"
    df = get_data(url)
    st.sidebar.title("Instellingen")
    groepering = st.sidebar.selectbox("Groeperen op", ["maandgem", "weekgem", "per_dag", "maand_per_jaar"])
    show_per_periode(df, "De Bilt", ["temp_avg"], groepering, "plotly")

        
# def main():
#     url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
#     df = get_data(url)
#     show_per_periode(df, "De Bilt", ["temp_avg"], "maandgem", "plotly")
if __name__ == "__main__":
    main()
    