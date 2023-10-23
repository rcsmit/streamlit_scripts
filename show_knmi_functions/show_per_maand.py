import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from show_knmi_functions.utils import get_data

def show_per_maand(df, gekozen_weerstation, what_to_show_, groeperen, graph_type):
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    df.set_index("YYYYMMDD")
    (month_min,month_max) = st.sidebar.slider("Maanden (van/tot en met)", 1, 12, (1,12))

    jaren = df["YYYY"].tolist()
    df = df[(df["MM"] >= month_min) & (df["MM"] <= month_max)]
    df['DD'] = df['DD'].astype(str).str.zfill(2)
    df['MM'] = df['MM'].astype(str).str.zfill(2)
    df["mmdd"] = df["MM"] +"-" + df["DD"]

    df["year"] = df["year"].astype(str)
            
    for what_to_show in what_to_show_:
        if groeperen == "maandgem":
            df_grouped = df.groupby(["year", "MM"]).mean(numeric_only = True).reset_index()
     
            df_grouped ["year"] = df_grouped ["year"].astype(str)
     
            df_pivoted = df_grouped.pivot(
                index="MM", columns="year", values=what_to_show
            ).reset_index()
        elif groeperen == "per_dag":
            df["MD"] = df["month_day"]
            df_grouped = df.groupby(["year", "mmdd"]).mean(numeric_only = True).reset_index()
            df_grouped ["year"] = df_grouped ["year"].astype(str)
            df_pivoted = df_grouped.pivot(                index="mmdd", columns="year", values=what_to_show            ).reset_index()
        
    
        if graph_type == "plotly":
            fig = go.Figure()
            #df["sma"] = df_pivoted[what_to_show].rolling(window=wdw, center=centersmooth).mean()
            #st.write(df_pivoted.columns)
            # for c in df_pivoted.columns:
            #     print (c)
            #     print ("linne 224")
            #     print (pd.Series(df_pivoted.index.values))
            if groeperen == "maandgem":
                sma = [go.Scatter(x=pd.Series(df_pivoted.MM), y=df_pivoted[c],  
                   mode='lines', name=f'{c}')
                   for c in df_pivoted.columns[1:]]
            else:
                # sma = [go.Scatter(x=[pd.Series(df_pivoted.index.values),df_pivoted.mmdd], y=df_pivoted[c],  
                #    mode='lines',  line=dict(width=.7), name=f'{c}')
                #     for c in df_pivoted.columns[1:]]
        

                        # create the traces
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
                # xaxis=dict(label=df_pivoted["mm-dd"]),
                yaxis=dict(title=what_to_show), 
                title=title,)
                #, xaxis=dict(tickformat="%d-%m")
            fig = go.Figure(data=data, layout=layout)
            #fig.update_layout(xaxis=dict(tickformat="%m-%d"))
            st.plotly_chart(fig, use_container_width=True)
        
        
        
        else:
            st.warning ("Under construction")

            # fig, ax = plt.subplots()
            # plt.title(f"{ what_to_show} - gemiddeld per maand in {gekozen_weerstation}")

           
            # if groeperen == "per_dag":
            #     major_format = mdates.DateFormatter("%b")
            #     ax.xaxis.set_major_formatter(major_format)
            # plt.grid()
            # ax.plot(df_pivoted)
            # plt.legend(df_pivoted.columns, title=df_pivoted.columns.name)

            # st.pyplot(fig)
            # st.subheader(f"Data of {what_to_show}")
            # st.write(df_pivoted)

        
def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    
if __name__ == "__main__":
    # main()
    print ("")