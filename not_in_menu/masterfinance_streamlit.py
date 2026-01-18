# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:34:44 2020

@author: rcxsm
"""

# C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\not_in_menu\edit_master_finance.py

import pandas as pd
import numpy as np
#import openpyxl
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import traceback
from math import sin,cos,pi
  

def select_period_oud(df, field, show_from, show_until, years):
    """Shows two inputfields (from/until and Select a period in a df.
    Args:
        df (df): dataframe
        field (string): Field containing the date
    Returns:
        df: filtered dataframe
    """

    if show_from is None:
        show_from = "2021-1-1"

    if show_until is None:
        show_until = "2030-1-1"
    #"Date_statistics"
    mask = (df[field].dt.date >= show_from) & (df[field].dt.date <= show_until)


    df = df.loc[mask]
    df = df.reset_index()
    
    # TODO: alleen de jaren laten zien die in de datumrange vallen

    mask = (df['jaar'].isin(years))
    df = df.loc[mask]
    df = df.reset_index()


    return df



def barchart (table,hr, titel, period):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=table[period], y= table[hr]  ))

    fig.update_layout(

        yaxis=dict(
            title=f"{hr} (€)",
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        title=dict(
                text=titel,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )
            )
    )
    st.plotly_chart(fig)

def barchart_in_uit(table, titel, period):
    months = table[period].to_list() 
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=table["IN"],
                    base=0,
                    marker_color='lightslategrey',
                    name='inkomsten'
                    ))
    fig.add_trace(go.Bar(x=months, y=table["UIT_TOT"],
                    base=0,
                    marker_color='crimson',
                    name='uitgaven'))

    fig.update_layout(
        yaxis=dict(
            title=f"(€)",
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        title=dict(
                text=titel,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )
            )
    )
    st.plotly_chart(fig)

def f(x):
    """convert a python datetime.datetime to excel serial date number

    Args:
        x ([type]): python datetime.datetime

    Returns:
        [type]: excel serial date number
    """
    #https://stackoverflow.com/a/9574948/2901002
    d = pd.Timestamp(1899, 12, 30)
    timedeltas = pd.to_timedelta(x, unit='d', errors='coerce')
    dates = pd.to_datetime(x, errors='coerce')
    return (timedeltas + d).fillna(dates)


def read(sheet_id):
    #filetype = 'google_sheet'
    filetype = 'xls'
    #file = 'C:\Users\rcxsm\Documents\pyhton_scripts'

    if filetype == 'csv':
        try:
            df = pd.read_csv(
                "masterfinance.csv",
                names=["id","bron","datum","bedrag",
                       "tegenpartij","hoofdrub","rubriek"],

                dtype={
                    "bron": "category",
                    "hoofdrub": "category",
                    "rubriek": "category",

                },
                delimiter=';',
                parse_dates=["datum"],
                encoding='latin-1'  ,
                dayfirst=True
            )
        except:
            st.warning("error met laden")
            st.stop()
    elif filetype == 'xls':
        file = "C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_2023.xlsx"   
        sheet = sheet_id
        try:
            df = pd.read_excel (file,
                                sheet_name= sheet,
                                header=0,
                                usecols= "a,b,g,h,l,m,n,o,p",
                                names=["id","bron","datum","bedrag",
                       "tegenpartij","in_uit","hoofdrub","rubriek", "rubriek_azie"],)
            #df["datum"] = pd.to_datetime(df["datum"], format="%Y-%m-%d")
            df.datum=pd.to_datetime(df.datum,errors='coerce', dayfirst=True)
            
        except Exception as e:
            st.warning("error met laden")
            st.warning(f"{e}")
            st.warning(traceback.format_exc())
            st.stop()

    elif filetype =='google_sheet':
        sheet_name = "gegevens"
        url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        df = pd.read_csv(url, delimiter=',')
        st.write(df.dtypes)
        df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")
        # df = df[:-1]  #remove last row which appears to be a Nan
    else:
        st.write("type doesnt exist")
        pass
    df['jaar']=df['datum'].dt.strftime('%Y')
    # Ensure the 'jaar' column is of integer type
    df['jaar'] = df['jaar'].fillna(0).astype(int)

    df['maand']=df['datum'].dt.strftime('%m')
    df['invbedrag']= df['bedrag']* -1
    df['maand_']=df['datum'].dt.strftime('%Y-%m')
    df['jaar_']=df['datum'].dt.strftime('%Y')
    df = df[~((df["rubriek"] == "STARTING_BALANCE") & ((df["jaar"] == "2022") | (df["jaar"] == "2021")))]
    return df

def totalen_per_rub(df):
    st.header(f"Totalen per rubriek over de hele periode")
    rapport = df.groupby(["hoofdrub", "rubriek"])["bedrag"].sum()
    st.write(rapport)

def save_df(df, name):
    """  _ _ _ """
    OUTPUT_DIR = (
        "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\"
    )


    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")


def in_and_out_per_period(df, period):
    st.header(f"IN EN UIT PER {period}")
    table = pd.pivot_table(df, values='bedrag', index=[period],
            columns=['in_uit'],  aggfunc='sum',fill_value=0).reset_index()
    

    # for index, row in table.iterrows():
    #     table.at[index,"UIT_TOT"] = row.UIT + row.UIT_VL + row.UIT_AZIE + row.UIT_AZIE_VOORAF
    #     table.at[index,"verschil"] = row.IN + row.UIT + row.UIT_VL + row.UIT_AZIE + row.UIT_AZIE_VOORAF

    for c in ["UIT_AZIE_VOORAF","UIT_AZIE","UIT_VL","UIT", "IN"]: 
        if c not in table.columns:
            table[c] = 0  # or any other default value
            
    table["UIT_TOT"] =  table["UIT"] + table["UIT_VL"] + table["UIT_AZIE"]  + table["UIT_AZIE_VOORAF"]
    table["verschil"] = table["IN"] + table["UIT"] + table["UIT_VL"]+ table["UIT_AZIE"]  + table["UIT_AZIE_VOORAF"]
   

    #save_df(table, "IN_OUT_PER_YEAR")
    table_met_som = pd.pivot_table(df, values='bedrag', index=[period],
            columns=['in_uit'], margins=True, aggfunc='sum',fill_value=0).reset_index()

    barchart_in_uit(table,  f"Inkomsten en uitgaven per {period}", period)
    barchart(table,"verschil", f"Saldo van inkomsten en uitgaven per {period}", period)
    st.write(table_met_som)

def costs_per_day(df):
    # jaar, dagen in NL, dagen in Azie
    days =  [[2017,245,120],
                [2018,245,120],
                [2019,222,143],
                [2020,184,181],
                [2021,318,47],
                [2022,170,195]]

def uitgaven_per_period(df, period,modus):
    st.header(f"UITGAVES PER {period}")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m-%d')
    table_uitg = df[(df["in_uit"] == "UIT" )| (df["in_uit"] == "UIT_AZIE") | (df["in_uit"] == "_AZIE_VOORAF")]
    table_uitg_azie = df[ (df["in_uit"] == "UIT_AZIE")]
    # print (table_uitg.dtypes)
    # for i in range(len(table_uitg)):
    #     #table_uitg.iloc[i,5]= table_uitg.iloc[i,5]*-1
    #     table_uitg.at[i,'bedrag']= table_uitg.at[i,'bedrag']*-1
    # table_uitg['bedrag_'] = table_uitg.apply(lambda x: x['bedrag']*-1, axis=1)
    # table_uitg["bedrag"] = table_uitg["bedrag"]*-1

    # afgeraden!
    # for index, row in table_uitg.iterrows():
    #     df.at[index,'bedrag'] = row.bedrag *-1

    table_uitg_pivot = pd.pivot_table(table_uitg, values='invbedrag', index=[period],
            columns=[modus], aggfunc='sum', fill_value=0, margins=False).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_azie = pd.pivot_table(table_uitg_azie, values='invbedrag', index=[period],
            columns=[modus], aggfunc='sum', fill_value=0, margins=True).round().reset_index()  #.plot.bar(stacked=True)

    columnlist =  table_uitg_pivot.columns.tolist()
    columnlist = columnlist[1:]
    fig = px.bar( table_uitg_pivot , x=period, y=columnlist, barmode = 'stack', title=f"Uitgaven per {period}")
    st.plotly_chart(fig)

    table_w =  table_uitg_pivot.astype(str)
    table_w_azie =  table_uitg_pivot_azie.astype(str)
    st.subheader("pivottable")
    st.write(table_w)
    st.write (" TABLE W AZIE")
    st.write(table_w_azie)
    save_df(table_w_azie, "Table vanuit masterfinance")



def sunburst_chart(df, years_to_show):
    st.header("Sunburst diagram")
    df = df.fillna(0)
    df["jaar"] = df["jaar"].astype(int) 
    df = df.replace("zorgtoeslag", "zorgverzekering")
  
    df_sunburst = df.groupby(["jaar", "in_uit","hoofdrub", "rubriek","rubriek_azie"]).sum(numeric_only=True).reset_index()
    df_sunburst = df_sunburst[(df_sunburst["in_uit"] != "IN") & (df_sunburst["bedrag"] <0 ) &(df_sunburst["in_uit"] != "KRUIS") & (df_sunburst["in_uit"] != "STARTING_BALANCE")& (df_sunburst["in_uit"] != "FICTIEVE_W_V")]
    if (df_sunburst["invbedrag"] < 0).any():
        st.error("Error: There are negative values in the 'invbedrag' column.")
        st.stop()

    divide_by = st.sidebar.number_input("Delen door (99 for given)", 0.1,99.0,99.0)
    if divide_by !=99:
        df_sunburst["invbedrag_divided"] = df_sunburst["invbedrag"] / divide_by
    else:
        print(len(years_to_show)) 
        # Given data MONTHS IN ASIA PER YEAR
        if 1==1:

           
    
              # year, asia, europe
            months_asia = 1.5+4.6+1.75+ 1.6+ 1.5+4.33+ 0
            months_europe = 10.5+(12-(4.6+1.75))+ (12-(1.6+ 1.5))+(12-(4.33+ 1.75))


            # Calculate and apply divide_by for each category and year
            divide_by_values =[
                [ "UIT", months_europe],
                    [ "UIT_VL", months_europe+months_asia ],
                    [ "UIT_AZIE", months_asia],
                    [ "UIT_AZIE_VL", months_asia],
                    [ "UIT_AZIE_VOORAF", months_asia],

                    [ "UIT_AZIE_VLIEGTICKETS", months_asia]
            ]    

            for d in divide_by_values:
            
                mask = (df_sunburst["in_uit"] == d[0])
            
                df_sunburst.loc[mask, "invbedrag_divided"] = df_sunburst.loc[mask, "invbedrag"] / d[1]

    # attention, if there are negative values, the sunburst is not shown
    show_sunburst(years_to_show, df_sunburst, divide_by)
    #show_sunburst_annotated(years_to_show, df_sunburst)
    for y in years_to_show:
        df_sunburst_year = df_sunburst[df_sunburst['jaar'] == y]
        if len(df_sunburst_year) >0 :
             # year, asia, europe
            years_months = [
                (2020, 6,6),
                (2021, 1.5,10.5),
                (2022, 4.6+1.75, 12-(4.6+1.75)),
                (2023, 1.6+ 1.5,12-1.6+ 1.5),
                (2024, 4.33+ 0,12-(4.33+ 1.75)),
                (2025, 5,7)
            ]
                    
            # Calculate and apply divide_by for each category and year
            for y in years_months:
                divide_by_values =[
                    [y[0], "UIT", y[2]],
                    [y[0], "UIT_VL", 12 ],
                    [y[0], "UIT_AZIE", y[1]],
                    [y[0], "UIT_AZIE_VL", y[1]],
                    [y[0], "UIT_AZIE_VOORAF", y[1]],

                    [y[0], "UIT_AZIE_VLIEGTICKETS", y[1]]
                ]    
    
                for d in divide_by_values:
                
                    mask = (df_sunburst_year["jaar"] == y[0]) & (df_sunburst_year["in_uit"] == d[1])
                    #mask = (df_sunburst["in_uit"] == d[1])
                
                
                    df_sunburst_year.loc[mask, "invbedrag_divided"] = df_sunburst_year.loc[mask, "invbedrag"] / d[2]
            show_sunburst(y, df_sunburst_year, divide_by)
            #show_sunburst_annotated(y, df_sunburst_year, divide_by)
        else:
            st.warning(f"No items for year {y}")

def show_sunburst(years_to_show, df_sunburst, divide_by):
    st.subheader(f" -- {years_to_show} ---")
    
    st.write(df_sunburst)
    
    fig = px.sunburst(df_sunburst,
                  path=["in_uit","hoofdrub", "rubriek","rubriek_azie"],
                  values="invbedrag_divided",
                  title=f"Uitgaven {years_to_show}",width=1000, height=1000,) 
  
    ## set marker colors whose labels are " " to transparent
    # https://stackoverflow.com/questions/71442845/how-to-avoid-none-when-plotting-sunburst-chart-using-plotly
    try:
        marker_colors = list(fig.data[0].marker['colors'])
        marker_labels = list(fig.data[0]['labels'])
        new_marker_colors = ["rgba(0,0,0,0)" if label=="xxx" else color for (color, label) in zip(marker_colors, marker_labels)]
        marker_colors = new_marker_colors

        fig.data[0].marker['colors'] = marker_colors
    except Exception as e:
        print (f"Error {e} {years_to_show}")
    st.plotly_chart(fig)

def show_sunburst_annotated(years_to_show, df, divide_by):
    # https://stackoverflow.com/questions/70129355/value-annotations-around-plotly-sunburst-diagram
    st.subheader(f" -- {years_to_show} ---")
    st.write (df)
    df["invbedrag_divided"] = df["invbedrag"] / divide_by
    aaa = 'in_uit'
    bbb = 'hoofdrub'
    ccc = 'rubriek'
    ddd = "invbedrag_divided"

    fig = px.sunburst(df, path=[aaa, bbb, ccc], values=ddd, width=600, height=600, title=f"Uitgaven {years_to_show}",)
    totals_groupby =  df.groupby([aaa, bbb, ccc]).sum()
    totals_groupby["aaa_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=aaa).transform('sum')
    totals_groupby["aaa_bbb_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb]).transform('sum')
    totals_groupby["aaa_bbb_ccc_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb,ccc]).transform('sum')
    totals_groupby = totals_groupby.sort_values(by=["aaa_sum","aaa_bbb_sum","aaa_bbb_ccc_sum"], ascending=[0,0,0])
    #st.write(totals_groupby)

   
    ## calculate the angle subtended by each category
    sum_ddd = getattr(df,ddd).sum()
    delta_angles = 360*totals_groupby[ddd] / sum_ddd
   
    # for i in range(len(totals_groupby)):
    #     #st.write(totals_groupby.iat[i,4])
    #     if totals_groupby.iat[i,4] < sum_ddd/20:
    #         totals_groupby.iat[i,4] = None

    annotations = [format(v,".0f") for v in  getattr(totals_groupby,ddd).values]
    
    ## calculate cumulative sum starting from 0, then take a rolling mean 
    ## to get the angle where the annotations should go
    angles_in_degrees = pd.concat([pd.DataFrame(data=[0]),delta_angles]).cumsum().rolling(window=2).mean().dropna().values

    def get_xy_coordinates(angles_in_degrees, r=1):
        return [r*cos(angle*pi/180) for angle in angles_in_degrees], [r*sin(angle*pi/180) for angle in angles_in_degrees]
        #return [r*cos(angle) for angle in angles_in_degrees], [r*sin(angle) for angle in angles_in_degrees]

    x_coordinates, y_coordinates = get_xy_coordinates(angles_in_degrees, r=1.5)
    fig.add_trace(go.Scatter(
        x=x_coordinates,
        y=y_coordinates,
        mode="text",
        text=annotations,
        hoverinfo="skip",
        textfont=dict(size=14)
    ))

    padding = 0.50
    fig.update_layout( margin=dict(l=20, r=20, t=20, b=20),
        
        xaxis=dict(
            range=[-1 - padding, 1 + padding], 
            showticklabels=False
        ), 
        yaxis=dict(
            range=[-1 - padding, 1 + padding],
            showticklabels=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
       
    )
    st.plotly_chart(fig)




def save_df(df, name):
    """  _ _ _ """
    OUTPUT_DIR = ""
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")

def uitgaves_categorie_per_period(df,hr, modus, period):
    df = df.fillna(0)

    st.header(f"UITGEGEVEN per {period} aan {hr}")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m')

    df_gt_hr = df.loc[df[modus] ==hr].copy(deep=False)

    table = df_gt_hr.groupby([period]).sum(numeric_only=True).reset_index()
    table = table[[period, "invbedrag"]]
    barchart(table,"invbedrag", f"Uitgegeven aan {hr}", period)
    lengte= len(table)
    som = table["invbedrag"].sum()
    avg = round(som/ lengte ,2)

    table = table.astype(str)

    st.write(table)
    st.write (f"Gemiddeld per {period} : € {avg}")
    with st.expander("Show items", expanded = False):
        st.write(df_gt_hr)


def interface_selectperiod():
    DATE_FORMAT = "%m/%d/%Y"
    start_ = "2016-01-01"
    today = datetime.date.today().strftime("%Y-%m-%d")
    from_ = st.sidebar.text_input("startdate (yyyy-mm-dd)", start_)

    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    until_ = st.sidebar.text_input("enddate (yyyy-mm-dd)", today)

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    return FROM, UNTIL

def pivot_tables(df):

    st.subheader("Bedragen per rubriek per maand/jaaar")
    # Zorg ervoor dat de datumkolom het juiste type heeft
    df['datum'] = pd.to_datetime(df['datum'])
    df=df[df['hoofdrub'] !='STARTING_BALANCE']
    df['bedrag'] = np.where(df['in_uit'] != 'IN', df['bedrag'] * -1, df['bedrag'])
    in_uit_totalen = df.groupby(['in_uit'])['bedrag'].sum().reset_index()
    in_uit_totalen.rename(columns={'bedrag': 'Totaal in_uit'}, inplace=True)


    # Totaal bedrag per hoofdrub
    hoofdrub_totalen = df.groupby(['in_uit','hoofdrub'])['bedrag'].sum().reset_index()
    hoofdrub_totalen.rename(columns={'bedrag': 'Totaal per hoofdrub'}, inplace=True)

    # Totaal bedrag per rubriek
    rubriek_totalen = df.groupby(['in_uit','hoofdrub','rubriek'])['bedrag'].sum().reset_index()
    rubriek_totalen.rename(columns={'bedrag': 'Totaal per rubriek'}, inplace=True)

    
    # Totaalsaldo
    saldo = df['bedrag'].sum()

    # Resultaten weergeven
    st.write("Totaal per in/uit:")
    st.write(in_uit_totalen)
    st.write("Totaal per rubriek:")
    st.write(rubriek_totalen)
    st.write("\nTotaal per hoofdrub:")
    st.write(hoofdrub_totalen)
    st.write("\nEindsaldo:")
    st.write(f"{saldo:.2f}")
    for tijdsperiode in ["jaar_", "maand_"]:
            
        for index in [['in_uit'],['in_uit', 'hoofdrub',],['in_uit', 'hoofdrub', 'rubriek']]:

         
            # Totaal bedrag per in_uit, hoofdrub, rubriek, en maand_
            pivot_df = df.pivot_table(
                values='bedrag',
                index=index,
                columns=tijdsperiode,
                aggfunc='sum',
                fill_value=0
            )

            # Sorteren van de kolommen (maanden) op volgorde van links naar rechts
            pivot_df = pivot_df.sort_index(axis=1)

            # Resultaten weergeven
            st.write(pivot_df)
            make_graph_pivot(pivot_df, tijdsperiode, index)



    df_sorted = df.sort_values(by=['in_uit', 'hoofdrub', 'rubriek','datum'])


# Print de gesorteerde dataframe
    st.write(df_sorted)

def make_graph_pivot(pivot_df, tijdsperiode, index):
    
        # Create a new figure
        fig = go.Figure()

        # Loop over each row in the pivot table and add it as a trace
        for index, row in pivot_df.iterrows():
            try:
                in_uit, hoofdrub, rubriek = index  # Extract index values
            except:
                try:
                    in_uit, hoofdrub = index  # Extract index values
                    rubriek = ""
                except:
                    in_uit = index  # Extract index values
                    hoofdrub, rubriek = "",""

            trace_name = f"{in_uit} - {hoofdrub} - {rubriek}"  # Name of the trace
            
            fig.add_trace(
                go.Scatter(
                    x=pivot_df.columns,  # x-axis: months
                    y=row,               # y-axis: values for this row
                    mode='lines',
                    name=trace_name      # Unique name for each line
                )
            )

        # Update layout for better readability
        fig.update_layout(
            title=f"Line Graph of Bedrag per {tijdsperiode} - {index}",
            xaxis_title=f"{tijdsperiode}",
            yaxis_title="Bedrag",
            legend_title="Categorieën",
            template="plotly_white"
        )

        # Show the plot
        st.plotly_chart(fig)


def bereken_balanstotaal_per_maand(df):  
    st.subheader("Balanstotaal per maand")   
    # Zorg ervoor dat 'datum' een datetime-type is
   
    # Sorteren van de dataframe op 'bron' en 'datum' om het lopend totaal correct te berekenen
    df = df.sort_values(by=['bron', 'datum'])

    # Bereken het lopend totaal per 'bron'
    df['lopend_totaal'] = df.groupby('bron')['bedrag'].cumsum()
    for periode in ["maand","jaar"]:
        if periode == "maand":
            # Voeg een kolom toe voor de laatste dag van de maand
            df['laatste_dag_periode'] = df['datum'] + pd.offsets.MonthEnd(0)
        else:
            df['laatste_dag_periode'] = df['datum'] + pd.offsets.YearEnd(0)
        df['laatste_dag_periode'] = df['laatste_dag_periode'].dt.normalize()
        
        pivot_df = df.pivot_table(
            values='lopend_totaal',
            index='bron',
            columns='laatste_dag_periode',
            aggfunc='last',
            # margins=True,  # Voeg totalen toe
            # margins_name='Totaal'  # Naam van de totalenrij
        )

        # Vul NaN-waarden in met de laatste bekende waarde in de rij
        pivot_df = pivot_df.ffill(axis=1)
        
        
        # Bereken de totalen per kolom
        column_totals = pivot_df.sum(axis=0)

        # Voeg de totalen toe als een nieuwe rij
        pivot_df.loc['Totaal'] = column_totals

        # Formatteer de kolommen naar 'yyyy-mm'
        pivot_df.columns = pivot_df.columns.strftime('%Y-%m')
        st.write(pivot_df)
        # Maak een nieuwe Plotly-figuur
        fig = go.Figure()

        # Voeg voor elke bron een lijn toe aan de grafiek
        for bron in pivot_df.index:
            fig.add_trace(
                go.Scatter(
                    x=pivot_df.columns,  # x-waarden: de laatste dagen van de maand
                    y=pivot_df.loc[bron],  # y-waarden: de lopende totalen voor de bron
                    mode='lines',
                    name=bron  # Naam van de lijn
                )
            )


        # Layout aanpassen voor betere leesbaarheid
        fig.update_layout(
            title="Lopend Totaal per Bron",
            xaxis_title=f"Laatste dag van de {periode}",
            yaxis_title="Lopend Totaal",
            legend_title="Bron",
            template="plotly_white"
        )

        # Toon de grafiek
        st.plotly_chart(fig)

def give_totals_boodschappen(df):
  
    # Assuming 'df' is your DataFrame
    # First, ensure that the 'Bedrag' column is of numeric data type (e.g., float or int)
    # If it's not already, you can convert it like this:
    # df['Bedrag'] = pd.to_numeric(df['Bedrag'], errors='coerce')

    # Filter rows where 'category' is equal to 'boodschappen'
    st.subheader("Boodschappen")
    #boodschappen_df = df[df['rubriek'] == 'boodschappen_sevenum']
    boodschappen_df = df[df['hoofdrub'] == 'BOODSCH']
    st.write(boodschappen_df)
    # Extract the month and year from the datetime column (assuming it's named 'date')
    boodschappen_df['Month'] = boodschappen_df['datum'].dt.strftime('%Y-%m')
    boodschappen_df['Year'] = boodschappen_df['datum'].dt.strftime('%Y')

    # Group by month and calculate the sum of 'Bedrag'
    monthly_result = boodschappen_df.groupby('Month')['bedrag'].sum().reset_index()
    monthly_result.rename(columns={'bedrag': 'Total_Bedrag_Month'}, inplace=True)

    # Group by year and calculate the sum of 'Bedrag'
    yearly_result = boodschappen_df.groupby('Year')['bedrag'].sum().reset_index()
    yearly_result.rename(columns={'bedrag': 'Total_Bedrag_Year'}, inplace=True)

    # Print the resulting DataFrames
    st.write("Monthly Total:")
    st.write(monthly_result)

    st.write("\nYearly Total:")
    st.write(yearly_result)

    # Define the number of months for each year
    months_per_year = {
        '2021': 6.0,
        '2022': 3.8,
        '2023': 7.5,
        '2024':2.5
    }
    yearly_totals = boodschappen_df.groupby('Year')['bedrag'].sum()
    # Calculate the average monthly spending for each year
    average_monthly_spending = yearly_totals / pd.Series(months_per_year)

    # Print the result
    st.write("Average Monthly Spending:")
    st.write(average_monthly_spending)

def main():
    st.header("Financial sheet Rene")
    # Use 2 decimal places in output display
    pd.options.display.float_format = '{:.2f}'.format
    # Don't wrap repr(DataFrame) across additional lines
    pd.set_option("display.expand_frame_repr", False)

  
    sheet_id = "INVOER"
    df = read(sheet_id)
    # Sidebar menu
    option = st.sidebar.radio(
        "Select a function to run:",
        [
            "in_and_out_per_period",
            "sunburst_chart",
            "uitgaven_per_period",
            "uitgaves_categorie_per_period",
            "totalen_per_rub",
            "pivot_tables",
            "bereken_balanstotaal_per_maand",
            "boodschappen"
        ]
    )

    FROM, UNTIL = interface_selectperiod()
    
    years_possible   =  df['jaar'].drop_duplicates().sort_values().tolist()
    years = st.sidebar.multiselect("Jaren",years_possible, years_possible)

    df = select_period_oud(df, "datum", FROM, UNTIL, years).copy()
   
    lijst_met_hoofdrubrieken  =  df['hoofdrub'].drop_duplicates().sort_values().tolist()
    lijst_met_rubrieken  =  df['rubriek'].drop_duplicates().sort_values().tolist()
    if option in ["in_and_out_per_period","uitgaven_per_period","uitgaves_categorie_per_period"]:
        period =  st.sidebar.selectbox("Period",["jaar", "maand_"], index=1)
    if option in ["uitgaven_per_period","uitgaves_categorie_per_period"]:
        modus_ =  st.sidebar.selectbox("Modus",["hoofdrubriek", "rubriek"], index=0)

        if modus_ == "hoofdrubriek":
            modus = 'hoofdrub'
            rubriek =  st.sidebar.selectbox("Hoofdrubriek",lijst_met_hoofdrubrieken, index=3)
        else:
            modus = 'rubriek'
            rubriek =  st.sidebar.selectbox("Rubriek",lijst_met_rubrieken, index=16)
    
    
    # Call the selected function
    if option == "in_and_out_per_period":
        in_and_out_per_period(df, period)
    elif option == "uitgaven_per_period":
        uitgaven_per_period(df, period, modus)
    elif option == "uitgaves_categorie_per_period":
        uitgaves_categorie_per_period(df, rubriek, modus, period)
    elif option == "sunburst_chart":
        sunburst_chart(df, years)
    
    elif option == "totalen_per_rub":
        totalen_per_rub(df)
    elif option == "pivot_tables":
        pivot_tables(df)
    elif option == "bereken_balanstotaal_per_maand":
        bereken_balanstotaal_per_maand(df)
    elif option =="boodschappen":
        give_totals_boodschappen(df)
if __name__ == "__main__":
    import os
    import datetime
    os.system('cls')

    print(f"--------------{datetime.datetime.now()}-------------------------")

    
    main()