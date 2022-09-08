# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:34:44 2020

@author: rcxsm
"""

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
        file = "C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_.xlsm"   
        sheet = sheet_id
        try:
            df = pd.read_excel (file,
                                sheet_name= sheet,
                                header=0,
                                usecols= "a,b,f,h,k,l,m,n",
                                names=["id","bron","datum","bedrag",
                       "tegenpartij","in_uit","hoofdrub","rubriek"],)
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
    df['maand']=df['datum'].dt.strftime('%m')
    df['invbedrag']= df['bedrag']* -1
    df['maand_']=df['datum'].dt.strftime('%Y-%m')
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
            columns=['in_uit'],  aggfunc=np.sum,fill_value=0).reset_index()
    st.write(table)

    # for index, row in table.iterrows():
    #     table.at[index,"UIT_TOT"] = row.UIT + row.UIT_VL + row.UIT_AZIE + row.UIT_AZIE_VOORAF
    #     table.at[index,"verschil"] = row.IN + row.UIT + row.UIT_VL + row.UIT_AZIE + row.UIT_AZIE_VOORAF

    table["UIT_TOT"] =  table["UIT"] + table["UIT_VL"] + table["UIT_AZIE"]  + table["UIT_AZIE_VOORAF"]
    table["verschil"] = table["IN"] + table["UIT"] + table["UIT_VL"]+ table["UIT_AZIE"]  + table["UIT_AZIE_VOORAF"]
    st.write(table) 
    #save_df(table, "IN_OUT_PER_YEAR")
    table_met_som = pd.pivot_table(df, values='bedrag', index=[period],
            columns=['in_uit'], margins=True, aggfunc=np.sum,fill_value=0).reset_index()

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
            columns=[modus], aggfunc=np.sum, fill_value=0, margins=False).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_azie = pd.pivot_table(table_uitg_azie, values='invbedrag', index=[period],
            columns=[modus], aggfunc=np.sum, fill_value=0, margins=True).round().reset_index()  #.plot.bar(stacked=True)

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
    # df = df[df["rubriek"] == "kruis"]
    # df = df[df["jaar"] == "2019"]

    # st.write(df)


    df = df.replace("zorgtoeslag", "zorgverzekering")
  
    df_sunburst = df.groupby(["jaar", "in_uit","hoofdrub", "rubriek"]).sum().reset_index()
    df_sunburst = df_sunburst[df_sunburst["in_uit"] != "IN"]
   
    st.write(df_sunburst)
    show_sunburst(years_to_show, df_sunburst)
    for y in years_to_show:
        df_sunburst_year = df_sunburst[df_sunburst['jaar'] == y]
        if len(df_sunburst_year) >0 :
            show_sunburst(y, df_sunburst_year)

def show_sunburst(years_to_show, df_sunburst):
    st.subheader(f" -- {years_to_show} ---")
    fig = px.sunburst(df_sunburst,
                  path=["in_uit","hoofdrub", "rubriek"],
                  values="invbedrag",
                  title=f"Uitgaven {years_to_show}",width=750, height=750,)  
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

    table = df_gt_hr.groupby([period]).sum().reset_index()
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
    today = datetime.today().strftime("%Y-%m-%d")
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

def main():
    st.header("Financial sheet Rene")
    # Use 2 decimal places in output display
    pd.options.display.float_format = '{:.2f}'.format
    # Don't wrap repr(DataFrame) across additional lines
    pd.set_option("display.expand_frame_repr", False)

    # password_ok = st.secrets["finance_password"]
    # password_ok = "password123"
    # password_input = st.sidebar.text_input("password", "password123",  type="password")
    # if password_input == password_ok:
    #     #sheet_id = st.secrets["finance_sheet_id"]
    #     df = read(sheet_id)

    # else:
    #     st.warning ("Enter the right password to enter")
    #    st.stop()
    sheet_id = "INVOER"
    df = read(sheet_id)
    FROM, UNTIL = interface_selectperiod()

    years_possible   =  df['jaar'].drop_duplicates().sort_values().tolist()
    years = st.sidebar.multiselect("Jaren",years_possible, years_possible)

    df = select_period_oud(df, "datum", FROM, UNTIL, years)
    lijst_met_hoofdrubrieken  =  df['hoofdrub'].drop_duplicates().sort_values().tolist()
    lijst_met_rubrieken  =  df['rubriek'].drop_duplicates().sort_values().tolist()
    period =  st.sidebar.selectbox("Period",["jaar", "maand_"], index=1)
    modus_ =  st.sidebar.selectbox("Modus",["hoofdrubriek", "rubriek"], index=0)

    if modus_ == "hoofdrubriek":
        modus = 'hoofdrub'
        rubriek =  st.sidebar.selectbox("Hoofdrubriek",lijst_met_hoofdrubrieken, index=3)
    else:
        modus = 'rubriek'
        rubriek =  st.sidebar.selectbox("Rubriek",lijst_met_rubrieken, index=16)
    in_and_out_per_period(df, period)
    sunburst_chart(df, years)
    uitgaven_per_period(df, period, modus)
    uitgaves_categorie_per_period(df,rubriek,modus, period)
    totalen_per_rub(df)
    

if __name__ == "__main__":
    import os
    import datetime

    print(f"--------------{datetime.datetime.now()}-------------------------")

    os.system('cls')

    main()