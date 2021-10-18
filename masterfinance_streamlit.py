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


def select_period_oud(df, field, show_from, show_until):
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
    return df


def save_df(df, name):
    """  Saves the df """
    name_ =  name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)
    print("--- Saving " + name_ + " ---")

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
    years = table[period].to_list()
    table["UIT_INV"] = table["UIT"]*-1
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years, y=table["IN"],
                    base=0,
                    marker_color='lightslategrey',
                    name='inkomsten'
                    ))
    fig.add_trace(go.Bar(x=years, y=table["UIT_INV"],
                    base=table["UIT"],
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
    filetype = 'google_sheet'
    filetype = 'xls'
    #file = 'C:\Users\rcxsm\Documents\phyton_scripts'

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
        file = "C:\\Users\\rcxsm\\Documents\\xls\\masterfinance.xlsm"
        sheet = "INVOER"
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
        #     pass
        #df['bedrag'] = df['bedrag'].str.replace(',', '.').astype(float)

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

    #st.write(df['bedrag2'])
    #df['bedrag'] = float(df['bedrag'])

    #df['datum'] = df['datum'].apply(f)



    df['jaar']=df['datum'].dt.strftime('%Y')
    df['maand']=df['datum'].dt.strftime('%m')
    df['invbedrag']= df['bedrag']* -1

    #df.datum=df['datum'].dt.strftime('%Y-%m')
    df['maand_']=df['datum'].dt.strftime('%Y-%m')

    # SHOW COMPLETE DATAFRAME
    #pd.set_option("display.max_rows", None, "display.max_columns", None)
    # st.write("========= DATAFRAME ===========")
    # st.write(df)
    # df.to_csv(r'testdatumpandas.txt', header=None, index=None, sep=' ', mode='a')
    return df


def totalen_per_rub(df):
    st.header(f"Totalen per rubriek over de hele periode")
    rapport = df.groupby(["hoofdrub", "rubriek"])["bedrag"].sum()
    st.write(rapport)


def in_and_out_per_period(df, period):
    st.header(f"IN EN UIT PER {period}")
    table = pd.pivot_table(df, values='bedrag', index=[period],
            columns=['in_uit'],  aggfunc=np.sum,fill_value=0).reset_index()
    table["verschil"] = table["IN"] + table["UIT"]
    table_met_som = pd.pivot_table(df, values='bedrag', index=[period],
            columns=['in_uit'], margins=True, aggfunc=np.sum,fill_value=0).reset_index()

    barchart_in_uit(table,  f"Inkomsten en uitgaven per {period}", period)
    barchart(table,"verschil", f"Saldo van inkomsten en uitgaven per {period}", period)
    st.write(table_met_som)

def uitgaven_per_period(df, period,modus):
    st.header(f"UITGAVES PER {period}")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m-%d')
    table_uitg = df.loc[df["in_uit"] == "UIT"]
    table_uitg["bedrag"] = table_uitg["bedrag"]*-1

    table_uitg_pivot = pd.pivot_table(table_uitg, values='bedrag', index=[period],
            columns=[modus], aggfunc=np.sum, fill_value=0, margins=False).round().reset_index()  #.plot.bar(stacked=True)

    columnlist =  table_uitg_pivot.columns.tolist()
    columnlist = columnlist[1:]
    fig = px.bar( table_uitg_pivot , x=period, y=columnlist, barmode = 'stack', title=f"Uitgaven per {period}")
    st.plotly_chart(fig)

    table_w =  table_uitg_pivot.astype(str)
    st.subheader("pivottable")
    st.write(table_w)



def uitgaves_categorie_per_period(df,hr, modus, period):
    df = df.fillna(0)

    st.header(f"UITGEGEVEN per {period} aan {hr}")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m')

    df_gt_hr = df.loc[df[modus] ==hr].copy(deep=False)

    table = df_gt_hr.groupby([period]).sum().reset_index()
    # table = pd.pivot_table(df_gt_hr, values='invbedrag', index=['maand_'],
    #         columns=[xxx], aggfunc=np.sum,fill_value='', margins=False).reset_index()
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
    start_ = "2021-01-01"
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

    password_ok = st.secrets["finance_password"]
    password_input = st.sidebar.text_input("password", "WvN122334455",  type="password")
    if password_input == password_ok:
        sheet_id = st.secrets["finance_sheet_id"]
        df = read(sheet_id)

    else:
        st.warning ("Enter the right password to enter")
        st.stop()
    FROM, UNTIL = interface_selectperiod()

    df = select_period_oud(df, "datum", FROM, UNTIL)
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
    uitgaven_per_period(df, period, modus)
    uitgaves_categorie_per_period(df,rubriek,modus, period)
    totalen_per_rub(df)




main()
