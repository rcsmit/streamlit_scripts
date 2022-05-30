# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:34:44 2020

@author: rcxsm
"""

from socketserver import DatagramRequestHandler
import pandas as pd
import numpy as np
#import openpyxl
import streamlit as st
import datetime as dt
from datetime import datetime, timedelta
from datetime import date
 
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
import traceback

#import datetime

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
        #file = "C:\\Users\\rcxsm\\Documents\\xls\\masterfinance_.xlsm"  
        file = r"C:\Users\rcxsm\Documents\xls\kasboeken_azie.xlsx"
        file = r"C:\Users\rcxsm\Documents\wereknd 28052022 1247a1753.xlsx"
        sheet = sheet_id
        try:
            df_x = pd.read_excel (file,
                                sheet_name= sheet,
                                header=0,
                                usecols= "a,b, e,g, h,r",
                                names=["id","country_year", "datum","in_uit", "rubriek","bedrag"],)


            #df["datum"] = pd.to_datetime(df["datum"], format="%Y-%m-%d")
            df_x.datum=pd.to_datetime(df_x.datum,errors='coerce', dayfirst=True)
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
    df = df_x[(df_x["country_year"] != "TRANSIT2020") & (df_x["in_uit"] != "IN") & (df_x["rubriek"]!= "TODO") & (df_x["rubriek"]!= "TINA_ISA") & (df_x["rubriek"]!= "MISC__")& (df_x["rubriek"]!= "KRUIS")& (df_x["rubriek"]!= "KRUIS_UIT") & (df_x["rubriek"]!= "KRUIS_DERDEN")& (df_x["rubriek"]!= "PLANE")& (df_x["rubriek"]!= "BANK_KRUIS")]
    df['jaar']=df['datum'].dt.strftime('%Y')
    df['maand']=df['datum'].dt.strftime('%m')
    df['invbedrag']= df['bedrag']* -1
    df['maand_']=df['datum'].dt.strftime('%Y-%m')
    lijst_met_country_year  =  df["country_year"].drop_duplicates().sort_values().tolist()
    
    def numOfDays(date1, date2):
        return (date2-date1).days
    lijst0 =[]
    for c_y in lijst_met_country_year:
        dfcy= df[df["country_year"] == c_y]
        min_ = min(dfcy['datum'])
        max_ = max(dfcy['datum'])
        #st.write(f'{c_y}  {min_.strftime("%d/%m/%Y")}  {max_.strftime("%d/%m/%Y")} {numOfDays(min_, max_)+1}')
        lijst0.append([c_y,min_.strftime("%d/%m/%Y"),max_.strftime("%d/%m/%Y"),numOfDays(min_, max_)+1])

    df_country_year_days_ = pd.DataFrame(lijst0, columns =['country_year', 'start', 'eind', 'aantal_dagen'])
    totaal_dagen = df_country_year_days_["aantal_dagen"].sum()
 
    df1 =  pd.DataFrame({'country_year': ['Total'], 'aantal_dagen': [totaal_dagen]})
   

    data = [df_country_year_days_, df1]
    df_country_year_days = pd.concat(data).reset_index(drop=True)
    #df_country_year_days = df_country_year_days.append(dict, ignore_index = True)
  
    #st.write(df_country_year_days)


    return df, df_country_year_days

def totalen_per_rub(df):
    st.header(f"Totalen per rubriek over de hele periode")
    rapport = df.groupby(["rubriek"])["bedrag"].sum()
    st.write(rapport)

def save_df(df, name):
    """  _ _ _ """
    OUTPUT_DIR = ""
    name_ = OUTPUT_DIR + name + ".csv"
    compression_opts = dict(method=None, archive_name=name_)
    df.to_csv(name_, index=False, compression=compression_opts)

    print("--- Saving " + name_ + " ---")

def uitgaven_categorie_per_country_year(df, df_country_year_days, period,modus):
    st.header(f"UITGAVES PER country_year")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m-%d')
    table_uitg = df

    table_uitg_pivot = pd.pivot_table(table_uitg, values='bedrag', index=["country_year"],
            columns=[modus], aggfunc=np.sum,  margins = False, fill_value=0).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_with_sum = pd.pivot_table(table_uitg, values='bedrag', index=["country_year"],
            columns=[modus], aggfunc=np.sum,  margins = True, margins_name='Total', fill_value=0).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_only_sum = table_uitg_pivot_with_sum[["country_year","Total"]]

    numbers_df_country_year_days = df_country_year_days[["aantal_dagen"]]
    # st.write(table_uitg_pivot_with_sum.values[:,0])
    # st.write(df_country_year_days.values[:,3])
    #table_country_year_per_dag = table_uitg_pivot_with_sum.values[:,1:]/df_country_year_days.values[:,3]
    table_uitg_pivot_with_sum_only_num = table_uitg_pivot_with_sum._get_numeric_data() 
    table_country_year_per_dag = table_uitg_pivot_with_sum_only_num.div(df_country_year_days.iloc[:,3], axis='rows').round(2)



    #df5=  df1.div(df3.iloc[:,0], axis='rows')
    columnlist =  table_uitg_pivot.columns.tolist()
    columnlist = columnlist[1:]
    fig = px.bar( table_uitg_pivot , x="country_year", y=columnlist, barmode = 'stack', title=f"Uitgaven per country_year (euro)")
    st.plotly_chart(fig)
    
    frames = [df_country_year_days, table_country_year_per_dag ]
    result = pd.concat(frames, axis=1)
    st.write(result)
    result1 = result.iloc[:-1 , :]
    result_without_total = result1.iloc[:, :-1 ]

    table_w =  table_uitg_pivot_with_sum.astype(str)
 
    
    table_w4 = result.astype(str)
    st.subheader("pivottable country year")
    st.write(table_w)

    st.subheader("Per countryyear per dag")
   
    st.write(table_w4)
    columnlist =  result_without_total.columns.tolist()
    columnlist = columnlist[4:]
    fig = px.bar( result_without_total , x="country_year", y=columnlist, barmode = 'stack', title=f"Uitgaven per dag (euro)")
    st.plotly_chart(fig)


    #save_df(table_w2, "table_uitg_only_sum")

def uitgaven_per_period(df, period,modus):
    st.header(f"UITGAVES PER {period}")
    df.datum=pd.to_datetime(df.datum)
    df.datum=df['datum'].dt.strftime('%Y-%m-%d')
    table_uitg = df # df[(df["in_uit"] == "UIT" )| (df["in_uit"] == "UIT_AZIE") | (df["in_uit"] == "_AZIE_VOORAF")]
    #st.write(table_uitg)
    #table_uitg["bedrag"] = table_uitg["bedrag"]*-1

    table_uitg_pivot = pd.pivot_table(table_uitg, values='bedrag', index=[period],
            columns=[modus], aggfunc=np.sum,  margins = False, fill_value=0).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_with_sum = pd.pivot_table(table_uitg, values='bedrag', index=[period],
            columns=[modus], aggfunc=np.sum,  margins = True, margins_name='Total', fill_value=0).round().reset_index()  #.plot.bar(stacked=True)
    table_uitg_pivot_only_sum = table_uitg_pivot_with_sum[[period,"Total"]]
    columnlist =  table_uitg_pivot.columns.tolist()
    columnlist = columnlist[1:]
    fig = px.bar( table_uitg_pivot , x=period, y=columnlist, barmode = 'stack', title=f"Uitgaven per {period} (euro)")
    st.plotly_chart(fig)

    table_w =  table_uitg_pivot_with_sum.astype(str)
    table_w2 =  table_uitg_pivot_only_sum.astype(str)
    
    st.subheader(f"pivottable per {period}")
    st.write(table_w)
    st.write(table_w2)
    #save_df(table_w2, "table_uitg_only_sum")

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
    start_ = "2017-01-01"
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
    df, df_country_year_days = read(sheet_id)
    FROM, UNTIL = interface_selectperiod()

    df = select_period_oud(df, "datum", FROM, UNTIL)
    # lijst_met_hoofdrubrieken  =  df['hoofdrub'].drop_duplicates().sort_values().tolist()
    lijst_met_rubrieken  =  df['rubriek'].drop_duplicates().sort_values().tolist()
    period =  st.sidebar.selectbox("Period",["jaar", "maand_"], index=1)
    #modus_ =  st.sidebar.selectbox("Modus",["hoofdrubriek", "rubriek"], index=1)
    modus_ = 'rubriek'
    if modus_ == "hoofdrubriek":
        modus = 'hoofdrub'
        # rubriek =  st.sidebar.selectbox("Hoofdrubriek",lijst_met_hoofdrubrieken, index=3)
        rubriek = "Foo"
    else:
        modus = 'rubriek'
        rubriek =  st.sidebar.selectbox("Rubriek",lijst_met_rubrieken, index=16)
  
    uitgaven_per_period(df, period, modus)
    uitgaven_categorie_per_country_year(df, df_country_year_days, period,modus)
    uitgaves_categorie_per_period(df,rubriek,modus, period)
    totalen_per_rub(df)

if __name__ == "__main__":
    main()