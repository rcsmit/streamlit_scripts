import pandas as pd
import streamlit as st
# import numpy as np
import datetime as dt
import plotly.express as px
from get_forex_data_st import * 

def read_csv(url):

    df_ = pd.read_csv(
        url,


        # dtype={
        #     "bron": "category",
        #     "hoofdrub": "category",
        #     "rubriek": "category",

        # },
        delimiter=',',
        parse_dates=["date"],
        encoding='latin-1'  ,
        dayfirst=True
    )
    
    #.set_index('acco_type').stack().rename(columns={'price_per_night':'month'})
    #df_["maand_str"] = df_["maand_int"].astype(str)
    
    

    return df_

def make_frequency_table(df, periode, cat):
    
   
    
    df_test = df.query(f"PERIODE=='{periode}' & CAT=='{cat}' & UITGAVEN !=0")
    average = round(df_test["UITGAVEN"].mean(),1)
    std =  round(df_test["UITGAVEN"].std(),1)
    st.write(f"Frequentietabel {periode} (n={len(df_test)}) - {cat}- mean={average} - std ={std}")
    fig1= px.histogram(data_frame = df_test
            ,x = 'UITGAVEN'
            )
    st.plotly_chart(fig1, use_container_width=True)
  

def make_table(df):

    

    # print (df)
    # #df_test = df[df["PERIODE"] == "BALI2022" or df["CAT"] =="BF"]
    # df_test = df.query('PERIODE=="THB2017" & CAT=="H"')
    # df_test.sort_values(by='date')
    # print (df_test["UITGAVEN_EUR"].mean())
    # print (df_test)
    # print (len(df_test))
  

    list_of_periods = df.PERIODE.unique()
    print(list_of_periods)
    x= []
    for l in list_of_periods:
        df_ = df[df["PERIODE"] == l]
        min = df_["date"].min()
        max = df_["date"].max()

        min_ = min.strftime( "%Y-%m-%d")
        max_ = max.strftime( "%Y-%m-%d")
        x.append([l, min_, max_, (max-min).days])

    df_days = pd.DataFrame(x, columns=["PERIODE", "start", "end","number_of_days"])

    df_pivot = df.pivot_table(index='PERIODE', columns='CAT', values='UITGAVEN_EUR',  aggfunc='sum')
    df_pivot = reduce_columns(df_pivot)

    df_pivot = pd.merge(df_pivot, df_days, on="PERIODE")
    df_pivot = df_pivot.set_index("PERIODE")

    df_pivot = df_pivot.iloc[:,1:-3].div(df_pivot.number_of_days, axis=0).round(2)

    df_pivot=df_pivot.fillna(0).reset_index()
    df_pivot = pd.merge(df_pivot, df_days, on="PERIODE").sort_values(by="start")
    df_pivot = df_pivot.iloc[:,:-3]
    df_pivot['Total'] = df_pivot.sum(axis=1, numeric_only=True)
    st.write (df_pivot)
    print (df_pivot.columns.tolist())

def reduce_columns(df_pivot):
    df_pivot = df_pivot.fillna(0)
    df_pivot["MISC"] = df_pivot['MISC'] + df_pivot['MISC_']+df_pivot['MISC__'] + df_pivot['UNKNOWN'] + df_pivot['TINA_ISA'] + df_pivot['COVID']+ df_pivot['VISA'] 
    del df_pivot['MISC_']
    del df_pivot['MISC__']
    del df_pivot["UNKNOWN"]
    del df_pivot['TINA_ISA']
    del  df_pivot['COVID'] 
    del df_pivot['VISA']

    df_pivot["H"] = df_pivot["H"] + df_pivot["H_"]
    del df_pivot["H_"]
    del df_pivot["IN"]
    del df_pivot["CROSS"]
    df_pivot = df_pivot.reset_index()
   

    verw= ['AMS2017', 'UNKNOWN', 'AMS2021']
    for v in verw:
        df_pivot = df_pivot[df_pivot["PERIODE"] != v]
    return df_pivot

def get_and_prepare_data():
    df_uitgaves = read_csv(r"C:\Users\rcxsm\Documents\python_scripts\python_scripts_rcsmit\input\fianciele sheet azie verzamel - INPUT.csv")
    df_currency = get_forex_info()
    
    df_currency["VND"] = df_currency["VND"]/10000

    df_currency["IDR"] = df_currency["IDR"]/1000
    currencies =  df_currency.columns.tolist()
    for c in currencies:
        if c != "Date":
            df_currency[c] = df_currency[c].rolling(window = 5, center = False).mean() 
    df_currency = df_currency.melt(
            "Date", var_name="currency", value_name="rate"
        )
    df = pd.merge(df_uitgaves, df_currency,how="inner", left_on=["date", "currency"], right_on=["Date", "currency"])
    df = df.fillna(0)
    df['CAT'] = df['CAT'].str.upper()
    df["UITGAVEN"] = (df["UIT"]+df["UIT CB/HC"])
    df["UITGAVEN_EUR"] = (df["UIT"]+df["UIT CB/HC"])/df["rate"]
    return df, df_currency

def main():
    df,df_currency = get_and_prepare_data()
    make_table(df)
    list = ["CM2016","CM2017","CM2018","CM2019","CM2021"]
    for l in list:
        make_frequency_table(df, l, "D")
    st.subheader("Historical forex data")
    make_graph_currency(df_currency)

if __name__ == "__main__":
    main()

