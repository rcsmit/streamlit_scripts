# Berekening inkomstenbelasting
# Alleen inkomen uit arbeid
# Onder 65 jaar

import streamlit as st

from re import X
import pandas as pd
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
from inkomstenbelasting_helpers import *


def salaris_per_maand(max_value_ink):
    st.subheader("Salaris per maand excl. vakantiegeld")
    for inkomen in range(0,max_value_ink,1000):
        
        st.write(f"{inkomen} - {round(inkomen/12/1.08,2)}")

def create_df(tabeldata):
    df = pd.DataFrame(tabeldata, columns = ["inkomen",  "inkomensten_belasting", "heffingskorting", "arbeidskorting","te_betalen_belasting", "nettoloon", "huurtoeslag","zorgtoeslag", "kindgebonden_budget"])
    df["belastingdruk_%"] = round(  df["te_betalen_belasting"]/df["inkomen"]*100,2)
    df["besteedbaar_inkomen"] =  df["nettoloon"]+ df["zorgtoeslag"]+ df["huurtoeslag"]+df["kindgebonden_budget"]
    df["toeslagen"] =  df["zorgtoeslag"]+ df["huurtoeslag"] +df["kindgebonden_budget"]
    df['te_betalen_belasting_diff'] = df['te_betalen_belasting'].diff()
    df['besteedbaar_inkomen_diff'] = df['besteedbaar_inkomen'].diff()
    df["toeslagen_diff"] = df["toeslagen"].diff()
    return df

def main():
    st.header("Inkomstenbelasting 2022 zonder toeslagpartner")
   

    tabeldata=[]   
    inkomen = int(st.sidebar.number_input("Maximum waarde bruto inkomen",0,10_000_000,30_000,1000))
    
    rekenhuur = int(st.sidebar.number_input("Rekenhuur",0,10000,700))
    huishouden = st.sidebar.selectbox("Type huishouden", ["EP",  "EPAOW"], index=0) # "MP","MPAOW"
    if huishouden =="EP" or huishouden =="EPAOW":
        number_household = 1
    else: 
        number_household = int(st.sidebar.number_input("Aantal mensen in huishouden",0,10000,1))

    toeslagpartner =  False #st.sidebar.selectbox("Toeslagpartner",[True,False],index = 1)
    aantal_kinderen =  int(st.sidebar.number_input("Aantal kinderen",0,10,0))
    aantal_kinderen_12_15 =  int(st.sidebar.number_input("Aantal kinderen 12-15",0,10,0))
    aantal_kinderen_16_17 = int(st.sidebar.number_input("Aantal kinderen 16-17",0,10,0))
    if (aantal_kinderen < aantal_kinderen_12_15+aantal_kinderen_16_17):
        st.error("Aantal kinderen klopt niet")
        st.stop()
    regel = calculate_nettoloon(inkomen/12,12,rekenhuur,huishouden,number_household, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17)
    tabeldata.append(regel)
    
    df = create_df(tabeldata)
    st.table (df)
    # https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express
    

  
    st.write("belastingdruk_% = inkomstenbelasting / bruto inkomen")
    st.write("Netto inkomen = bruto inkomen - inkomstenbelasting")
    st.write("Besteedbaar inkomen = netto inkomen + huurtoeslag + zorgtoeslag")
    st.write("ONDER VOORBEHOUD VAN FOUTEN")
    st.write("Zie ook https://www.rijksoverheid.nl/documenten/kamerstukken/2021/09/21/tabellen-marginale-druk-pakket-belastingplan-2022")


if __name__ == "__main__":
    main()
