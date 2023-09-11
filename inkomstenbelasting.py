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

def create_df(tabeldata,uren_per_week, methode):
    df = pd.DataFrame(tabeldata, columns = ["uren_per_week","bruto_inkomen",  "inkomensten_belasting", "heffingskorting", "arbeidskorting","te_betalen_belasting", "nettoloon", "huurtoeslag","zorgtoeslag", "kindgebonden_budget",  "bbb_inkomen"])
    df["belastingdruk_%"] = round(  df["te_betalen_belasting"]/df["bruto_inkomen"]*100,2)
    df["besteedbaar_inkomen"] =  df["nettoloon"]+ df["zorgtoeslag"]+ df["huurtoeslag"]+df["kindgebonden_budget"]
    if methode == "inkomen":
        df["besteedbaar_per_uur"] = round(df["besteedbaar_inkomen"] /(4.33*uren_per_week * 12),2)
    else:
        df["besteedbaar_per_uur"] = round(df["besteedbaar_inkomen"] /(4.33*df["uren_per_week"] * 12),2)
    df["toeslagen"] =  df["zorgtoeslag"]+ df["huurtoeslag"] +df["kindgebonden_budget"]
    df['te_betalen_belasting_diff'] = df['te_betalen_belasting'].diff()
    df['besteedbaar_inkomen_diff'] = df['besteedbaar_inkomen'].diff()
    df["toeslagen_diff"] = df["toeslagen"].diff()
    return df

def main():
    st.header("Inkomstenbelasting 2022 zonder toeslagpartner")
   
    st.warning("ONDER VOORBEHOUD VAN FOUTEN. Sterk indicatief. Laat eea narekenen alvorens belangrijke beslissingen te nemen. Aantal werkuren per week heeft ook invloed op pensioensopbouw en sociale verzekeringen.")
    tabeldata=[]   
    methode = st.sidebar.selectbox("Methode (jaarinkomen / uren per week)", ["inkomen", "uren_per_week"],0)
    if methode == "inkomen":
        max_value_ink = int(st.sidebar.number_input("Maximum waarde bruto inkomen",0,10_000_000,110_000,1000))
        stappen = int(st.sidebar.number_input("Stappen",0,10_000,1_000,1000))
    elif methode =="uren_per_week":
        uur_salaris = st.sidebar.number_input("uursalaris (incl. reserveringen etc. minimum: 11,51 / modaal:18,80)", 0.1,100.00,12.00)

        max_value_ink = int(uur_salaris * 4.33 *41 * 12)

        stappen = int( uur_salaris * 4.33  * 12)
    rekenhuur = int(st.sidebar.number_input("Rekenhuur (>764 geen huurtoeslag)",0,10000,700))
    huishouden = st.sidebar.selectbox("Type huishouden", ["EP",  "EPAOW"], index=0) # "MP","MPAOW"
    if huishouden =="EP" or huishouden =="EPAOW":
        number_household = 1
    else: 
        number_household = int(st.sidebar.number_input("Aantal mensen in huishouden",0,10000,1))

    toeslagpartner =  False #st.sidebar.selectbox("Toeslagpartner",[True,False],index = 1)
    aantal_kinderen =  int(st.sidebar.number_input("Aantal kinderen",0,10,0))
    aantal_kinderen_12_15 =  int(st.sidebar.number_input("Aantal kinderen 12-15",0,10,0))
    aantal_kinderen_16_17 = int(st.sidebar.number_input("Aantal kinderen 16-17",0,10,0))
    if methode == "inkomen":
        uren_per_week_ink = int(st.sidebar.number_input("Aantal uren per week",1,100,40))
    else:
        uren_per_week_ink = 1
    bbb_grens = 30000
    if (aantal_kinderen < aantal_kinderen_12_15+aantal_kinderen_16_17):
        st.error("Aantal kinderen klopt niet")
        st.stop()

   
    for inkomen in range(0,max_value_ink,stappen):
        if methode == "inkomen":
            uren_per_week = uren_per_week_ink  
        else:  
            uren_per_week =  int(inkomen / (uur_salaris * 4.33  * 12))
        regel = calculate_nettoloon(inkomen/12,12,rekenhuur,huishouden,number_household, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17, uren_per_week, bbb_grens)
        
        tabeldata.append(regel)

    
       
    df = create_df(tabeldata, uren_per_week, methode)
    st.write (df)
    # https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express
    
    #if methode == "inkomen":
    to_show_ = [["nettoloon","besteedbaar_inkomen", "bbb_inkomen", "bruto_inkomen"], "besteedbaar_per_uur", "te_betalen_belasting", "belastingdruk_%", "zorgtoeslag", "huurtoeslag","kindgebonden_budget","toeslagen", ["huurtoeslag","zorgtoeslag","kindgebonden_budget","toeslagen"], "toeslagen_diff", "besteedbaar_inkomen_diff", "te_betalen_belasting_diff"]
    #else:
    #    to_show_ = ["nettoloon",["besteedbaar_inkomen", "bbb_inkomen"], "te_betalen_belasting", "belastingdruk_%", "zorgtoeslag", "huurtoeslag","kindgebonden_budget","toeslagen", ["huurtoeslag","zorgtoeslag","kindgebonden_budget","toeslagen"], "toeslagen_diff", "besteedbaar_inkomen_diff", "te_betalen_belasting_diff"]
    
    
    for to_show in to_show_:   
        if methode == "inkomen": 
            x_ = "bruto_inkomen"
            x_title = "Bruto inkomen per jaar"
        else:
            x_ = "uren_per_week"
            x_title = "Uren per week"
            
    
        fig = px.line(df,x=x_,y=to_show)
        fig.layout.xaxis.title= x_title
        if to_show == ["nettoloon","besteedbaar_inkomen", "bbb_inkomen", "bruto_inkomen"]:
            for t in to_show:
                df[f"{t}_m"] = df[t]/ 12
            to_show_m = ["nettoloon_m", "besteedbaar_inkomen_m", "bbb_inkomen_m", "bruto_inkomen_m"]
            if methode == "uren_per_week":
                st.subheader("Gevolgen meer/minder werken")
                col1,col2 = st.columns(2)
                with col1:
                    uren_1 = st.number_input("Uren per week oude situatie", 1,40,32)
                with col2:
                    uren_2 = st.number_input("Uren per week nieuwe situatie", 1,40,36)

                filtered_data_1 = df[df['uren_per_week'] == uren_1]
                filtered_data_2 = df[df['uren_per_week'] == uren_2]
                
                # Combine the two dataframes into one
                combined_data = pd.concat([filtered_data_1, filtered_data_2])

                # Reset the index if needed
                combined_data.reset_index(drop=True, inplace=True)

                st.write(combined_data)
                
                bruto_inkomen_m_diff = round(filtered_data_2['bruto_inkomen_m'].iloc[0] - filtered_data_1['bruto_inkomen_m'].iloc[0],2)
                bruto_inkomen_u_diff = round((filtered_data_1['bruto_inkomen_m'].iloc[0]/(4.33*uren_1)) - (filtered_data_2['bruto_inkomen_m'].iloc[0]/(4.33*uren_2)),2)
                               

                nettoloon_m_diff = round(filtered_data_2['nettoloon_m'].iloc[0] - filtered_data_1['nettoloon_m'].iloc[0],2)
                besteedbaar_inkomen_m_diff = round(filtered_data_2['besteedbaar_inkomen_m'].iloc[0] - filtered_data_1['besteedbaar_inkomen_m'].iloc[0],2)
                
                
                nettoloon_u_diff =           round((filtered_data_2['nettoloon_m'].iloc[0]/(4.33*uren_2)) -            (filtered_data_1['nettoloon_m'].iloc[0]/(4.33*uren_1)),2)
                besteedbaar_inkomen_u_diff = round((filtered_data_2['besteedbaar_inkomen_m'].iloc[0]/(4.33*uren_2)) - (filtered_data_1['besteedbaar_inkomen_m'].iloc[0]/(4.33*uren_1)),2)
                
                st.write(f"Verschil Bruto inkomen: per jaar : {round(12* bruto_inkomen_m_diff)} | per maand : {bruto_inkomen_m_diff} ")
                            
                st.write(f"Verschil netto loon: per jaar : {round(12* nettoloon_m_diff)} | per maand : {nettoloon_m_diff} | per uur {nettoloon_u_diff} ")
                st.write(f"**Verschil besteedbaar inkomen : per jaar : {round(12* besteedbaar_inkomen_m_diff)} | per maand : {besteedbaar_inkomen_m_diff} | per uur : {besteedbaar_inkomen_u_diff}**")
            fig2 = px.line(df,x=x_,y=to_show_m)
            fig2.layout.xaxis.title=x_title
            st.subheader("Per maand")
            st.plotly_chart(fig2) 
         #fig.layout.yaxis.title=to_show
        # if type(to_show) == list :
        #     fig = px.Figure()
        #     for ts in to_show:
        #         #fig = px.line(df,x="inkomen",y=df[ts])
        #         fig.add_trace( px.line(df,x="inkomen",y=df[ts]))
        #         fig.layout.yaxis.title="bedragen"
        # else:
        #     fig = px.line(df,x="inkomen",y=to_show)
        #     fig.layout.yaxis.title=to_show

        # in het geval van een secundaire y-as
        # fig2 = px.line(df,x="inkomen",y="belastingdruk")
        # #fig2 = px.line(df,x="inkomen",y="huurtoeslag")
        # fig2.update_traces(yaxis="y2")
        # subfig.add_traces(fig.data + fig2.data)
        # subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
        
        
        #subfig.layout.yaxis2.title="Percentage"
        #plotly.offline.plot(subfig)
        st.subheader(to_show)
        st.plotly_chart(fig) 
    # salaris_per_maand(max_value_ink)
    st.write("belastingdruk_% = inkomstenbelasting / bruto inkomen")
    st.write("Netto inkomen = bruto inkomen - inkomstenbelasting")
    st.write("Besteedbaar inkomen = netto inkomen + huurtoeslag + zorgtoeslag")
    
    st.write("Zie ook https://www.rijksoverheid.nl/documenten/kamerstukken/2021/09/21/tabellen-marginale-druk-pakket-belastingplan-2022")


if __name__ == "__main__":
    main()
