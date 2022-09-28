# Berekening inkomstenbelasting
# Alleen inkomen uit arbeid
# Onder 65 jaar

import streamlit as st

from re import X
import pandas as pd
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
def calculate_inkomstenbelasting(inkomen):
    grens_2022 = 69399
    grens_2021 = 68508
    grens = grens_2022
    if inkomen < grens:
        belasting = inkomen*0.3707
    else:
        belasting = (grens*0.3707)+((inkomen-grens)* 0.495)
  
    return round(belasting)

def calculate_heffingskorting(inkomen):
    if inkomen < 21317:
        heffingskorting = 2888
    elif inkomen>69398:
        heffingskorting = 0
    else:
        heffingskorting =  2888 - (0.06007 * (inkomen-  21317))
    return round(heffingskorting)

def calculate_arbeidskorting(inkomen):
    
    if inkomen <=10350:
        arbeidskorting =	0.04541 * inkomen
    if inkomen >= 10351 and inkomen<22357:
        arbeidskorting =  470 + (0.28461* (inkomen - 10350))

    if inkomen >= 22357 and inkomen<36650:
        arbeidskorting =  3887 + (0.02610 * (inkomen -  22356))
	
    if inkomen >= 36650 and inkomen<109347:
        arbeidskorting =  4260 - (0.05860 * (inkomen - 36649))
    if inkomen >= 109347:
        arbeidskorting = 0
   
    return round(arbeidskorting)


def calculate_zorgtoeslag(inkomen):
    # https://www.vergelijkdezorgverzekeringen.nl/zorgtoeslag/
    # https://www.belastingdienst.nl/wps/wcm/connect/nl/zorgtoeslag/content/hoeveel-zorgtoeslag
    # zorgtoeslag = (-0.0114* inkomen) + 365.15
    if inkomen <= 22000 : zorgtoeslag = 111
    if inkomen == 22000 : zorgtoeslag =	 111
    if inkomen == 22500	: zorgtoeslag =	 109
    if inkomen == 23000	: zorgtoeslag =	 104
    if inkomen == 23500	: zorgtoeslag =	 98
    if inkomen == 24000	: zorgtoeslag =	 92
    if inkomen == 24500	: zorgtoeslag =	 87
    if inkomen == 25000	: zorgtoeslag =	 81
    if inkomen == 25500	: zorgtoeslag =	 75
    if inkomen == 26000	: zorgtoeslag =	 69
    if inkomen == 26500	: zorgtoeslag =	 64
    if inkomen == 27000	: zorgtoeslag =	 58
    if inkomen == 27500	: zorgtoeslag =	 52
    if inkomen == 28000	: zorgtoeslag =	 47
    if inkomen == 28500	: zorgtoeslag =	 41
    if inkomen == 29000	: zorgtoeslag =	 35
    if inkomen == 29500	: zorgtoeslag =	 30
    if inkomen == 30000	: zorgtoeslag =	 24
    if inkomen == 30500	: zorgtoeslag =	 18
    if inkomen == 31000	: zorgtoeslag =	 13
    if inkomen == 31500	: zorgtoeslag =	 7
    if inkomen >31998 : zorgtoeslag = 0  
    zorgtoeslag = zorgtoeslag*12
    return zorgtoeslag

def calculate_huurtoeslag(inkomen):
    # aleenstaande- en wonende werkende man uit 1980, 36 uur per week werkend, huur = 700
    if inkomen <= 17000 : huurtoeslag = 355
    if inkomen == 18000 : huurtoeslag = 340
    if inkomen == 19000 : huurtoeslag = 315
    if inkomen == 20000 : huurtoeslag = 290
    if inkomen == 21000 : huurtoeslag = 263
    if inkomen == 22000 : huurtoeslag =	 235
    if inkomen == 23000	: huurtoeslag =	 206
    if inkomen == 24000	: huurtoeslag =	 175
    if inkomen == 25000	: huurtoeslag =	 146
    if inkomen == 26000	: huurtoeslag =	 125
    if inkomen == 27000	: huurtoeslag =	 102
    if inkomen == 28000	: huurtoeslag =	 80
    if inkomen == 29000	: huurtoeslag =	 56
    if inkomen == 30000	: huurtoeslag =	 32
    if inkomen == 31000	: huurtoeslag =	 14

    if inkomen >=32000 : huurtoeslag = 0  
   
    huurtoeslag_ = huurtoeslag *12
    # huur = 700
    # if huur < 233.99:
    #     huurtoeslag = 0
    # if huur >233.99 and huur <442.46:
    #     huurtoeslag = 100* (huur-233.99)
    # if huur >442.46 and huur < 633.25:
    #     x = huur - 442.46
    #     huurtoeslag = (442.46-233.99) + 0.65*x
    # if huur > 633.25 and huur < 763.47:
    #     y = huur - 633.25
    #     huurtoeslag = (442.46-233.99) + (0.65*(633.25-442.46)) + (0.4* y)
    # huurtoeslag = 0
    return huurtoeslag_

  
def calculate_nettoloon(inkomen):
    inkomensten_belasting = calculate_inkomstenbelasting(inkomen)
    heffingskorting = calculate_heffingskorting(inkomen)
    arbeidskorting = calculate_arbeidskorting(inkomen)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    huurtoeslag = calculate_huurtoeslag(inkomen)
    zorgtoeslag = calculate_zorgtoeslag(inkomen)
    regel = [inkomen,  inkomensten_belasting, heffingskorting, arbeidskorting,te_betalen_belasting, netto_loon, huurtoeslag, zorgtoeslag]
    return regel

def salaris_per_maand(max_value_ink):
    st.subheader("Salaris per maand excl. vakantiegeld")
    for inkomen in range(0,max_value_ink,1000):
        
        st.write(f"{inkomen} - {round(inkomen/12/1.08,2)}")

def main():
    st.header("Inkomstenbelasting 2022")
    st.write("* Alleen inkomen uit arbeid")
    st.write("* Alleenstaand, geen kinderen, onder 65 jaar")

    tabeldata=[]   
    max_value_ink = int(st.sidebar.number_input("Maximum waarde bruto inkomen",0,10_000_000,110_000,1000))
    stappen = int(st.sidebar.number_input("Stappen",0,10_000,1_000,1000))
    for inkomen in range(0,max_value_ink,stappen):
        regel = calculate_nettoloon(inkomen)
        tabeldata.append(regel)
       
    df = pd.DataFrame(tabeldata, columns = ["inkomen",  "inkomensten_belasting", "heffingskorting", "arbeidskorting","te_betalen_belasting", "nettoloon", "huurtoeslag","zorgtoeslag"])
    df["belastingdruk_%"] = round(  df["te_betalen_belasting"]/df["inkomen"]*100,2)
    df["besteedbaar_inkomen"] =  df["nettoloon"]+ df["zorgtoeslag"]+ df["huurtoeslag"]
    df["toeslagen"] =  df["zorgtoeslag"]+ df["huurtoeslag"]
    df['te_betalen_belasting_diff'] = df['te_betalen_belasting'].diff()
    df['besteedbaar_inkomen_diff'] = df['besteedbaar_inkomen'].diff()
    df["toeslagen_diff"] = df["toeslagen"].diff()
    st.write (df)
    # https://stackoverflow.com/questions/62853539/plotly-how-to-plot-on-secondary-y-axis-with-plotly-express
    
    to_show_ = ["nettoloon","besteedbaar_inkomen", "te_betalen_belasting", "belastingdruk_%", "zorgtoeslag", "huurtoeslag","toeslagen", ["huurtoeslag","zorgtoeslag","toeslagen"], "toeslagen_diff", "besteedbaar_inkomen_diff", "te_betalen_belasting_diff"]
    for to_show in to_show_:    
        fig = px.line(df,x="inkomen",y=to_show)
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
        fig.layout.xaxis.title="Bruto inkomen per jaar"
        
        #subfig.layout.yaxis2.title="Percentage"
        #plotly.offline.plot(subfig)
        st.subheader(to_show)
        st.plotly_chart(fig) 
    salaris_per_maand(max_value_ink)
    st.write("belastingdruk_% = inkomstenbelasting / bruto inkomen")
    st.write("Netto inkomen = bruto inkomen - inkomstenbelasting")
    st.write("Besteedbaar inkomen = netto inkomen + huurtoeslag + zorgtoeslag")
    st.write("ONDER VOORBEHOUD VAN FOUTEN")
    st.write("Zie ook https://www.rijksoverheid.nl/documenten/kamerstukken/2021/09/21/tabellen-marginale-druk-pakket-belastingplan-2022")
if __name__ == "__main__":
    main()
