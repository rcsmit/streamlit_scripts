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

def calculate_zorgtoeslag(toetsingsinkomen):
# https://www.belastingdienst.nl/wps/wcm/connect/nl/zorgtoeslag/content/hoeveel-zorgtoeslag
# https://download.belastingdienst.nl/toeslagen/docs/berekening_zorgtoeslag_tg0821z21fd.pdf
    # zorgtoeslag = (-0.0114* inkomen) + 365.15
    if toetsingsinkomen >31998 :
        zorgtoeslag = 0 
    else:
        standaardpremie = 1749
        drempelinkomen = 22356
        z = (toetsingsinkomen - drempelinkomen)
        if z<0:z=0
        normpremie = (0.01848 * drempelinkomen )+ (0.1361 * z)
        zorgtoeslag = standaardpremie - normpremie        
    return zorgtoeslag
def calculate_huurtoeslag(inkomen, rekenhuur,huishouden,number_household):
    """Huur toeslag berekenen

    Args:
        inkomen (int): Inkomen per jaar
        rekenhuur (int): Huur per maand
        huishouden (string): Een-of meerpersoons huishouden, met of zonder AOW: "EP", "MP, "EPAOW", "MPAOW"
        number_household (int): Aantal mensen in het huishouden

    Returns:
        int : huurtoeslag bedrag per jaar
    """    
    # https://www.volkshuisvestingnederland.nl/onderwerpen/huurtoeslag/werking-en-berekening-huurtoeslag
    # https://wetten.overheid.nl/BWBR0008659/2022-01-01
    # https://www.rigo.nl/wp-content/uploads/2007/06/De-huurtoeslag.pdf

    #https://download.belastingdienst.nl/toeslagen/docs/berekening_huurtoeslag_tg0831z21fd.pdf

    # Eenpersoons
    if huishouden == "EP":
        a,b,minimuminkomensijkpunt,  minimumnormhuur, basishuur_min = 5.96879*10**-7, 0.002363459319, 17350, 220.68, 237.62
    elif huishouden == "MP":
        a,b,minimuminkomensijkpunt,  minimumnormhuur, basishuur_min = 3.42858*10**-7, 0.002093692299,22500,220.68,237.62
    elif huishouden == "EPAOW":
        a,b,minimuminkomensijkpunt,  minimumnormhuur, basishuur_min = 8.00848*10**-7,  -0.003802527235, 19075,218.86,235.8
    elif huishouden == "MPAOW":
        a,b,minimuminkomensijkpunt,  minimumnormhuur, basishuur_min = 4.99095*10**-7, -0.004173489348,25450,217.05,233.99
    else:
        print (" ERROR in huishouden")

    taakstellingsbedrag = 16.94
    kwaliteitskortingsgrens = 442.46
    aftoppingsgrens = 633.25 if number_household <=2 else 678.66
    maximale_huurgrens = 763.47

    if rekenhuur > maximale_huurgrens :
        huurtoeslag, basishuur,A,B,C = 0,0,0,0,0
    else:
        if inkomen<minimuminkomensijkpunt:
            basishuur = basishuur_min
         
        else:   
            basishuur = (a*(inkomen**2) + (b* inkomen)) + taakstellingsbedrag
            if basishuur <basishuur_min : 
                basishuur = basishuur_min

        # DEEL A
        rr = rekenhuur if rekenhuur <= kwaliteitskortingsgrens else  kwaliteitskortingsgrens

        A = rr - basishuur
        if A <0: A=0
        # print(f"{A} = {rr} - {basishuur}")
        #DEEL B
        if rekenhuur>kwaliteitskortingsgrens:
            ss=rekenhuur if rekenhuur <=aftoppingsgrens else  aftoppingsgrens

            tt = basishuur if basishuur>=kwaliteitskortingsgrens else  kwaliteitskortingsgrens
            
            B = 0.65*(ss-tt)
            
            if B<0:B=0
            # print (f"B: {B} =  0.65*({ss}-{tt})")
        else:
            B=0
        
        # DEEL C
        # – Het gaat om een eenpersoonshuishouden.
        # – Iemand in het huishouden heeft op de datum van berekening de AOW‑leeftijd of is ouder.
        # – De woning is aangepast vanwege een handicap
        if rekenhuur > aftoppingsgrens:
            uu = basishuur if basishuur>=aftoppingsgrens else aftoppingsgrens
            C = 0.4*(rekenhuur-uu)
            if C<0:C=0
            # print (f"{C} = 0.4*({rekenhuur}-{uu}) ")
        else:
            C = 0 
        huurtoeslag = (A+B+C)*12        

    return huurtoeslag


  
def calculate_nettoloon(inkomen,rekenhuur,huishouden,number_household):
    inkomensten_belasting = calculate_inkomstenbelasting(inkomen)
    heffingskorting = calculate_heffingskorting(inkomen)
    arbeidskorting = calculate_arbeidskorting(inkomen)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    huurtoeslag = calculate_huurtoeslag(inkomen,  rekenhuur,huishouden,number_household)
    zorgtoeslag = calculate_zorgtoeslag(inkomen)
    regel = [inkomen,  inkomensten_belasting, heffingskorting, arbeidskorting,te_betalen_belasting, netto_loon, huurtoeslag, zorgtoeslag]
    return regel

def salaris_per_maand(max_value_ink):
    st.subheader("Salaris per maand excl. vakantiegeld")
    for inkomen in range(0,max_value_ink,1000):
        
        st.write(f"{inkomen} - {round(inkomen/12/1.08,2)}")

def create_df(tabeldata):
    df = pd.DataFrame(tabeldata, columns = ["inkomen",  "inkomensten_belasting", "heffingskorting", "arbeidskorting","te_betalen_belasting", "nettoloon", "huurtoeslag","zorgtoeslag"])
    df["belastingdruk_%"] = round(  df["te_betalen_belasting"]/df["inkomen"]*100,2)
    df["besteedbaar_inkomen"] =  df["nettoloon"]+ df["zorgtoeslag"]+ df["huurtoeslag"]
    df["toeslagen"] =  df["zorgtoeslag"]+ df["huurtoeslag"]
    df['te_betalen_belasting_diff'] = df['te_betalen_belasting'].diff()
    df['besteedbaar_inkomen_diff'] = df['besteedbaar_inkomen'].diff()
    df["toeslagen_diff"] = df["toeslagen"].diff()
    return df

def main():
    st.header("Inkomstenbelasting 2022")
   

    tabeldata=[]   
    max_value_ink = int(st.sidebar.number_input("Maximum waarde bruto inkomen",0,10_000_000,110_000,1000))
    stappen = int(st.sidebar.number_input("Stappen",0,10_000,1_000,1000))
    rekenhuur = int(st.sidebar.number_input("Rekenhuur",0,10000,700))
    huishouden = st.sidebar.selectbox("Type huishouden", ["EP", "MP", "EPAOW", "MPAOW"], index=0)
    if huishouden =="EP" or huishouden =="EPAOW":
        number_household = 1
    else: 
        number_household = int(st.sidebar.number_input("Aantal mensen in huishouden",0,10000,1))
    for inkomen in range(0,max_value_ink,stappen):
        regel = calculate_nettoloon(inkomen,rekenhuur,huishouden,number_household)
        tabeldata.append(regel)
       
    df = create_df(tabeldata)
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
