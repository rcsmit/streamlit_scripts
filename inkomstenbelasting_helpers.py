import streamlit as st

from re import X
import pandas as pd
import plotly.express as px
import plotly
from plotly.subplots import make_subplots


#tarieven 2023
# https://www2.deloitte.com/nl/nl/pages/tax/articles/belastingplan-2023-overzicht-maatregelen-loonbelasting-inkomstenbelasting.html


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

def calculate_kindgebonden_budget(inkomen, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17):
    """_summary_

    Args:
        inkomen (int): Totaal belastbaar inkomen per jaar
        toeslagpartner (boolean): Is er een toeslagpartner
        aantal_kinderen (int): aantal kinderen onder de 18 totaal
        aantal_kinderen_12_15 (int): aantal kinderen 12 tot en met 15
        aantal_kinderen_16_17 (int): aantal kinderen 16 of 17 jaar

    Returns:
        int: Bedrag kindgebonden budget per jaar
    """    
    # https://download.belastingdienst.nl/toeslagen/docs/berekening_kindgebonden_budget_tg0811z21fd.pdf
    # geen toeslagpartner
    #     
    if aantal_kinderen == 0:
        kindgebonden_budget = 0
    else:
        if aantal_kinderen < aantal_kinderen_12_15+aantal_kinderen_16_17:
            raise Exception ("Aantal kinderen klopt niet")
        if toeslagpartner == False:
            if aantal_kinderen == 1: maximaal_bedrag_kindgebonden_budget = 4505
            if aantal_kinderen == 2: maximaal_bedrag_kindgebonden_budget = 5611
            if aantal_kinderen == 3: maximaal_bedrag_kindgebonden_budget = 6612
            if aantal_kinderen >= 4: maximaal_bedrag_kindgebonden_budget = 6612 + (aantal_kinderen-3)*1001
        else:
            if aantal_kinderen == 1: maximaal_bedrag_kindgebonden_budget = 1220
            if aantal_kinderen == 2: maximaal_bedrag_kindgebonden_budget = 2326
            if aantal_kinderen == 3: maximaal_bedrag_kindgebonden_budget = 3327
            if aantal_kinderen >= 4: maximaal_bedrag_kindgebonden_budget = 6612 + (aantal_kinderen-3)*1001
        
        maximaal_bedrag_kindgebonden_budget = maximaal_bedrag_kindgebonden_budget + (251*aantal_kinderen_12_15)
        maximaal_bedrag_kindgebonden_budget = maximaal_bedrag_kindgebonden_budget + (447*aantal_kinderen_16_17)
        
        if toeslagpartner == False:
            if inkomen > 22356:
                vermindering =  0.0675 * (inkomen - 22356)
                kindgebonden_budget = maximaal_bedrag_kindgebonden_budget - vermindering
            else:
                kindgebonden_budget = maximaal_bedrag_kindgebonden_budget
        else:
            if inkomen > 39596:
                vermindering =  0.0675 * (inkomen - 39596)
                kindgebonden_budget = maximaal_bedrag_kindgebonden_budget - vermindering
            else:
                kindgebonden_budget = maximaal_bedrag_kindgebonden_budget
        if kindgebonden_budget < 0 : kindgebonden_budget=0
    return kindgebonden_budget


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
  
def calculate_nettoloon(inkomen,rekenhuur,huishouden,number_household, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17):
    inkomensten_belasting = calculate_inkomstenbelasting(inkomen)
    heffingskorting = calculate_heffingskorting(inkomen)
    arbeidskorting = calculate_arbeidskorting(inkomen)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    huurtoeslag = calculate_huurtoeslag(inkomen,  rekenhuur,huishouden,number_household)
    zorgtoeslag = calculate_zorgtoeslag(inkomen)
    kindgebonden_budget =  calculate_kindgebonden_budget(inkomen, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17)
    regel = [inkomen,  inkomensten_belasting, heffingskorting, arbeidskorting,te_betalen_belasting, netto_loon, huurtoeslag, zorgtoeslag, kindgebonden_budget]
    return regel