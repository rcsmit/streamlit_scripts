# import streamlit as st

# from re import X
import pandas as pd
# import plotly.express as px
# import plotly
# from plotly.subplots import make_subplots


#tarieven 2023
# https://www2.deloitte.com/nl/nl/pages/tax/articles/belastingplan-2023-overzicht-maatregelen-loonbelasting-inkomstenbelasting.html


# https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/themaoverstijgend/brochures_en_publicaties/handboek-loonheffingen-2022
# https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/themaoverstijgend/brochures_en_publicaties/rekenvoorschriften-voor-de-geautomatiseerde-loonadministratie-januari-2022


def calculate_inkomstenbelasting(inkomen):
    grens_2023 = 73031
   
    grens = grens_2023
    if inkomen < grens:
        belasting = inkomen*0.3693
    else:
        belasting = (grens*0.3693)+((inkomen-grens)* 0.495)
  
    return round(belasting)

def calculate_arbeidskorting_niet_alle_maanden_werken(maandsalaris, aantal_maanden):
    arbeidskorting_per_maand = calculate_arbeidskorting_2022(maandsalaris*12)/12
    totale_arbeidskorting = arbeidskorting_per_maand * aantal_maanden
    return totale_arbeidskorting


def calculate_heffingskorting(inkomen):
    if inkomen < 22261:
        heffingskorting = 3070
    elif inkomen>73031:
        heffingskorting = 0
    else:
        heffingskorting =  3070 - (0.06095 * (inkomen-  22660))
   
    return round(heffingskorting)
def calculate_arbeidskorting(inkomen):
    arbeidskorting = 0
    if inkomen <=10741:
        arbeidskorting =	0.08321 * inkomen
    if inkomen >= 10741 and inkomen<23201:
        arbeidskorting =  884 + (0.29861* (inkomen - 10740))

    if inkomen >= 23201 and inkomen<37691:
        arbeidskorting =  4605 + (0.03085 * (inkomen -  23200))
	
    if inkomen >= 37691 and inkomen<115295:
        arbeidskorting =  5052 - (0.06510 * (inkomen - 37690))
    if inkomen >= 115295:
        arbeidskorting = 0
   
    return round(arbeidskorting)

def calculate_arbeidskorting_2022(inkomen):
    arbeidskorting = 0
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

def calculate_heffingskorting_2022(inkomen):
    if inkomen < 21317:
        heffingskorting = 2888
    elif inkomen>69398:
        heffingskorting = 0
    else:
        heffingskorting =  2888 - (0.06007 * (inkomen-  21317))
    return round(heffingskorting)

  
def calculate_inkomstenbelasting_2022(inkomen):
    grens_2022 = 69399
    grens_2021 = 68508
    grens = grens_2022
    if inkomen < grens:
        belasting = inkomen*0.3707
    else:
        belasting = (grens*0.3707)+((inkomen-grens)* 0.495)
  
    return round(belasting)

def calculate_zorgtoeslag(toetsingsinkomen):
# https://www.belastingdienst.nl/wps/wcm/connect/nl/zorgtoeslag/content/hoeveel-zorgtoeslag
# https://download.belastingdienst.nl/toeslagen/docs/berekening_zorgtoeslag_tg0821z21fd.pdf
# via https://www.belastingdienst.nl/wps/wcm/connect/bldcontentnl/themaoverstijgend/brochures_en_publicaties/brochures_en_publicaties_intermediair
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
  
def calculate_nettoloon_simpel (maand_inkomen, aantal_maanden):
    inkomen = maand_inkomen * aantal_maanden
    arbeidskorting = calculate_arbeidskorting_niet_alle_maanden_werken(maand_inkomen, aantal_maanden)
    inkomensten_belasting = calculate_inkomstenbelasting(inkomen)
    heffingskorting = calculate_heffingskorting(inkomen)
    #arbeidskorting = calculate_arbeidskorting(inkomen)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    #print (f"{inkomen=} {netto_loon=}- {inkomensten_belasting=} - {heffingskorting=} - {arbeidskorting=} {te_betalen_belasting=}")
    return netto_loon

def calculate_nettoloon_simpel_2022 (inkomen):
    #print (f"inkomen = {inkomen}")
    inkomensten_belasting = calculate_inkomstenbelasting_2022(inkomen)
    heffingskorting = calculate_heffingskorting_2022(inkomen)
    arbeidskorting = calculate_arbeidskorting_2022(inkomen)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    #print (f"{te_betalen_belasting=}")
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    return netto_loon

def twee_vs_drie():
    """Bereken het verschil in inkomstenbelasting tussen 2022 en 2023
    """    
    import pandas as pd

    import plotly.express as px
    import plotly
    plotly.offline.init_notebook_mode(connected=True)
    list = []
    for i in range(1000,200000,1000):
        twee = calculate_nettoloon_simpel_2022(i)
        drie = calculate_nettoloon_simpel(i)
        list.append([i,twee,drie,i-twee,i-drie,drie-twee])
    columns = ["bruto_inkomen", "2022", "2023","te_betalen_2022", "te betalen_2023", "verschil"] 
    df = pd.DataFrame(list, columns=columns)#.set_index("months_working")
    df["bruto_maand"] = (df["bruto_inkomen"]/12)
    df["verschil_maand"] = df["verschil"]/12
    df["netto_maand_2023"] = df["2023"]/12
    print (df)
    fig = px.line(df, x="bruto_inkomen", y=["2022","2023","te_betalen_2022", "te betalen_2023"], title = "Netto loon 2022&2023 vs inkomen")
    plotly.offline.plot(fig)
    # fig2 = px.line(df, x="bruto_inkomen", y=["verschil"], title = "Netto loon 2022&2023 vs inkomen")
    # plotly.offline.plot(fig2)
    # fig3 = px.line(df, x="bruto_maand", y=["netto_maand_2023"], title = "Netto 2023 en Verschil per maand vs salaris per maand")
    # plotly.offline.plot(fig3)
    # fig3 = px.line(df, x="bruto_maand", y=["verschil_maand"], title = "Netto 2023 en Verschil per maand vs salaris per maand")
    # plotly.offline.plot(fig3)

def calculate_nettoloon(maand_inkomen,aantal_maanden,rekenhuur,huishouden,number_household, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17, bbb_grens):
    """_summary_

    Args:
        inkomen (_type_): _description_
        rekenhuur (_type_): _description_
        huishouden (_type_): _description_
        number_household (_type_): _description_
        toeslagpartner (_type_): _description_
        aantal_kinderen (_type_): _description_
        aantal_kinderen_12_15 (_type_): _description_
        aantal_kinderen_16_17 (_type_): _description_
        bbb_grens : inkomen tot waar je geen inkomstenbelasting betaalt

    Returns:
        _type_: inkomen,  inkomensten_belasting, heffingskorting, arbeidskorting,te_betalen_belasting, netto_loon, huurtoeslag, zorgtoeslag, kindgebonden_budget
 
    """
    inkomen = maand_inkomen * aantal_maanden
    inkomensten_belasting = calculate_inkomstenbelasting(inkomen)
    heffingskorting = calculate_heffingskorting(inkomen)
    arbeidskorting = calculate_arbeidskorting(inkomen)
    arbeidskorting = calculate_arbeidskorting_niet_alle_maanden_werken(1600,4)

    te_betalen_belasting = inkomensten_belasting - heffingskorting - arbeidskorting
    if te_betalen_belasting <0:
        te_betalen_belasting = 0

    netto_loon = inkomen - te_betalen_belasting
    huurtoeslag = calculate_huurtoeslag(inkomen,  rekenhuur,huishouden,number_household)
    zorgtoeslag = calculate_zorgtoeslag(inkomen)
    kindgebonden_budget =  calculate_kindgebonden_budget(inkomen, toeslagpartner,aantal_kinderen, aantal_kinderen_12_15, aantal_kinderen_16_17)
    
    if inkomen <bbb_grens:
        bbb_inkomen = inkomen
    else:
        bbb_inkomen = netto_loon
    regel = [inkomen,  inkomensten_belasting, heffingskorting, arbeidskorting,te_betalen_belasting, netto_loon, huurtoeslag, zorgtoeslag, kindgebonden_budget,bbb_inkomen]
    return regel

# #print(calculate_nettoloon_simpel_2022 (1000+(2150*7*1.18)))
def main_aantal_maanden():

    # Define the salaries and number of months
    #salaries = range(1000, 11000, 1000)
    salaries = range(1500, 5000, 200)
    num_months = range(1, 13)

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Add the months as columns
    for month in num_months:
        df[f"{month} months"] = ""

    # Add the salaries and percentage of salary earned per month for each month
    for salary in salaries:
        row = []
        sup =0
        for month in num_months:
            
            if month == 0:
                percentage = 0
            else:
                bruto = month*salary*1.18
                netto = calculate_nettoloon_simpel(salary*1.18,month)
                te_betalen = bruto - netto
                percentage = round(te_betalen/bruto*100,1)
            # zet de percentage te betalen belasting in een tabel
            #row.append(f"{percentage:.1f}%")

            # zet het extra netto salaris voor een maand meer werken in een tabel
            row.append(int(netto-sup))
            sup= netto

            # zet de te betalen belasting in een tabel
            #row.append(f"{int((te_betalen/month)/salary*100)}%")

            
        
        df_row = pd.Series(row, index=df.columns)
        df = pd.concat([df, df_row.to_frame().T])

    # Set the index to the salaries
    df.set_index(pd.Index([f"EUR {salary}" for salary in salaries]), inplace=True)

    # Display the DataFrame
    print(df)

def main_aantal_uren_per_week():

    # Define the salaries and number of months
    #salaries = range(1000, 11000, 1000)
    salaries = range(1500, 2500, 100)
    parttime_perc = range(10,110,10)

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Add the months as columns
    for perc in parttime_perc:
        df[f"{perc} %"] = ""

    # Add the salaries and percentage of salary earned per month for each month
    for salary in salaries:
        row = []
        sup =0
        for perc in parttime_perc:
            
            bruto = perc*salary*1.18/100*12
            netto = calculate_nettoloon_simpel(perc*salary*1.18/100,12)
            te_betalen = bruto - netto
            percentage = round(te_betalen/bruto*100,1)
            #row.append(netto)
            # zet de percentage te betalen belasting in een tabel
            #row.append(f"{percentage:.1f}%")

            # zet het extra netto salaris voor een maand meer werken in een tabel
            row.append(int(netto-sup))
            sup= netto

            # zet de te betalen belasting in een tabel
            #row.append(f"{int((te_betalen/month)/salary*100)}%")

            
        
        df_row = pd.Series(row, index=df.columns)
        df = pd.concat([df, df_row.to_frame().T])

    # Set the index to the salaries
    df.set_index(pd.Index([f"EUR {salary}" for salary in salaries]), inplace=True)

    # Display the DataFrame
    print(df)

   

main_aantal_uren_per_week()
