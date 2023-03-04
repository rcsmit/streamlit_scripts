def calculate_huurtoeslag(inkomen, rekenhuur,huishouden,number_household):
    # https://www.volkshuisvestingnederland.nl/onderwerpen/huurtoeslag/werking-en-berekening-huurtoeslag
    # https://wetten.overheid.nl/BWBR0008659/2022-01-01


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
        huurtoeslag = A+B+C        

    return huurtoeslag, basishuur,A,B,C

def main():
    max_value_ink = 35_000
    for inkomen in range(16_000,max_value_ink ,1000):
        huur = 700
    
        huishouden = "EP"

        number_household=1
        huurtoeslag, basishuur,a,b,c = calculate_huurtoeslag(inkomen, huur, huishouden,number_household)
        print (f"{inkomen=} | {round(huurtoeslag,2)=} || {int(basishuur)=} | {int(a)} | {int(b)} |{int(c)}")

main()