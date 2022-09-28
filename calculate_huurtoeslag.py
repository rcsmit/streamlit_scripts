def calculate_huurtoeslag(inkomen, huur):
    # https://www.volkshuisvestingnederland.nl/onderwerpen/huurtoeslag/werking-en-berekening-huurtoeslag
    # https://wetten.overheid.nl/BWBR0008659/2022-01-01

    # gevonden met fitter
    # a = 5.680160068245925*10**-7
    # b = 0.0038134086872441313

    #https://wetten.overheid.nl/BWBR0045799/2022-01-01 2022
    a = 5.96879*10**-7
    b = 0.002363459319

    # 2021 https://wetten.overheid.nl/BWBR0044343/2021-01-01
    # a= 6.23385*10**-7
    # b= 0.002453085056

    WMLmaand =  1264.80 # https://wetten.overheid.nl/BWBR0002638/2022-08-02/#HoofdstukII_Artikel8 8.1.a
    minimuminkomensijkpunt = 17350 # (0.81 *12*WMLmaand) + 572 #eenpersoonshuishoudens
    _minimuminkomensijkpunt = (1.08 *12*WMLmaand) + 144 #eenpersoonshuishoudens
    
    if huur > 763.47:
        huurtoeslag = 0
    else:
        if inkomen < minimuminkomensijkpunt:
    
            normhuur = 220.68
        else:
            normhuur = a*(inkomen**2) + (b* inkomen)
        # if normhuur > 448.57:
        #     normhuur = 448.57
        

        # 100% tussen de minimale basishuur (inkomensafhankelijke eigen bijdrage) en de kwaliteitskortingsgrens (€442,46)
        # 65% tussen de kwaliteitskortingsgrens en de aftoppingsgrens (€633,25 of €678,66)
        # 40% voor de meeste huurders tussen de aftoppingsgrens en de maximale huurgrens (€763,47).
        basishuur = 220.68-1.82
        kwaliteitskortingsgrens = 442.46
        aftoppingsgrens = 663.25
        maximale_huurgrens = 763.47
        a,b,c = 0,0,0


       
        if huur < normhuur:
            pass

        elif huur >normhuur and kwaliteitskortingsgrens <442.46:
            a = 1.00* (huur-normhuur)
        elif huur >kwaliteitskortingsgrens and huur < aftoppingsgrens:
            a = 1.00* (kwaliteitskortingsgrens-normhuur)
            b  = 0.65 * (huur- kwaliteitskortingsgrens)

        elif huur > aftoppingsgrens and huur < maximale_huurgrens:
            a = 1.00* (kwaliteitskortingsgrens-normhuur)
            b  = 0.65 * (aftoppingsgrens - kwaliteitskortingsgrens)
            c = 0.4 * (huur -aftoppingsgrens)
            

        elif huur > maximale_huurgrens:
            a = 1.00* (kwaliteitskortingsgrens-normhuur)
            b  = 0.65 * (aftoppingsgrens - kwaliteitskortingsgrens)
            c = 0.4 * (maximale_huurgrens -aftoppingsgrens)
        
        huurtoeslag = a+b+c
        
        if huurtoeslag<0 :
            huurtoeslag=0
      

    return huurtoeslag, normhuur,a,b,c

def main():
    max_value_ink = 50_000
    huur = 700
    for inkomen in range(0,max_value_ink ,1000):
        huurtoeslag, normhuur,a,b,c = calculate_huurtoeslag(inkomen, huur)
        print (f"{inkomen} | {int(huurtoeslag)} | {int(normhuur)} | {int(a)} | {int(b)} |{int(c)}")

main()