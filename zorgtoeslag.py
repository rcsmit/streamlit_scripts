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
            # if inkomen <= 22000 : zorgtoeslag = 111
            # if inkomen == 22000 : zorgtoeslag =	 111
            # if inkomen == 22500	: zorgtoeslag =	 109
            # if inkomen == 23000	: zorgtoeslag =	 104
            # if inkomen == 23500	: zorgtoeslag =	 98
            # if inkomen == 24000	: zorgtoeslag =	 92
            # if inkomen == 24500	: zorgtoeslag =	 87
            # if inkomen == 25000	: zorgtoeslag =	 81
            # if inkomen == 25500	: zorgtoeslag =	 75
            # if inkomen == 26000	: zorgtoeslag =	 69
            # if inkomen == 26500	: zorgtoeslag =	 64
            # if inkomen == 27000	: zorgtoeslag =	 58
            # if inkomen == 27500	: zorgtoeslag =	 52
            # if inkomen == 28000	: zorgtoeslag =	 47
            # if inkomen == 28500	: zorgtoeslag =	 41
            # if inkomen == 29000	: zorgtoeslag =	 35
            # if inkomen == 29500	: zorgtoeslag =	 30
            # if inkomen == 30000	: zorgtoeslag =	 24
            # if inkomen == 30500	: zorgtoeslag =	 18
            # if inkomen == 31000	: zorgtoeslag =	 13
            # if inkomen == 31500	: zorgtoeslag =	 7
        
        
    return zorgtoeslag

def main():
    for inkomen in range(1000,33_000,1000):
        zorgtoeslag = calculate_zorgtoeslag(inkomen)
        print (f"{inkomen} - {zorgtoeslag}")

main()