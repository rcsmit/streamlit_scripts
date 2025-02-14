from kerykeion import AstrologicalSubject
import swisseph as swe

import pandas as pd
import datetime
from skyfield.api import load
import ephem 
"""

https://github.com/jdempcy/hdkit converted to Python
uses also https://onecompiler.com/python/3z7ecjyb5

"""

def decdeg2dms(dd):
    """Convert degrees in decimals to DMS
    Args:
        dd (float) : degrees in decimals
    Returns:
        D,M,S: degrees, minutes, secondes
    """    
    mult = -1 if dd < 0 else 1
    mnt,sec = divmod(abs(dd)*3600, 60)
    deg,mnt = divmod(mnt, 60)
    return mult*deg, mult*mnt, mult*sec


def get_design_datetime(sun_position_at_birth_dec,y,m,d,h,min,delta):
    """ Get the design date time. It is the moment that the sun was 88 degrees in retrograde. 
        We also have to converrt this date time to Greenwich Mean Time

    Args:
        sun_position_at_birth_dec (float): sun position of birth in decimals
        y (int): year of birth
        m (int): month of birth
        d (int): day of birth
        h (int): hour of birth
        min (int): minutes of birth
        delta (int) : time difference with GMT

    Returns:
        _type_: _description_
    """
    # Desired Sun position (88° retrograde)
    desired_position = -88

    birthdate = datetime.datetime(y,m,d,h,min)  # Example birthdate
    time_of_birth = datetime.time(h,min )
    # Calculate the Sun's position at the time of birth
    birth_datetime = datetime.datetime.combine(birthdate, time_of_birth)
    
    # Calculate the date and time when the Sun was at the desired position
    delta_position = sun_position_at_birth_dec - desired_position

    # Calculate the date and time for the desired position
    design_datetime_local = birth_datetime - datetime.timedelta(days=delta_position / 360 * 365)

    # Define a timedelta of [delta] hours
    hours_to_add = datetime.timedelta(hours=delta)
    design_datetime_gmt = design_datetime_local + hours_to_add
    
    # Extract individual components from the GMT datetime
    gmt_year = design_datetime_gmt.year
    gmt_month = design_datetime_gmt.month
    gmt_day = design_datetime_gmt.day
    gmt_hour = design_datetime_gmt.hour
    gmt_minute = design_datetime_gmt.minute
    return  gmt_year,gmt_month,gmt_day,gmt_hour,gmt_minute

def main_kerykeion(y,m,d,h,min, birthplace, birthcountry, delta):
    """Calculates the DMS-info for the Personality and the design with Kerykeion.
        Earth and South-node is missing

        Design is 88° retrograde of the Sun from the time of birth. So you find
        when the Sun was 88° before and calculate all planetary positions for that
        time.
    Args:
        y (int): year of birth
        m (int): month of birth
        d (int): day of birth
        h (int): hour of birth
        min (int): minutes of birth
        birthplace (str) : place of birth
        birthcountry (str) : country of birth
        delta (int) : time difference with GMT

    Returns:
        _type_: _description_
    """    
    
    personality = AstrologicalSubject("Jonah", y,m,d,h,min, birthplace, birthcountry)
    sun_position_at_birth_dec = personality.sun['position']

    gmt_year,gmt_month,gmt_day,gmt_hour,gmt_minute = get_design_datetime(sun_position_at_birth_dec, y,m,d,h,min,delta ) 
    design = AstrologicalSubject("Jonah",  gmt_year,gmt_month,gmt_day,gmt_hour,gmt_minute , "London", "GB")
    items = [personality,design]
    items_t = ["personality","design"]
    
    data_list = []
    for item,item_t in zip(items,items_t):
        print ("--------------------------")
       
        sun = item.sun
        
        moon = item.moon
        mercury = item.mercury
        venus =  item.venus
        mars = item.mars
        jupiter = item.jupiter
        saturn = item.saturn
        uranus = item.uranus
        neptune = item.neptune
        pluto = item.pluto
        mean_node = item.mean_node
        true_node = item.true_node
        chiron =item.chiron
        pl = [sun,moon, mean_node, true_node, mercury, venus,mars,jupiter,saturn,uranus, neptune, pluto,chiron ]
                
        # Create a list of dictionaries
        

        for p in pl:
            if (item_t =="personality") & (p['name'] == "Sun"):
                sun_position_at_birth_dec = p['position']
            name = p['name']
            sign = p['sign']
            pos_dec = p['position']
            position_dms = decdeg2dms(p['position'])
            degree = position_dms[0]
            minutes = position_dms[1]
            seconds = round(position_dms[2])

            data = {
                'item' :item_t,
                'Name': name,
                'Sign': sign,
                'Pos_dec' : pos_dec,
                'Degree': degree,
                'Minutes': minutes,
                'Seconds': seconds
            }
            data_list.append(data)


        # planet_positions[planet] = {"sign": planet_sign, "degree": planet_degree}
        #     # Calculate South Node and Earth positions based on North Node and Sun positions
        # sn_sign, sn_degree = get_sign_and_degree((planet_positions['North Node']['degree'] + 180) % 360)
        # planet_positions['South Node'] = {"sign": sn_sign, "degree": sn_degree}
        # data = {
        #         'item' :item_t,
        #         'Name': name,
        #         'Sign': sign,
        #         'Pos_dec' : pos_dec,
        #         'Degree': degree,
        #         'Minutes': minutes,
        #         'Seconds': seconds
        #     }
        #     data_list.append(data)
        # earth_sign, earth_degree = get_sign_and_degree((planet_positions['Sun']['degree'] + 180) % 360)
        # planet_positions['Earth'] = {"sign": earth_sign, "degree": earth_degree}

        position_dms = decdeg2dms((sun['position']+180)%360)
        degree = position_dms[0]
        minutes = position_dms[1]
        seconds = round(position_dms[2])
        data = {
                'item' :item_t,
                'Name': "earth",
                'Sign': None,
                'Pos_dec' : (sun['position']+180)%360,
                'Degree': degree,
                'Minutes': minutes,
                'Seconds': seconds
            }
        data_list.append(data)
    # Create a DataFrame from the list of dictionaries
    df_kerykeion = pd.DataFrame(data_list)

    # Print the DataFrame
    print("df_kerykeion")
    print(df_kerykeion)
    # print(sun_position_at_birth_dec)
    return df_kerykeion

def generate_report(df):
    """Generates the report as dataframe.

    Args:
        df (_type_): _description_
    """    
    gateOrder = [41, 19, 13, 49, 30, 55, 37, 63, 22, 36, 25, 17, 21, 51, 42, 3, 27, 24, 2, 23, 8, 20, 16, 35, 45, 12, 15, 52, 39, 53, 62, 56, 31, 33, 7, 4, 29, 59, 40, 64, 47, 6, 46, 18, 48, 57, 32, 50, 28, 44, 1, 43, 14, 34, 9, 5, 26, 11, 10, 58, 38, 54, 61, 60]
    
    allSigns =  ["Ari", "Tau", "Gem", "Can", "Leo", "Vir", "Lib", "Sco", "Sag", "Cap", "Aqu", "Pis"]
    data_list = []
    
    for i in range(len(df)):
        item = df.iloc[i]["item"]
        planet = df.iloc[i]["Name"]
        #decimalDegrees = df.iloc[i]["Pos_dec"]
        decimalDegrees = df.iloc[i]["Degree"]  +  df.iloc[i]["Minutes"] / 60  +  df.iloc[i]["Seconds"] / 3600
        sign = df.iloc[i]["Sign"]

        # https://github.com/jdempcy/hdkit/blob/main/sample-apps/v1/substructure.js
        signDegrees =0
        for iteratingSign in allSigns:
            if iteratingSign == sign:
                break  
            signDegrees += 30  

        signDegrees += decimalDegrees

        # // Human Design gates start at Gate 41 at 02º00'00" Aquarius, so we have to adjust from 00º00'00" Aries.
		# // The distance is 58º00'00" exactly.
        signDegrees += 58
        if (signDegrees > 360):
            signDegrees -= 360
        percentageThrough = (signDegrees / 360)
        gate = gateOrder[int(percentageThrough * 64)] # in the app is 1 decimal
        
        exactLine = 384 * percentageThrough
        line = (exactLine % 6) + 1

        # Color
        exactColor = 2304 * percentageThrough
        color = (exactColor % 6) + 1

        # Tone
        exactTone = 13824 * percentageThrough
        tone = (exactTone % 6) + 1

        # Base
        exactBase = 69120 * percentageThrough # e.g. 46151
        base = (exactBase % 5) + 1
        data = {
                'item' :item,
                'Planet': planet,
                'Gate': gate,
                'Line' : int(line),
                'Color': int(color),
                'Tone': int(tone),
                'Base': int(base),
                'ExcactLine' : int(exactLine),
                'ExcactColor': int(exactColor),
                'ExcactTone': int(exactTone),
                'ExcactBase': int(exactBase)
            }

        data_list.append(data)

    # Create a DataFrame from the list of dictionaries
    df_result = pd.DataFrame(data_list)
    print ("------")
    print ("df_result")
    print (df_result)
    make_chart(df_result)

def make_chart(df_result):
    """Makes the chart. If the number is surrounde by bracket, that number is in a colored circle. If not, it's not

    Args:
        df_result (_type_): _description_
    """

    gates_chart= df_result["Gate"].to_list()
   
    gates= [[64,61,63], [47,24,4,17,11,43],[6223,56,35,12,45,33,8,31,20,16],
            [1,13,25,46,2,15,10,7],[21,40,26,51],[48,57,44,50,32,29,18],
            [59,9,3,42,27,34,5,14,29],[36,22,37,6,49,55,30],[41,39,19,52,60,53,54,38,58]]
    for g in gates:
        for h in g:
            if h in gates_chart:
                print (f"[{h}]", end = " ")
            else:
                print (h , end = " ")
        print("")

def main():
    y,m,d,h,min = 1983, 9, 25, 20, 48
    birthplace = "Malden, MA (US)"
    birthcountry = "US"
    delta = 7
    df_kerykeion = main_kerykeion(y,m,d,h,min, birthplace, birthcountry, delta)
    generate_report(df_kerykeion)
    # expected output 
    # https://github.com/jdempcy/hdkit/blob/main/sample-apps/v1/sample-data/charts/jonah-dempcy-chart.js
def mainx():
    
    #y,m,d,h,min = 1983, 9, 25, 20, 48
    y,m,d,h,min = 1983, 9, 26, 3, 48

    

    # Create a timescale and ask the current time.
    ts = load.timescale()
    t = ts.now()

    # Load the JPL ephemeris DE421 (covers 1900-2050).
    planets = load('de421.bsp')
    earth, mars = planets['earth'], planets['mars']

    # What's the position of Mars, viewed from Earth?
    astrometric = earth.at(t).observe(mars)
    ra, dec, distance = astrometric.radec()

    print(ra)
    print(dec)
    print(distance)
    
if __name__ == "__main__":
    main()
    mainx()