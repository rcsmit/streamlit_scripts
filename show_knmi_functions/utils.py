import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
from skmisc.loess import loess

# C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\show_knmi_functions\utils.py:8: UserWarning: registration of accessor <class 'show_knmi_functions.utils.LoessAccessor'> under name 'loess' for type <class 'pandas.core.series.Series'> is overriding a preexisting attribute with the same name.
 # @pd.api.extensions.register_series_accessor("loess")
# https://stackoverflow.com/questions/69720999/how-to-prevent-pandas-accessor-to-issue-override-warning
try:
    #delete the accessor to avoid warning 
    del pd.DataFrame.loess
except AttributeError:
    pass

# Define the pandas accessor
@pd.api.extensions.register_series_accessor("loess")
class LoessAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def apply(self, ybounds=None, it=1):
        t = np.arange(len(self._obj))
        y = self._obj.values
        _, loess_values, ll, ul = loess_skmisc(t, y, ybounds, it)
        return loess_values


@st.cache_data (ttl=60 * 60 * 24)
def get_data(url):
    header = None
    print (url)
    with st.spinner(f"GETTING ALL DATA ... {url}"):
        # url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        # url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        try:
        
            df = pd.read_csv(
                url,
                delimiter=",",
                header= header,
                comment="#",
                low_memory=False,
            )

        except:
            st.error("Error reading data")
            st.stop()

        
        # TG        : Etmaalgemiddelde temperatuur (in 0.1 graden Celsius) / Daily mean temperature in (0.1 degrees Celsius)
        # TN        : Minimum temperatuur (in 0.1 graden Celsius) / Minimum temperature (in 0.1 degrees Celsius)
        # TNH       : Uurvak waarin TN is gemeten / Hourly division in which TN was measured
        # TX        : Maximum temperatuur (in 0.1 graden Celsius) / Maximum temperature (in 0.1 degrees Celsius)
        # TXH       : Uurvak waarin TX is gemeten / Hourly division in which TX was measured
        # T10N      : Minimum temperatuur op 10 cm hoogte (in 0.1 graden Celsius) / Minimum temperature at 10 cm above surface (in 0.1 degrees Celsius)
        # T10NH     : 6-uurs tijdvak waarin T10N is gemeten / 6-hourly division in which T10N was measured; 6=0-6 UT; 12=6-12 UT; 18=12-18 UT; 24=18-24 UT
        # SQ        : Zonneschijnduur (in 0.1 uur) berekend uit de globale straling (-1 voor <0.05 uur) / Sunshine duration (in 0.1 hour) calculated from global radiation (-1 for <0.05 hour)
        # SP        : Percentage van de langst mogelijke zonneschijnduur / Percentage of maximum potential sunshine duration
        # Q         : Globale straling (in J/cm2) / Global radiation (in J/cm2)
        # DR        : Duur van de neerslag (in 0.1 uur) / Precipitation duration (in 0.1 hour)
        # RH        : Etmaalsom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm) / Daily precipitation amount (in 0.1 mm) (-1 for <0.05 mm)
        # UN        : Minimale relatieve vochtigheid (in procenten)
        # UX        : Maximale relatieve vochtigheid (in procenten)
        # EV24      : Referentiegewasverdamping (Makkink) (in 0.1 mm) / Potential evapotranspiration (Makkink) (in 0.1 mm)
        # FHX       : Hoogste uurgemiddelde windsnelheid (in 0.1 m/s)
        #  0  1          2    3    4  5   6     7   8   9    10  11  12 3   4  5  16     7    8  9 20  
        # STN,YYYYMMDD,DDVEC,FHVEC,FG,FHX,FHXH,FHN,FHNH,FXX,FXXH,TG,TN,TNH,TX,TXH,T10N,T10NH,SQ,SP,Q,
        # 21 22  3   4    5   6 7  8  9   30   1    2   3   4  5  6
        # DR,RH,RHX,RHXH,PG,PX,PXH,PN,PNH,VVN,VVNH,VVX,VVXH,NG,UG,UX,UXH,UN,UNH,EV24

        # 0   1           2     3     4    5       6     7      8    9    100    11   12
        # STN,YYYYMMDD,   TG,   TN,   TX, T10N,   SQ,   SP,    Q,   DR,   RH,   UN,   UX
        column_replacements = [
            [0, "STN"],
            [1, "YYYYMMDD"],
            [2, "temp_avg"],
            [3, "temp_min"],
            [4, "temp_max"],
            [5, "T10N"],
            [6, "zonneschijnduur"],
            [7, "perc_max_zonneschijnduur"],
            [8, "glob_straling"],
            [9, "neerslag_duur"],
            [10, "neerslag_etmaalsom"],
            [11, "RH_min"],
            [12, "RH_max"],
            [13, "EV24"],
            [14, "wind_max"]
        ]
   
        for c in column_replacements:
            df = df.rename(columns={c[0]: c[1]})
        
        df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
        df["YYYY"] = df["YYYYMMDD"].dt.year
        df["MM"] = df["YYYYMMDD"].dt.month
        df["DD"] = df["YYYYMMDD"].dt.day
        df["dayofyear"] = df["YYYYMMDD"].dt.dayofyear
        df["count"] = 1
        month_long_to_short = {
            "January": "Jan",
            "February": "Feb",
            "March": "Mar",
            "April": "Apr",
            "May": "May",
            "June": "Jun",
            "July": "Jul",
            "August": "Aug",
            "September": "Sep",
            "October": "Oct",
            "November": "Nov",
            "December": "Dec",
        }
        month_number_to_short = {
            "1": "Jan",
            "2": "Feb",
            "3": "Mar",
            "4": "Apr",
            "5": "May",
            "6": "Jun",
            "7": "Jul",
            "8": "Aug",
            "9": "Sep",
            "10": "Oct",
            "11": "Nov",
            "12": "Dec",
        }
        df["month"] = df["MM"].astype(str).map(month_number_to_short)
        df["year"] = df["YYYY"].astype(str)
        df["month"] = df["month"].astype(str)
        df["day"] = df["DD"].astype(str)
        df["month_year"] = df["month"] + " - " + df["year"]
        df["year_month"] = df["year"] + " - " +  df["MM"].astype(str).str.zfill(2)
        df["month_day"] = df["month"] + " - " + df["day"]
        
        to_divide_by_10 = [
            "temp_avg",
            "temp_min",
            "temp_max",
            "zonneschijnduur",
            "neerslag_duur",
            "neerslag_etmaalsom",
            "EV24",
            "wind_max"
        ]
        df["glob_straling"] = pd.to_numeric(df["glob_straling"], errors='coerce')
        df['neerslag_etmaalsom'].replace(" ", 0)
       
        for d in to_divide_by_10:
            try:
                df[d] = pd.to_numeric(df[d], errors='coerce')
                try:   
                    df[d] = df[d] / 10
                except:
                    df[d] = df[d]
            except:
                print(f"[{d}] doesnt exist in this dataframe")

    df["spec_humidity_knmi_derived"] = df.apply(lambda x: rh2q(x['RH_min'],x['temp_max'], 1020),axis=1)
    df["abs_humidity_knmi_derived"] =df.apply(lambda x: rh2ah(x['RH_min'],x['temp_max']),axis=1)
    df["globale_straling_log10"] =df.apply(lambda x: log10(x['glob_straling']),axis=1) #  np.log10(df["glob_straling"])
    mask = (df['neerslag_duur'].notna()) & (df['neerslag_duur'].ne(0))
    df.loc[mask, 'neerslag_etmaalsom_div_duur'] = df.loc[mask, 'neerslag_etmaalsom'] / df.loc[mask, 'neerslag_duur']     
    df['neerslag_etmaalsom'] = df['neerslag_etmaalsom'].replace(-0.1, 0)
    #df['gevoelstemperatuur'] = df.apply(feels_like_temperature, axis=1)

    df['gevoelstemperatuur_avg'] = df.apply(feels_like_temperature, axis=1, temp_type="temp_avg")
    df['gevoelstemperatuur_max'] = df.apply(feels_like_temperature, axis=1, temp_type="temp_max")
    return df


def date_to_daynumber(date_str):
    """
    Convert a date in "dd-mm" format to the day number of the year.

    Args:
        date_str (str): The date string in "dd-mm" format.

    Returns:
        int or str: The day number of the year, or an error message if the date is not valid.
    """
    # Dictionary with the number of days in each month (non-leap year)
    days_in_month = {
        '01': 31, '02': 28, '03': 31, '04': 30, '05': 31, '06': 30,
        '07': 31, '08': 31, '09': 30, '10': 31, '11': 30, '12': 31
    }

    try:
        day, month = date_str.split('-')
        day = int(day)
        if month not in days_in_month:
            return "Invalid date: The month is not valid."
        max_days = days_in_month[month]
        if day < 1 or day > max_days:
            return "Invalid date: The day is not valid."

        # Calculate the day number
        day_number = sum(days_in_month[m] for m in list(days_in_month.keys())[:list(days_in_month.keys()).index(month)]) + day

        return day_number
    except ValueError:
        st.error  ("Invalid date: The date format should be 'dd-mm'.")
        st.stop()

def rh2q(rh, t, p ):
    """Compute the Specific Humidity (Bolton 1980):
            e = 6.112*exp((17.67*Td)/(Td + 243.5));
            q = (0.622 * e)/(p - (0.378 * e));
                where:
                e = vapor pressure in mb;
                Td = dew point in deg C;
                p = surface pressure in mb;
                q = specific humidity in kg/kg.

            (Note the final specific humidity units are in g/kg = (kg/kg)*1000.0)

    Args:
        rh ([type]): rh min in percent
        t ([type]): temp max in deg C

    Returns:
        [type]: [description]
    """
    # https://archive.eol.ucar.edu/projects/ceop/dm/documents/refdata_report/eqns.html

    #Td = math.log(e/6.112)*243.5/(17.67-math.log(e/6.112))
    try:
        es = 6.112 * math.exp((17.67 * t)/(t + 243.5))
        e = es * (rh / 100)
        q_ = (0.622 * e)/(p - (0.378 * e)) * 1000
        x  = round(q_,2)
    except:
        x= None

    return x  

def rh2ah(rh, t ):
    """Relative humidity to absolute humidity via the equasion of Clausius-Clapeyron

    Args:
        rh ([type]): rh min
        t ([type]): temp max

    Returns:
        [type]: [description]
    """
    # return (6.112 * ((17.67 * t) / (math.exp(t) + 243.5)) * rh * 2.1674) / (273.15 + t )
    #  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7831640/
    try: 
        x= (6.112 * math.exp((17.67 * t) / (t + 243.5)) * rh * 2.1674) / (273.15 + t )
    except:
        x= None
    return  x


# Function to convert Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# Function to convert Fahrenheit to Celsius
def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

# Function to calculate Heat Index
def calculate_heat_index(T, RH):
    try:
        # https://wonder.cdc.gov/wonder/help/Climate/ta_htindx.PDF
        # Formula for heat index calculation in Fahrenheit
        HI = (-42.379 + 2.04901523 * T + 10.14333127 * RH 
            - 0.22475541 * T * RH - 0.00683783 * T**2 
            - 0.05481717 * RH**2 + 0.00122874 * T**2 * RH 
            + 0.00085282 * T * RH**2 - 0.00000199 * T**2 * RH**2)
    except:
        HI = None
    return HI

# Function to calculate Wind Chill
def calculate_wind_chill(T, V):
    # Formula for wind chill calculation in Fahrenheit
    # https://unidata.github.io/MetPy/v0.10/_static/FCM-R19-2003-WindchillReport.pdf

    WC = 35.74 + 0.6215 * T - 35.75 * (V**0.16) + 0.4275 * T * (V**0.16)
    return WC

# Function to determine the feels-like temperature
def feels_like_temperature(row, temp_type):
    # Dictionary to map temp_type to the corresponding column
    temp_mapping = {
        'temp_avg': 'temp_avg',
        'temp_max': 'temp_max'
    }

    # Get the temperature based on the temp_type
    T_C = row.get(temp_mapping.get(temp_type))
    if T_C is None:
        raise ValueError(f"Invalid temperature type: '{temp_type}'. Use 'temp_avg' or 'temp_max'.")

    # Calculate average relative humidity, considering missing values
    RH_min = row.get('RH_min')
    RH_max = row.get('RH_max')
    
    try:
        RH = (RH_min + RH_max) / 2
    except:
        #  unsupported operand type(s) for /: 'str' and 'int'
        RH = None

    # Convert wind speed from m/s to mph (default to 0 if wind_max is missing)
    V_mph = row.get('wind_max', 0) * 2.23694  # Default to 0 if wind_max is missing

    # Convert Celsius to Fahrenheit
    T_F = celsius_to_fahrenheit(T_C)

    # Determine feels-like temperature
    if T_F >= 80 and RH is not None:
        # Calculate Heat Index
        feels_like_F = calculate_heat_index(T_F, RH)
    elif T_F <= 50 and V_mph >= 3:
        # Calculate Wind Chill
        feels_like_F = calculate_wind_chill(T_F, V_mph)
    else:
        feels_like_F = T_F  # No adjustment

    # Convert back to Celsius
    feels_like_C = fahrenheit_to_celsius(feels_like_F)
    return feels_like_C


def log10(t):
    try:
        x = np.log10(t)
    except:
        x = None
    return x

def check_from_until(from_, until_):
    """Checks whether the start- and enddate are valid.

    Args:
        from_ (string): start date
        until_ (string): end date

    Returns:
        FROM, UNTIL : start- and end date in datetime

    """
    
    try:
        FROM = dt.datetime.strptime(from_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the startdate is in format yyyy-mm-dd")
        st.stop()

    try:
        UNTIL = dt.datetime.strptime(until_, "%Y-%m-%d").date()
    except:
        st.error("Please make sure that the enddate is in format yyyy-mm-dd")
        st.stop()

    if FROM >= UNTIL:
        st.warning("Make sure that the end date is not before the start date")
        st.stop()

    return FROM, UNTIL

def list_to_text(what_to_show_):
    """converts list to text to use in plottitle

    Args:
        what_to_show_ (list with strings): list with the fields

    Returns:
        string: text to use in plottitle
    """
    what_to_show_ = what_to_show_ if type(what_to_show_) == list else [what_to_show_]
    w = ""
    for w_ in what_to_show_:
        if w_ == what_to_show_[-1]:
            w += w_
        elif w_ == what_to_show_[-2]:
            w += w_ + " & "
        else:
            w += w_ + ", "

    return w


def find_date_for_title(day, month):
    months = [
        "januari",
        "februari",
        "maart",
        "april",
        "mei",
        "juni",
        "juli",
        "augustus",
        "september",
        "oktober",
        "november",
        "december",
    ]
    # ["January", "February",  "March", "April", "May", "June", "July", "August", "September", "Oktober", "November", "December"]
    return str(day) + " " + months[month - 1]

@st.cache_data (ttl=60 * 60 * 24)
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

def download_button(df):    
    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='df_knmi.csv',
        mime='text/csv',
    )


def get_weerstations():
    weer_stations = [
        [209, "IJmond"],
        [210, "Valkenburg Zh"],
        [215, "Voorschoten"],
        [225, "IJmuiden"],
        [235, "De Kooy"],
        [240, "Schiphol"],
        [242, "Vlieland"],
        [248, "Wijdenes"],
        [249, "Berkhout"],
        [251, "Hoorn Terschelling"],
        [257, "Wijk aan Zee"],
        [258, "Houtribdijk"],
        [260, "De Bilt"],
        [265, "Soesterberg"],
        [267, "Stavoren"],
        [269, "Lelystad"],
        [270, "Leeuwarden"],
        [273, "Marknesse"],
        [275, "Deelen"],
        [277, "Lauwersoog"],
        [278, "Heino"],
        [279, "Hoogeveen"],
        [280, "Eelde"],
        [283, "Hupsel"],
        [285, "Huibertgat"],
        [286, "Nieuw Beerta"],
        [290, "Twenthe"],
        [308, "Cadzand"],
        [310, "Vlissingen"],
        [311, "Hoofdplaat"],
        [312, "Oosterschelde"],
        [313, "Vlakte van De Raan"],
        [315, "Hansweert"],
        [316, "Schaar"],
        [319, "Westdorpe"],
        [323, "Wilhelminadorp"],
        [324, "Stavenisse"],
        [330, "Hoek van Holland"],
        [331, "Tholen"],
        [340, "Woensdrecht"],
        [343, "Rotterdam Geulhaven"],
        [344, "Rotterdam"],
        [348, "Cabauw Mast"],
        [350, "Gilze-Rijen"],
        [356, "Herwijnen"],
        [370, "Eindhoven"],
        [375, "Volkel"],
        [377, "Ell"],
        [380, "Maastricht"],
        [391, "Arcen"],
    ]
    return weer_stations

def show_weerstations():
    MAPBOX = "pk.eyJ1IjoicmNzbWl0IiwiYSI6Ii1IeExqOGcifQ.EB6Xcz9f-ZCzd5eQMwSKLQ"
    # original_Name
    df_map=  pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/weerstations.csv",
        comment="#",
        delimiter=",",
        low_memory=False,
    )
    df_map = df_map[["original_Name", "lat", "lon"]]

    # Adding code so we can have map default to the center of the data
    midpoint = (np.average(df_map['lat']), np.average(df_map['lon']))
    import pydeck as pdk
    tooltip = {
            "html":
                "{original_Name} <br/>"
            }
        
    layer1= pdk.Layer(
            'ScatterplotLayer',     # Change the `type` positional argument here
                df_map,
                get_position=['lon', 'lat'],
                auto_highlight=True,
                get_radius=4000,          # Radius is given in meters
                get_fill_color=[180, 0, 200, 140],  # Set an RGBA value for fill
                pickable=True)
    layer2 =  pdk.Layer(
                    type="TextLayer",
                    data=df_map,
                    pickable=False,
                    get_position=["lon", "lat"],
                    get_text="original_Name",
                    get_color=[0, 0, 0],
                    get_angle=0,
                    sizeScale= 0.5,
                    # Note that string constants in pydeck are explicitly passed as strings
                    # This distinguishes them from columns in a data set
                    getTextAnchor= '"middle"',
                    get_alignment_baseline='"bottom"'
                )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=pdk.ViewState(
             longitude=midpoint[1],
            latitude=midpoint[0],
            pitch=0,
            zoom=6,
        ),
        layers=[layer1, layer2
            
        ],tooltip = tooltip
    ))

    st.write(df_map)

    st.sidebar.write("Link to map with KNMI stations on Google Maps https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN&ll=52.17534745851063%2C5.197922250000001&z=7")

def help():
    st.header("Help")
    st.write ("Hier zijn de verschillende mogelijkheden")
    st.subheader("Doorlopend per dag")
    st.write("Wat was de temperatuur in de loop van de tijd?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/doorlopend_per_dag.png")

    st.subheader("Aantal keren")
    st.write("Hoeveel tropische dagen hebben we gehad in een bepaaalde periode?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/aantal_keren.png")

    st.subheader("Specifieke dag")
    st.write("Welke temperatuur was het op nieuwjaarsdag door de loop van de tijd?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/specifieke_dag.png")

    st.subheader("Last day")
    st.write ("Wanneer was het voor het laatst 0 graden in de afgelopen jaren.")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/lastday.png")

    st.subheader("Jaargemiddelde")
    st.write("Wat was het jaargemiddelde?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/jaargemiddelde.png")
    st.write("Kies hier volledige jaren als periode")

    st.subheader("Maandgemiddelde")
    st.write("Wat was het jaargemiddelde?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/maandgemiddelde.png")
    st.write("Kies hier volledige jaren als periode")

    st.subheader("Per dag in div jaren")
    st.write("Kan ik 2021 met 2021 per dag vergelijken?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/per_dag_div_jaren_2020_2021.png")

    st.subheader("Per maand in diverse jaren")
    st.write("Kan ik 2021 met 2021 per maaand vergelijken?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/per_maand_div_jaren_2020_2021.png")

    st.subheader("Spaghettiplot")
    st.write("Spaghettiplot. Laatste jaar en gemiddelde extra benadrukt")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/spaghettiplot.png")

    st.subheader("Percentiles")
    st.write("Wat zijn de uitschieters in het jaar? - kies hiervoor een lange periode")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/percentiles.png")
    
    st.subheader("Polorplot/radarchart")
    st.write("A polar plot")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/polarplot.png")
    

    st.subheader("Show year histogram animation")
    st.write("_")
    st.subheader("Does rain predict rain")
    st.write("reproducing https://medium.com/towards-data-science/does-rain-predict-rain-us-weather-data-and-the-correlation-of-rain-today-and-tomorrow-3a62eda6f7f7")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/rainpredict.png")
    
    st.subheader("Neerslagtekort")
    st.write("reproducing RIVM graph")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/neerslagtekort.png")

    st.subheader("Neerslagtekort meerdere")
    st.write("reproducing RIVM graph")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/neerslagtekortmeerdere.png")
    
    st.subheader("Weerstations")
    st.write("Link to map with KNMI stations https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN&ll=52.17534745851063%2C5.197922250000001&z=7")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/weerstations.png")

                
    st.image(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
        )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )


def loess_skmisc(t, y,  ybounds=None, it=1):
    """Make a plot with scikit-misc. Scikit-misc is the perfect reproduction of the method used by KNMI
    See https://github.com/rcsmit/streamlit_scripts/blob/main/loess_scikitmisc.py for the complete version. 
    See https://github.com/rcsmit/streamlit_scripts/blob/main/loess.py for a comparison of the various methods.

    Args:
        t : list of time values, increasing by 1.
        y : list of  values
        ybounds : list or array-like, optional
            Lower/upper bound on the value range of y (default: [-Inf, Inf]).
        it : number of iterations
     
    Returns:
        loess : list with the smoothed values
        ll : lower bounds
        ul : upper bounds

    span = 42/len(y), wat de 30 jarig doorlopend gemiddelde benadert
    https://www.knmi.nl/kennis-en-datacentrum/achtergrond/standaardmethode-voor-berekening-van-een-trend
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python

 

    # Set default value for ybounds
    if ybounds is None:
        ybounds = [-np.inf, np.inf]
    elif len(ybounds) != 2:
        ybounds = [-np.inf, np.inf]

    ybounds = sorted(ybounds)

    # Dimensions and checks
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dt = np.diff(t)[0]
    n = len(y)
    ig = ~np.isnan(y)
    yg = y[ig]
    tg = t[ig]
    ng = sum(ig)

    if ng <= 29:
        st.error("Insufficient valid data (less than 30 observations")
        st.stop()

        raise ValueError("Insufficient valid data (less than 30 observations).")

    # Check values of bounds
    if np.any(yg < ybounds[0]) or np.any(yg > ybounds[1]):
        raise ValueError("Stated bounds are not correct: y takes values beyond bounds.")

    span = 42/len(y)
    
    l = loess(t,y)
    
    # MODEL and CONTROL. Essential for replicating the results from the R script.
    #
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_model.html#skmisc.loess.loess_model
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_control.html#skmisc.loess.loess_control
   
    l.model.span = span
    l.model.degree = 1
    l.control.iterations = it # must be 1 for replicating the R-script
    l.control.surface = "direct"
    l.control.statistics = "exact"

    l.fit()
    pred = l.predict(t, stderror=True)
    conf = pred.confidence()
    #ste = pred.stderr
    loess_values = pred.values
    ll = conf.lower
    ul = conf.upper
    
    return t, loess_values, ll, ul