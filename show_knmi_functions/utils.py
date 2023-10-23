import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
from skmisc.loess import loess

def get_data(url):
    header = None
    with st.spinner(f"GETTING ALL DATA ... {url}"):
        # url =  "https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns=251&vars=TEMP&start=18210301&end=20210310"
        # https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script
        # url = f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens?stns={stn}&vars=ALL&start={fromx}&end={until}"
        try:
            df = pd.read_csv(
                url,
                delimiter=",",
                header=header,
                comment="#",
                low_memory=False,
            )

        except:
            st.write("FOUT BIJ HET INLADEN.")
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
        #  0  1          2    3    4  5   6     7   8   9    10  11  12 3   4  5  16     7    8  9 20  
        # STN,YYYYMMDD,DDVEC,FHVEC,FG,FHX,FHXH,FHN,FHNH,FXX,FXXH,TG,TN,TNH,TX,TXH,T10N,T10NH,SQ,SP,Q,
        # 21 22  3   4    5   6 7  8  9   30   1    2   3   4  5  6
        # DR,RH,RHX,RHXH,PG,PX,PXH,PN,PNH,VVN,VVNH,VVX,VVXH,NG,UG,UX,UXH,UN,UNH,EV24
        column_replacements_knmi_nw_beerta = [
            [0, "STN"],
            [1, "YYYYMMDD"],
            [11, "temp_avg"],
            [12, "temp_min"],
            [14, "temp_max"],
            [16, "T10N"],
            [18, "zonneschijnduur"],
            [19, "perc_max_zonneschijnduur"],
            [20, "glob_straling"],
            [21, "neerslag_duur"],
            [22, "neerslag_etmaalsom"],
            [38, "RH_min"],
            [36, "RH_max"]
        ]
        # 0   1           2     3     4    5       6     7      8    9    100    11   12
        # STN,YYYYMMDD,   TG,   TN,   TX, T10N,   SQ,   SP,    Q,   DR,   RH,   UN,   UX
        column_replacements_knmi = [
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
            [12, "RH_max"]
        ]
        column_replacements_local = [
            [0, "ID"],
            [1, "STN"],
            [2, "YYYYMMDD"],
            [3, "temp_avg"],
            [4, "temp_min"],
            [5, "temp_max"],
            [6, "T10N"],
            [7, "zonneschijnduur"],
            [8, "perc_max_zonneschijnduur"],
            [9, "glob_straling"],
            [10, "neerslag_duur"],
            [11, "neerslag_etmaalsom"],
            [12, "RH_min"],
            [13, "RH_max"]
        ]

        #column_replacements = column_replacements_local if platform.processor() else column_replacements_knmi
        column_replacements = column_replacements_knmi
        for c in column_replacements:
            df = df.rename(columns={c[0]: c[1]})
        # if platform.processor(): 
        #     df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"], format="%Y-%m-%d")
        # else:
        print (df.dtypes)
        print (df)
        
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
        ]
        
        #divide_by_10 = False if platform.processor() else True
        divide_by_10 = True
        if divide_by_10:
            for d in to_divide_by_10:
                try:
                    df[d] = df[d] / 10
                except:
                    df[d] = df[d]

    df["spec_humidity_knmi_derived"] = df.apply(lambda x: rh2q(x['RH_min'],x['temp_max'], 1020),axis=1)
    df["abs_humidity_knmi_derived"] =df.apply(lambda x: rh2ah(x['RH_min'],x['temp_max']),axis=1)
    df["globale_straling_log10"] =df.apply(lambda x: log10(x['glob_straling']),axis=1) #  np.log10(df["glob_straling"])
    
    return df



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

def log10(t):
    try:
        x = np.log10(t)
    except:
        x = None
    return x

def check_from_until(from_, until_):
    """Checks whethe start- and enddate are valid.

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

@st.cache_data 
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
        #"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/leeg.csv",
        comment="#",
        delimiter=",",
        low_memory=False,
    )
    df_map = df_map[["original_Name", "lat", "lon"]]
    #df_map = df_map.head(1)
    
    #st.map(data=df_map, zoom=None, use_container_width=True)

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

    st.subheader("Jaargemiddelde")
    st.write("Wat was het jaargemiddelde?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/jaargemiddelde.png")
    st.write("Kies hier volledige jaren als periode")

    st.subheader("Per dag in div jaren")
    st.write("Kan ik 2021 met 2021 per dag vergelijken?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/per_dag_div_jaren_2020_2021.png")

    st.subheader("Per maand in diverse jaren")
    st.write("Kan ik 2021 met 2021 per maaand vergelijken?")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/per_maand_div_jaren_2020_2021.png")

    st.subheader("Percentiles")
    st.write("Wat zijn de uitschieters in het jaar? - kies hiervoor een lange periode")
    st.image("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/img_knmi/percentiles.png")
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


    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python

    

    # # Set default values for p, t1, and t2
    # if p is None:
    #     p = 0.95  # default confidence level
    # if t1 is None or t2 is None:
    #     t1 = np.inf
    #     t2 = -np.inf

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
        raise ValueError("Insufficient valid data (less than 30 observations).")

    # Check values of bounds
    if np.any(yg < ybounds[0]) or np.any(yg > ybounds[1]):
        raise ValueError("Stated bounds are not correct: y takes values beyond bounds.")

    # Averages over 30 time-steps
    avt, avy, avysd = None, None, None
    if ng > 29:
        avt = tg + dt / 2  # time (end of time-step, for 30-year averages)
        avy = np.convolve(yg, np.ones(30) / 30, mode="valid")
        avy2 = np.convolve(yg**2, np.ones(30) / 30, mode="valid")
        avysd = np.sqrt(avy2 - avy**2)
        ind = slice(
            15, ng - 14
        )  # was (15, ng-15) but gives an error, whether the df has an even or uneven length
        # [ValueError: x and y must have same first dimension, but have shapes (92,) and (93,)]
        avt = avt[ind]
        # avy = avy[ind]            # takes away y values, gives error
        # [ValueError: x and y must have same first dimension, but have shapes (93,) and (78,)]
        avysd = avysd[ind]


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