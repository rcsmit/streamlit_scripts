import pandas as pd
import numpy as np
import pydeck as pdk
import streamlit as st
import datetime as dt

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
        "img_knmi/weerstations.csv",
        #"img_knmi/leeg.csv",
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
    st.image("img_knmi/doorlopend_per_dag.png")

    st.subheader("Aantal keren")
    st.write("Hoeveel tropische dagen hebben we gehad in een bepaaalde periode?")
    st.image("img_knmi/aantal_keren.png")

    st.subheader("Specifieke dag")
    st.write("Welke temperatuur was het op nieuwjaarsdag door de loop van de tijd?")
    st.image("img_knmi/specifieke_dag.png")

    st.subheader("Jaargemiddelde")
    st.write("Wat was het jaargemiddelde?")
    st.image("img_knmi/jaargemiddelde.png")
    st.write("Kies hier volledige jaren als periode")

    st.subheader("Per dag in div jaren")
    st.write("Kan ik 2021 met 2021 per dag vergelijken?")
    st.image("img_knmi/per_dag_div_jaren_2020_2021.png")

    st.subheader("Per maand in diverse jaren")
    st.write("Kan ik 2021 met 2021 per maaand vergelijken?")
    st.image("img_knmi/per_maand_div_jaren_2020_2021.png")

    st.subheader("Percentiles")
    st.write("Wat zijn de uitschieters in het jaar? - kies hiervoor een lange periode")
    st.image("img_knmi/percentiles.png")
    st.subheader("Weerstations")
    st.write("Link to map with KNMI stations https://www.google.com/maps/d/u/0/edit?mid=1ePEzqJ4_aNyyTwF5FyUM6XiqhLZPSBjN&ll=52.17534745851063%2C5.197922250000001&z=7")
    st.image("img_knmi/weerstations.png")

                
    st.image(
            "https://raw.githubusercontent.com/rcsmit/COVIDcases/main/buymeacoffee.png"
        )

    st.markdown(
        '<a href="https://www.buymeacoffee.com/rcsmit" target="_blank">If you are happy with this dashboard, you can buy me a coffee</a>',
        unsafe_allow_html=True,
    )