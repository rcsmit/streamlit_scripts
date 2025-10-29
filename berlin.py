import pandas as pd
from pathlib import Path
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from folium import plugins

from xml.etree import ElementTree as ET
import pandas as pd
import re
try:
    st.set_page_config(
        page_title="BerlinMap",
        layout="wide"
    )
except:
    pass



def load_text(filename: str) -> str:
    return Path(filename).read_text(encoding="utf-8", errors="ignore")



def load_kml():
    # Parse the provided KML content into a DataFrame named df_ with columns:
    # LAYER, NAME, LAT, LON, REMARKS. Save it as a CSV and display it.


    #kml_text = r"""<?xml version="1.0" encoding="UTF-8"?> <kml xmlns="http://www.opengis.net/kml/2.2">   <Document>     <name>Berlijn 2025</name>     <description/>     <Style id="icon-1899-0288D1-labelson-nodesc">       <IconStyle>         <color>ffd18802</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Style id="icon-1899-097138-nodesc-normal">       <IconStyle>         <color>ff387109</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <LabelStyle>         <scale>0</scale>       </LabelStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Style id="icon-1899-097138-nodesc-highlight">       <IconStyle>         <color>ff387109</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <LabelStyle>         <scale>1</scale>       </LabelStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <StyleMap id="icon-1899-097138-nodesc">       <Pair>         <key>normal</key>         <styleUrl>#icon-1899-097138-nodesc-normal</styleUrl>       </Pair>       <Pair>         <key>highlight</key>         <styleUrl>#icon-1899-097138-nodesc-highlight</styleUrl>       </Pair>     </StyleMap>     <Style id="icon-1899-7CB342-labelson-nodesc">       <IconStyle>         <color>ff42b37c</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Style id="icon-1899-9C27B0-labelson">       <IconStyle>         <color>ffb0279c</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>     </Style>     <Style id="icon-1899-9C27B0-labelson-nodesc">       <IconStyle>         <color>ffb0279c</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Style id="icon-1899-E65100-labelson-nodesc">       <IconStyle>         <color>ff0051e6</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Style id="icon-1899-FFD600-labelson-nodesc">       <IconStyle>         <color>ff00d6ff</color>         <scale>1</scale>         <Icon>           <href>https://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png</href>         </Icon>         <hotSpot x="32" xunits="pixels" y="64" yunits="insetPixels"/>       </IconStyle>       <BalloonStyle>         <text><![CDATA[<h3>$[name]</h3>]]></text>       </BalloonStyle>     </Style>     <Folder>       <name><![CDATA[VEGAN & FOOD]]></name>       <Placemark>         <name>SOY</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4114422,52.5260438,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name><![CDATA[Lia's Kitchen - 100% Vegan]]></name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4137897,52.4914244,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>1990 Vegan Living</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.458578,52.5103695,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>li.ke : serious||thai||vegan</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4586932,52.5115871,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Vegan Vibes Berlin</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4585686,52.5132337,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Chay Long Hasenheide</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4107326,52.4889667,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>doen doen® kebap</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.455839,52.5099086,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Bonvivant Cocktail Bistro</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3537681,52.4946487,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Mamas Banh | Schöneberg</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3543674,52.4898478,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Klunkerkranich</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4318524,52.482195,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>FALCO´SLICE</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4419715,52.4978689,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Vöner</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4686933,52.506714,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Markthalle Neun</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4315988,52.502135,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Hako Ramen Kreuzberg</name>         <styleUrl>#icon-1899-7CB342-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3903131,52.489484,0           </coordinates>         </Point>       </Placemark>     </Folder>     <Folder>       <name>INFO</name>       <Placemark>         <name>Kiez Hostel Berlin</name>         <styleUrl>#icon-1899-FFD600-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4475397,52.509766,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Woning Remon / Elektro Reibsch Andreas König</name>         <styleUrl>#icon-1899-FFD600-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4609751,52.5084331,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Ostbahnhof</name>         <styleUrl>#icon-1899-FFD600-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4354247,52.5098945,0           </coordinates>         </Point>       </Placemark>     </Folder>     <Folder>       <name>Friedrichshein</name>       <Placemark>         <name>RAW-Gelände</name>         <description>outdoor bars, graffiti, live DJs</description>         <styleUrl>#icon-1899-9C27B0-labelson</styleUrl>         <Point>           <coordinates>             13.452028,52.5076567,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Boxhagener Platz</name>         <description>fkeemarket sunday</description>         <styleUrl>#icon-1899-9C27B0-labelson</styleUrl>         <Point>           <coordinates>             13.4597132,52.5108764,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Volkspark Friedrichshain</name>         <styleUrl>#icon-1899-9C27B0-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4363934,52.5280353,0           </coordinates>         </Point>       </Placemark>     </Folder>     <Folder>       <name><![CDATA[East side and Berlin Wall & HISTORY]]></name>       <Placemark>         <name>East Side Gallery</name>         <styleUrl>#icon-1899-E65100-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4396953,52.5050224,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Checkpoint Charlie</name>         <styleUrl>#icon-1899-E65100-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3903913,52.5074434,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Topographie des Terrors</name>         <styleUrl>#icon-1899-E65100-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3837099,52.506747,0           </coordinates>         </Point>       </Placemark>     </Folder>     <Folder>       <name>MITTE</name>       <Placemark>         <name>Dom van Berlijn</name>         <styleUrl>#icon-1899-0288D1-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.401078,52.5190608,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Unter Den Linden</name>         <styleUrl>#icon-1899-0288D1-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3891129,52.517171,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Brandenburger Tor</name>         <styleUrl>#icon-1899-0288D1-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.377704,52.5162746,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>Reichstag</name>         <styleUrl>#icon-1899-0288D1-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.3761871,52.5186202,0           </coordinates>         </Point>       </Placemark>       <Placemark>         <name>HÖR</name>         <styleUrl>#icon-1899-0288D1-labelson-nodesc</styleUrl>         <Point>           <coordinates>             13.4117694,52.488833,0           </coordinates>         </Point>       </Placemark>     </Folder>     <Folder>       <name>ALTERNATIVE BERLIN</name>       <Placemark>         <name>Tempelhofer Feld</name>         <styleUrl>#icon-1899-097138-nodesc</styleUrl>         <Point>           <coordinates>             13.4005926,52.4748569,0           </coordinates>         </Point>       </Placemark>     </Folder>   </Document> </kml>   """
    kml_text = load_text(r"C:\Users\rcxsm\Downloads\Berlijn 2025.kml")
    ns = {"kml": "http://www.opengis.net/kml/2.2"}

    root = ET.fromstring(kml_text)

    rows = []
    for folder in root.findall(".//kml:Folder", ns):
        layer = folder.find("kml:name", ns)
        layer_name = layer.text if layer is not None else ""
        # Normalize whitespace in layer_name
        if layer_name:
            layer_name = re.sub(r"\s+", " ", layer_name.strip())
        for pm in folder.findall("kml:Placemark", ns):
            name_el = pm.find("kml:name", ns)
            name = name_el.text if name_el is not None else ""
            if name:
                name = re.sub(r"\s+", " ", name.strip())
            desc_el = pm.find("kml:description", ns)
            desc = desc_el.text if desc_el is not None else ""
            if desc:
                desc = re.sub(r"\s+", " ", desc.strip())
            coords_el = pm.find(".//kml:Point/kml:coordinates", ns)
            lat = lon = None
            if coords_el is not None and coords_el.text:
                coord_text = re.sub(r"\s+", "", coords_el.text)
                # KML: lon,lat,alt
                parts = coord_text.split(",")
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                    except ValueError:
                        lon = lat = None
            rows.append({
                "Layer": layer_name,
                "Name": name,
                "LAT": lat,
                "LON": lon,
                "remarks": desc
            })

    df_ = pd.DataFrame(rows, columns=["Layer", "Name", "LAT", "LON", "remarks"])


    return df_


def main():
    data = [
        ["Oct 15 - Arrival (Friedrichshain)", "Simon-Dach-Straße (dining street)", 52.508307, 13.455119,''],
        ["Oct 15 - Arrival (Friedrichshain)", "RAW-Gelände", 52.507200, 13.454400,''],
        ["Oct 16 - East Side & Wall", "East Side Gallery", 52.504997, 13.439605,''],
        ["Oct 16 - East Side & Wall", "Oberbaumbrücke", 52.501200, 13.440700,''],
        ["Oct 16 - East Side & Wall", "Topography of Terror", 52.506687, 13.383505,''],
        ["Oct 16 - East Side & Wall", "Checkpoint Charlie", 52.507431, 13.390271,''],
        ["Oct 17 - Museum Island & Mitte", "Alexanderplatz", 52.521992, 13.413244,''],
        ["Oct 17 - Museum Island & Mitte", "Museum Island (UNESCO)", 52.516640, 13.402318,''],
        ["Oct 17 - Museum Island & Mitte", "Berlin Cathedral (Berliner Dom)", 52.518898, 13.401797,''],
        ["Oct 17 - Museum Island & Mitte", "Brandenburg Gate", 52.516266, 13.377775,''],
        ["Oct 17 - Museum Island & Mitte", "Reichstag Building", 52.518589, 13.376665,''],
        ["Oct 18 - Alternative Berlin", "Markthalle Neun", 52.501987, 13.431830,''],
        ["Oct 18 - Alternative Berlin", "Tempelhofer Feld (park on former airport)", 52.475545, 13.401893,''],
        ["Oct 19 - History & Politics", "Memorial to the Murdered Jews of Europe", 52.513943, 13.378155,''],
        ["Oct 19 - History & Politics", "German Historical Museum (DHM)", 52.518060, 13.396940,''],
        ["Oct 19 - History & Politics", "DDR Museum", 52.519226, 13.402517,''],
        ["Oct 19 - History & Politics", "Tiergarten (Großer Tiergarten)", 52.513988, 13.358462,''],
        ["Oct 19 - History & Politics", "Victory Column (Siegessäule)", 52.514534, 13.349862,''],
        ["Oct 19 - History & Politics", "Prater Biergarten", 52.540210, 13.409550,''],
        ["Oct 19 - History & Politics", "Kanaan (Israeli–Palestinian restaurant)", 52.543611, 13.419845,''],
        ["Oct 20 - Markets & Farewell", "Boxhagener Platz Flea Market", 52.510806, 13.459694,''],
        ["Oct 20 - Markets & Farewell", "Admiralbrücke (Landwehrkanal sunset spot)", 52.495269, 13.415164,''],
        ["Oct 20 - Markets & Farewell", "Berliner Unterwelten (Museum/tours)", 52.547812, 13.389246,''],
        ["Oct 20 - Markets & Farewell", "Monkey Bar (view over Zoo)", 52.505566, 13.337916,''],
        ["Oct 20 - Markets & Farewell", "Michelberger Hotel", 52.506035, 13.448735,''],
               
        ["Oct 17 - Museum Island & Mitte", "HÖR",13.4117694 ,52.488833,""],
        ["VEGAN", "SOY",13.4114422 ,52.5260438,""],
        ["VEGAN", "Lia's Kitchen - 100% Vegan",13.4137897 ,52.49142440000001,"Goedkope cocktails. Vegan food"],
        ["VEGAN", "1990 Vegan Living",13.458578 ,52.5103695,""],
        ["VEGAN", "li.ke : serious||thai||vegan",13.4586932 ,52.5115871,""],
        ["VEGAN", "Vegan Vibes Berlin",13.4585686 ,52.5132337,""],
        ["VEGAN", "Chay Long Hasenheide",13.4107326 ,52.4889667,"vegan hotpot"],
        ["VEGAN", "doen doen® kebap",13.455839 ,52.5099086,""],
        ["VEGAN", "Bonvivant Cocktail Bistro",13.3537681 ,52.4946487,""],
        ["VEGAN", "Mamas Banh | Schöneberg",13.3543674 ,52.4898478,""],
        ["VEGAN", "Klunkerkranich,POINT", 13.4318524, 52.482195, "bar with sunsetview, hippievibes"],
        ["VEGAN", "FALCO SLICE",13.4419715 ,52.4978689,""],
        ["INFO","Kiez Hostel Berlin", 13.4475397,52.509766,""],
        ["INFO","Woning Remon", 13.4609751, 52.5084331,""], 
        ["INFO","Ostbahnhof", 13.4354247, 52.50989449999999,""],

    ]





    st.header("Berlin map of Rene")
    #df_ = pd.DataFrame(data, columns=["Layer", "Name", "LAT", "LON","remarks"])
    df_ = load_kml()

    st.write(df_)
    df_=df_.fillna("")
    df_["LAT"] = df_["LAT"].astype(float)
    df_["LON"] = df_["LON"].astype(float)
    layer_list = df_['Layer'].unique().tolist()

    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list,layer_list)

    if layers_to_show == []:
        st.error("Choose a layer")
        st.stop()

    attribution= "CartoDB Positron"
    location = [  52.508307, 13.455119]
    # location = [ df_["LAT"].mean(), df_["LON"].mean()]
    m = folium.Map(location=location, zoom_start=12,  tiles = "CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)

    #kleur = ['red', 'green', 'yellow', 'purple', 'blue', 'pink','orange']
    kleur = [ '#00FF00','#FF0000', '#FFFF00', '#800080', '#0000FF', '#FFC0CB', '#FFA500','#008800',]

    for i,l in enumerate(layers_to_show):
        df = df_[df_['Layer'] == l]
        
        # create a marker cluster 
        # https://python-visualization.github.io/folium/latest/user_guide/plugins/marker_cluster.html
        marker_cluster = MarkerCluster(disableClusteringAtZoom=5).add_to(m)
        
        for index, row in df.iterrows():
            if row["remarks"] != "None":
                remarks_ = row["remarks"]
            else:
                remarks_ = " "
                
            depot_node = (row["LAT"], row["LON"]) 
            remarks = f'<b>{row["Name"]}</b><br><i>{remarks_}</i>'# <br><br>{row["city__"]}<br>{row["provincie"]}<br>{row["country__"]}<br>{row["continent"]}<br>{depot_node}'

            # folium.CircleMarker(location=depot_node,
            #                         radius=3,    
            #                         color=[kleur[i],''],
            #                         fill_color =[kleur[i],''],
            #                         fill_opacity=0.7,
            #                         ).add_to(marker_cluster)

            folium.CircleMarker(
                                    location=depot_node,
                                    radius=6,                     # a bit larger so color is visible
                                    color=kleur[i],               # single hex string
                                    fill=True,                    # important
                                    fill_color=kleur[i],
                                    fill_opacity=0.9,
                                    weight=2,
                                ).add_to(marker_cluster)
            # folium.map.Marker(depot_node,
            #                 icon=DivIcon(
            #                     icon_size=(30,30),
            #                     icon_anchor=(5,14),
            #                     html=f'<div style="font-size: 10pt">%s</div>' % row["Name"],
            #                 ),tooltip=remarks
            #                 ).add_to(marker_cluster)  

            label_html = f"""<div style="
                                font-size:8pt;
                                white-space:nowrap;      /* stop wrapping */
                                background:rgba(255,255,255,0.2);
                                padding:1px 2px;
                                border-radius:2px;
                                box-shadow:0 0 2px rgba(0,0,0,0.1);
                                ">
                                {row["Name"]}
                                </div>"""

            folium.Marker(
                        location=depot_node,
                        icon=DivIcon(
                            icon_size=(150, 24),        # give it room
                            icon_anchor=(5, 12),
                            html=label_html,
                        ),
                        tooltip=folium.Tooltip(remarks, sticky=True)
                    ).add_to(marker_cluster)

        st.markdown("""
                    <style>
                    .big-font {
                        font-size:30px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                            
        text= f"<font  class='big-font'  color={kleur[i]}>•</font> - {l}"
        st.sidebar.write(text, unsafe_allow_html=True)
            
        # Display the map in Streamlit
        # call to render Folium map in Streamlit
    st_data = st_folium(m, width=1500, returned_objects=[])
 

if __name__ == "__main__":
    main()

