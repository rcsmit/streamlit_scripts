import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from folium import plugins


try:
    st.set_page_config(page_title="Renato in Rome", layout="wide")
except:
    pass


from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
import pandas as pd
import streamlit as st


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns
    https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.sidebar.checkbox("Search / Add filters")

    if not modify:
        return df

    df = df.copy()

    modification_container = st.sidebar.container()

    with modification_container:
        # to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
    
        search_string =  st.sidebar.text_input("Search string")
        df = df[
            df["Layer"].str.contains(search_string, case=False, na=False) |
            df["naam"].str.contains(search_string, case=False, na=False) |
            df["address"].str.contains(search_string, case=False, na=False) |
            df["website"].str.contains(search_string, case=False, na=False) |
            df["remarks"].str.contains(search_string, case=False, na=False)
        ] 

    return df

def convert_df(df):
    """Convert df to csv (utf-8) to make it downloadable
    """     
    return df.to_csv(index=False).encode('utf-8')


def read():
    """Read the Google sheet
        Column names
        Id	Layer	naam	address	LON	LAT	website	remarks
    """    
    #https://docs.google.com/spreadsheets/d/1W26AXb91InFg7Lw6TR1mkVh3dm-OQbhGAx0BgNrJdok/edit?usp=sharing
    sheet_id = "1W26AXb91InFg7Lw6TR1mkVh3dm-OQbhGAx0BgNrJdok"
    sheet_name="MASTER"  
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=',')
    return df

def credits():
    """Show footer with credits
    """     
    credits = """
        <h1>Renato in Rome - credits</h1>
       
        Please send feedback to @rcsmit or rcx dot smit at gmail dot com

        """
    st.write(credits, unsafe_allow_html=True)
    
def main():
    st.header("Renato in Rome")
    df_ = read()
    
    df_=df_.fillna("")
    df_["LAT"] = df_["LAT"].str.replace(",",".").astype(float)
    df_["LON"] = df_["LON"].str.replace(",",".").astype(float)

    df_ = filter_dataframe(df_)

    layer_list = df_['Layer'].unique().tolist()
    if len(layer_list)>4:
        default_layer_list = layer_list[:4]
    else:
        default_layer_list = layer_list 
    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list,default_layer_list)

    if layers_to_show == []:
        st.error("Choose a layer")
        st.stop()
    attribution= "CartoDB Positron"
    	
    m = folium.Map(location=[41.833, 12.6391244], zoom_start=11,  tiles = "CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)

    #kleur = ['red', 'green', 'yellow', 'purple', 'blue', 'pink','orange']
    kleur = [
    "#FF5AB3",
    "#3ABF57",
    "#3357AB",
    "#FAB3A1",
    "#33FAB1",
    "#ABA3FF",
    "#FF5733",
    "#33FF57",
    "#3357FF",
    "#FF33A1",
    "#33FFA1",
    "#A133FF",
    "#FFA133",
    "#FF3333",
    "#33FF33",
    "#3333FF",
    "#FF3380",
    "#3380FF",
    "#80FF33",
    "#FF8033",
    "#33FF80",
    "#8033FF",
    "#FF3380",
    "#3380FF",
    "#80FF33",
    "#FF8033",
    "#33FF80",
    "#8033FF",
    "#33A1FF",
    "#FF33A1",
    "#A1FF33",
    "#A133FF",
    "#FFA1FF",
    "#FFA133"
]
    marker_cluster=[]

    #doesnt work
    styl = """  <style>
                        .mycluster {
                            width: 40px;
                            height: 40px;
                            background-color: greenyellow;
                            text-align: center;
                            font-size: 24px;
                        }

	                    </style>"""
    st.markdown(styl, unsafe_allow_html=True)
    def create_function(color):
        icon_create_function = """\
        
            function(cluster) {
                return L.divIcon({
                html: '<font size=4 color="""+ color+"""><center>' + cluster.getChildCount() + '</center></font>',
                className: 'mycluster marker-cluster marker-cluster-small',
                iconSize: new L.Point(40, 40),
                
                });
            }
            
            """

        return icon_create_function
    for i,l in enumerate(layers_to_show):
        marker_cluster.append(None)
        df = df_[df_['Layer'] == l]
        
        # create a marker cluster 
        # https://python-visualization.github.io/folium/latest/user_guide/plugins/marker_cluster.html
        icon_create_function= create_function(kleur[i])
        marker_cluster[i] = MarkerCluster(disableClusteringAtZoom=12, name = l, icon_create_function=icon_create_function,).add_to(m)
        
        for index, row in df.iterrows():
            if row["remarks"] != "None":
                remarks_ = row["remarks"]
            else:
                remarks_ = " "
            if row["website"] != "None":
                website = row["website"]
            else:
                website = " "
                
            depot_node = (row["LAT"], row["LON"]) 
            #google_link = f"https://www.google.com/maps/\@{row["LAT"]},{row["LON"]},17z"
            maps_search = f'https://www.google.com/maps/search/{row["naam"].replace(" ","+")}@/{row["LAT"]},{row["LON"]},15z' 
            remarks = f'<div style="font-size: 12pt; font-family: Arial, Helvetica, sans-serif;"><b>{row["naam"]}</b><br><i>{remarks_}</i><br>{website}<br>{row["address"]}<br>Click for more info</div>'
            html = f'<div style="font-size: 12pt;  font-family: Arial, Helvetica, sans-serif;"><b>{row["naam"]}</b><br>{row["address"]}<br><br><i>{remarks_}</i><br><A HREF="{website}" target="_blank">{website}</A><br><br><a href="{maps_search}" target="_blank">Google maps search</a><BR><BR><B>INFO FROM 2019 or earlier. CHECK INFO</B><br><br>{row["Layer"]}</div>'


            # Create an iframe to contain the HTML
            iframe = folium.IFrame(html, width=400, height=200)

            # Create a popup with the iframe
            popup = folium.Popup(iframe, max_width=2650)
            folium.CircleMarker(location=depot_node,
                                    radius=3,    
                                    color=[kleur[i]],
                                    fill_color =[kleur[i]],
                                    fill_opacity=0.7,
                                    ).add_to(marker_cluster[i])
            folium.map.Marker(depot_node,
                            icon=DivIcon(
                                icon_size=(30,30),
                                icon_anchor=(0,0),
                                html=f'<div style="width: 300px;font-size: 10pt">%s</div>' % row["naam"],
                            ),tooltip=remarks,  popup=popup
                            ).add_to(marker_cluster[i])  

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
    
    df_concat = pd.DataFrame()
    for i,l in enumerate(layers_to_show):
        df_temp = df_[df_['Layer'] == l]
        df_temp=df_temp[["naam","address","website","remarks"]]
        df_concat = pd.concat([df_concat,df_temp])
        st.subheader(l)
        st.table(df_temp)
        

    csv = convert_df(df_concat)

    st.download_button(
    f"Press to Download",
    csv,
    f"info_{l}.csv",
    "text/csv",
    key=f'download-csv-{l}'
    )
    credits()

if __name__ == "__main__":
    main()

