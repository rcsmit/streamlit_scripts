import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.features import DivIcon
from folium import plugins
def read():
    #https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing
    sheet_id = "1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0"
    sheet_name="MASTER"  
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=',')
    return df


def main():
    st.header("Conscious communities and ecovillages around the world")
    df = read()
    df["LAT"] = df["LAT"].astype(float)
    df["LON"] = df["LON"].astype(float)

    layer_list = df['Layer'].unique().tolist()

    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list,["Conscious_communities"])
    if layers_to_show == []:
        st.error("Choose a layer")
        st.stop()
    df = df[df['Layer'].isin(layers_to_show)]
   
    
    attribution= "CartoDB Positron"
    m = folium.Map(location=[10.5074, 52.2], zoom_start=2,  tiles = "CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)
    for index, row in df.iterrows():
        #folium.Marker(location=[row["LAT"], row["LON"]], tooltip=row["Name"]).add_to(m)
        if row["remarks"] != "None":
            remarks = row["remarks"]
        else:
            remarks = " "
        depot_node = (row["LAT"], row["LON"])            
        folium.CircleMarker(location=depot_node,
                                radius=3,    
                                color='red',
                                fill_color ='red',
                                fill_opacity=0.7,
                                ).add_to(m)
        folium.map.Marker(depot_node,
                        icon=DivIcon(
                            icon_size=(30,30),
                            icon_anchor=(5,14),
                            html=f'<div style="font-size: 10pt">%s</div>' % row["Name"],
                        ),tooltip=remarks
                        ).add_to(m)
        
    # Display the map in Streamlit
    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=1500, returned_objects=[])
    credits()

def credits():
     
    credits = """"
        <h1>Conscious Communities - credits</h1>
        <p>With contributions of
        <ul>
        <li>People who replied on numerous discussions in forums and facebookgroups, for example: (membership needed)</li>
            <ul>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/5430226180393792/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/5430226180393792/</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/4541454629270956/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/4541454629270956</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/1600295400053575/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/1600295400053575/</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/posts/5737144509701956/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/posts/5737144509701956/</a></li>
            <li><a href='https://www.facebook.com/groups/348616157134904' target='_blank'>https://www.facebook.com/groups/348616157134904</a></li>
            </ul>
        <li><a href='https://asliinwonderland.com/2021/07/02/conscious-communities-across-the-globe/' target='_blank'>Asli In Wonderland</a></li>
        <li><a href='https://goo.gl/maps/sxXi5DZjhh2WTJJb6' target='_blank'>Hippie spirit places around the world</a> by <a href='https://www.instagram.com/peggy.anke' target='_blank'>Peggy Anke</a></li>
        </ul>"""
    st.write(credits, unsafe_allow_html=True)
if __name__ == "__main__":
    
    main()
