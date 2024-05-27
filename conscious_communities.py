import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.features import DivIcon
from folium import plugins
def read():
    """Read the Google sheet
    """    
    #https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing
    sheet_id = "1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0"
    sheet_name="MASTER"  
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=',')
    return df

def credits():
    """Show footer with credits
    """     
    credits = """
        <h1>Conscious Communities - credits</h1>
        <p>With contributions of
        <ul>
        <li>People who replied on numerous discussions in forums and facebookgroups, for example: (membership needed)</li>
            <li><ul>
            <li><a href='https://www.facebook.com/groups/consciousnomadsworld/posts/503477371648781' target='_blank'>https://www.facebook.com/groups/consciousnomadsworld/posts/503477371648781</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/5430226180393792/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/5430226180393792/</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/4541454629270956/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/4541454629270956</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/permalink/1600295400053575/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/permalink/1600295400053575/</a></li>
            <li><a href='https://www.facebook.com/groups/KohPhanganCC/posts/5737144509701956/' target='_blank'>https://www.facebook.com/groups/KohPhanganCC/posts/5737144509701956/</a></li>
            <li><a href='https://www.facebook.com/groups/348616157134904' target='_blank'>https://www.facebook.com/groups/348616157134904</a></li></ul></li>
        <li><a href='https://asliinwonderland.com/2021/07/02/conscious-communities-across-the-globe/' target='_blank'>Asli In Wonderland</a></li>
        <li><a href='https://goo.gl/maps/sxXi5DZjhh2WTJJb6' target='_blank'>Hippie spirit places around the world</a> by <a href='https://www.instagram.com/peggy.anke' target='_blank'>Peggy Anke</a></li>
        <li><a href='https://www.pureportugal.co.uk/blogs/communities-in-portugal/'>Communities in Portugal by Jazz Meyer</a></li></ul>
        
        Google sheet: <a href='https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing'>https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing</a>
        
        Please send feedback to @rcsmit or rcx dot smit at gmail dot com

        """
    st.write(credits, unsafe_allow_html=True)
    
def main():
    st.header("Conscious communities and ecovillages around the world")
    df_ = read()
    df_=df_.fillna("")
    df_["LAT"] = df_["LAT"].astype(float)
    df_["LON"] = df_["LON"].astype(float)
    layer_list = df_['Layer'].unique().tolist()

    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list,layer_list)

    if layers_to_show == []:
        st.error("Choose a layer")
        st.stop()

    attribution= "CartoDB Positron"
    m = folium.Map(location=[10.5074, 52.2], zoom_start=2,  tiles = "CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)

    kleur = ['red', 'green', 'yellow', 'purple', 'blue', 'pink','orange']

    for i,l in enumerate(layers_to_show):
        df = df_[df_['Layer'] == l]
        
        for index, row in df.iterrows():
            if row["remarks"] != "None":
                remarks_ = row["remarks"]
            else:
                remarks_ = " "
                
            depot_node = (row["LAT"], row["LON"]) 
            remarks = f'<b>{row["Name"]}</b><br><i>{remarks_}</i><br><br>{row["city__"]}<br>{row["provincie"]}<br>{row["country__"]}<br>{row["continent"]}<br>{depot_node}'
                       
            folium.CircleMarker(location=depot_node,
                                    radius=3,    
                                    color=[kleur[i]],
                                    fill_color =[kleur[i]],
                                    fill_opacity=0.7,
                                    ).add_to(m)
            folium.map.Marker(depot_node,
                            icon=DivIcon(
                                icon_size=(30,30),
                                icon_anchor=(5,14),
                                html=f'<div style="font-size: 10pt">%s</div>' % row["Name"],
                            ),tooltip=remarks
                            ).add_to(m)  

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
    credits()

if __name__ == "__main__":
    main()

