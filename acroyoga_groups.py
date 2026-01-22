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
    sheet_name="ACROYOGA"  
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    df = pd.read_csv(url, delimiter=',')
    return df

def credits():
    """Show footer with credits
    """     
    credits = """
        <h1>Conscious Communities - credits</h1>
        
        Google sheet: <a href='https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing'>https://docs.google.com/spreadsheets/d/1pOuO8Z3w61VOpdcXVyKwyZRoZMmQG9AFCNQrAd-I5P0/edit?usp=sharing</a>
        
        Please send feedback to @rcsmit or rcx dot smit at gmail dot com

        """
    st.write(credits, unsafe_allow_html=True)

def make_clickable(link):
    # target _blank to open new window
    # extract clickable text to display for your link
    #text = link.split('=')[1]
    if 'facebook' in link:
        r= f'<a target="_blank" href="{link}">FB</a>'
    elif 'whatsapp' in link:
        r=f'<a target="_blank" href="{link}">WA</a>'
    else:
        r =""
    return r

def show_table(df):
    # link is the column with hyperlinks
    for c in ['facebook', 'whatsapp']:
    
        df[c] = df[c].apply(make_clickable)
    df = df[["Name", "country__", "whatsapp"]]
    df = df.to_html(escape=False)
    st.write(df, unsafe_allow_html=True)

   
def main():
    st.header("Acroyoga communities around the world")
    df_ = read()
    df_=df_.fillna("")
    df_["LAT"] = df_["LAT"].astype(float)
    df_["LON"] = df_["LON"].astype(float)
    layer_list = df_['Layer'].unique().tolist()

    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list,["Acroyoga"])
    if layers_to_show == []:
        st.error("Choose a layer")
        st.stop()

    attribution= "CartoDB Positron"
    m = folium.Map(location=[13.7563, 100.5], zoom_start=3,  tiles = "CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)

    kleur = ['red', 'green', 'yellow', 'purple', 'blue', 'pink','orange']
    
    # Add CSS styling once, outside the loop
    st.markdown("""
                <style>
                .big-font {
                    font-size:30px !important;
                }
                </style>
                """, unsafe_allow_html=True)
    
    # Collect all filtered data for the table
    all_filtered_data = []

    for i,l in enumerate(layers_to_show):
        df = df_[df_['Layer'] == l]
        all_filtered_data.append(df)
        
        for index, row in df.iterrows():
            if row["remarks"] != "None" and row["remarks"] != "":
                remarks_ = row["remarks"]
            else:
                remarks_ = " "
                
            depot_node = (row["LAT"], row["LON"]) 
            remarks = f'<b>{row["Name"]}</b><br><i>{remarks_}</i><br><br>{row["country__"]}<br>{row["whatsapp"]}<br>{row["facebook"]}' #<br>{depot_node}'
                       
            folium.CircleMarker(location=depot_node,
                                    radius=3,    
                                    color=kleur[i],
                                    fill_color=kleur[i],
                                    fill_opacity=0.7,
                                    ).add_to(m)
            folium.map.Marker(depot_node,
                            icon=DivIcon(
                                icon_size=(30,30),
                                icon_anchor=(5,14),
                                html=f'<div style="font-size: 10pt">%s</div>' % row["Name"],
                            ),tooltip=remarks
                            ).add_to(m)  
                            
        text= f"<font  class='big-font'  color={kleur[i]}>•</font> - {l}"
        st.sidebar.write(text, unsafe_allow_html=True)
            
    # Display the map in Streamlit
    # call to render Folium map in Streamlit
    st_data = st_folium(m, width=1500, returned_objects=[])
    
    # Show table with all selected layers combined
    if all_filtered_data:
        combined_df = pd.concat(all_filtered_data, ignore_index=True)
        show_table(combined_df)

    credits()

if __name__ == "__main__":
    main()

