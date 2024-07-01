import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from folium.features import DivIcon
from folium import plugins

try:
    st.set_page_config(layout="wide")
except:
    pass

# Google Sheet details (make it publicly accessible)
sheet_id = "1W26AXb91InFg7Lw6TR1mkVh3dm-OQbhGAx0BgNrJdok"
sheet_name = "MASTER"

# Website title and Column names
title = "Renato in Rome"
id_field = "Id"
category_field = "Layer"
name_field = "naam"
address_field = "address"
lon_field = "LON"
lat_field = "LAT"
website_field = "website"
remarks_field = "remarks"

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    modify = st.sidebar.checkbox("Search / Add filters")
    if not modify:
        return df

    search_string = st.sidebar.text_input("Search string")
    df = df[
        df[category_field].str.contains(search_string, case=False, na=False) |
        df[name_field].str.contains(search_string, case=False, na=False) |
        df[address_field].str.contains(search_string, case=False, na=False) |
        df[website_field].str.contains(search_string, case=False, na=False) |
        df[remarks_field].str.contains(search_string, case=False, na=False)
    ]
    return df

def convert_df(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def read_google_sheet() -> pd.DataFrame:
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    return pd.read_csv(url, delimiter=',')

def show_credits():
    credits = """
    <h1>Credits</h1>
    Please send feedback to @rcsmit or rcx dot smit at gmail dot com
    """
    st.write(credits, unsafe_allow_html=True)

def create_icon_function(color: str) -> str:
    # we have to divide by 2 because the markers are added twice.
    return f"""
    function(cluster) {{
        return L.divIcon({{
            html: '<font size=4 color="{color}"><center>' + cluster.getChildCount()/2 + '</center></font>',
            className: 'mycluster marker-cluster marker-cluster-small',
            iconSize: new L.Point(40, 40),
        }});
    }}
    """

def create_map(layers_to_show: list, df: pd.DataFrame):
    attribution = "CartoDB Positron"
    m = folium.Map(location=[41.833, 12.6391244], zoom_start=11, tiles="CartoDB Positron", attr=attribution)
    plugins.Geocoder().add_to(m)

    colors = [
        "#FF5AB3", "#3ABF57", "#3357AB", "#FAB3A1", "#33FAB1",
        "#ABA3FF", "#FF5733", "#33FF57", "#3357FF", "#FF33A1",
        "#33FFA1", "#A133FF", "#FFA133", "#FF3333", "#33FF33",
        "#3333FF", "#FF3380", "#3380FF", "#80FF33", "#FF8033",
        "#33FF80", "#8033FF", "#33A1FF", "#A1FF33", "#A133FF",
        "#FFA1FF", "#FFA133"
    ]

    marker_clusters = []
    for i, layer in enumerate(layers_to_show):
        icon_function = create_icon_function(colors[i % len(colors)])
        cluster = MarkerCluster(disableClusteringAtZoom=12, name=layer, icon_create_function=icon_function).add_to(m)
        df_layer = df[df[category_field] == layer]

        for _, row in df_layer.iterrows():
            depot_node = (row[lat_field], row[lon_field])
            remarks = row.get(remarks_field, " ")
            website = row.get(website_field, " ")
            maps_search = f'https://www.google.com/maps/search/{row[name_field].replace(" ", "+")}@/{row[lat_field]},{row[lon_field]},15z'
            html = f"""
            <div style="font-size: 12pt; font-family: Arial, Helvetica, sans-serif;">
                <b>{row[name_field]}</b><br>{row[address_field]}<br><br>
                <i>{remarks}</i><br>
                <a href="{website}" target="_blank">{website}</a><br><br>
                <a href="{maps_search}" target="_blank">Google maps search</a><br><br>
                <b>INFO FROM 2019 or earlier. CHECK INFO</b><br><br>{row[category_field]}
            </div>
            """
            iframe = folium.IFrame(html, width=400, height=300)
            popup = folium.Popup(iframe, max_width=2650)
            # markers are added twice to the cluster, we'll have to correct later
            folium.CircleMarker(
                location=depot_node, radius=3, color=colors[i % len(colors)],
                fill_color=colors[i % len(colors)], fill_opacity=0.7
            ).add_to(cluster)
            folium.Marker(
                location=depot_node,
                icon_size=(30,30),
                icon_anchor=(0,5),
                icon=DivIcon(html=f'<div style="width: 300px;font-size: 12pt">{row[name_field]}</div>'),
                tooltip=row[name_field], popup=popup
            ).add_to(cluster)

        marker_clusters.append(cluster)
        st.markdown("""
                    <style>
                    .big-font {
                        font-size:30px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                            
        text= f"<font  class='big-font'  color={colors[i]}>•</font> - {layer}"
        st.sidebar.write(text, unsafe_allow_html=True)
    return m, marker_clusters

def main():
    st.header(title)
    df = read_google_sheet().fillna("")
    df[lat_field] = df[lat_field].str.replace(",", ".").astype(float)
    df[lon_field] = df[lon_field].str.replace(",", ".").astype(float)

    df = filter_dataframe(df)

    layer_list = df[category_field].unique().tolist()
    default_layer_list = layer_list[:4] if len(layer_list) > 4 else layer_list
    layers_to_show = st.sidebar.multiselect("Layers to show", layer_list, default_layer_list)

    if not layers_to_show:
        st.error("Choose a layer / Nothing found")
        st.stop()

    map_obj, _ = create_map(layers_to_show, df)
    st_folium(map_obj, width=1500, returned_objects=[])

    df_concat = pd.DataFrame()
    for layer in layers_to_show:
        
        df_temp = df[df[category_field] == layer]
        df_temp=df_temp[[name_field,address_field,website_field,remarks_field]]
        df_concat = pd.concat([df_concat, df_temp[[name_field, address_field, website_field, remarks_field]]])
        st.subheader(layer)
        st.table(df_temp)

    csv = convert_df(df_concat)
    st.download_button("Click to download", csv, f"info_{layer}.csv", "text/csv", key=f'download-csv-{layer}')

    show_credits()

if __name__ == "__main__":
    main()
