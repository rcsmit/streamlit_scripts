import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from show_knmi_functions.utils import get_data
import plotly.graph_objects as go

#_lock = RendererAgg.lock

# This is our custom colormap, we could also use one of the colormaps that come with matplotlib, e.g. coolwarm or RdBu.
# the colors in this colormap come from http://colorbrewer2.org
# the 8 more saturated colors from the 9 blues / 9 reds

cmap = ListedColormap([
    '#08306b', '#08519c', '#2171b5', '#4292c6',
    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
    '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
])

cmap_blue = ListedColormap([
    '#08306b', '#08519c', '#2171b5', '#4292c6',
    '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',])
    


cmap_red = ListedColormap([
    '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
    '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
])



# Zelfde colorscale-gevoel als de matplotlib cmap (8 blauw / 8 rood, colorbrewer)
WARMING_STRIPES_COLORSCALE = [
    [0.000, '#08306b'], [0.125, '#08519c'], [0.250, '#2171b5'], [0.375, '#4292c6'],
    [0.500, '#6baed6'], [0.500, '#fee0d2'],  # midden = overgang blauw -> rood
    [0.625, '#fcbba1'], [0.750, '#fc9272'], [0.875, '#fb6a4a'], [1.000, '#67000d'],
]
 
 
def show_warmingstripes_plotly(df_, what_to_show, title):
    """Plotly-versie van de warming stripes met hover (jaar + waarde).
 
    Args:
        df_ (df): dataframe met kolom "YYYY" en de te tonen variabele(n)
        what_to_show (list): lijst van kolomnamen om te tonen
        title (str): titel van de plot
        style (str): "bar" -> staafjes per jaar (zoals showyourstripes.info)
                     "heatmap" -> doorlopende band (zoals "Classic")
    """
    style = st.sidebar.selectbox("Kies stijl", ["bar", "heatmap"], index=0, help="Bar = staafjes per jaar (zoals showyourstripes.info), heatmap = doorlopende band (zoals Classic)")
    modus=st.sidebar.selectbox("Kies modus", ["mean", "max", "min"], index=0, help="Mean = gemiddelde per jaar, max = maximum per jaar, min = minimum per jaar")
    modus2=st.sidebar.selectbox("Kies modus2", ["anomaly", "absolute_value"], index=0)
    for what in what_to_show:
        if modus=="mean":
            df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only=True).reset_index()
        elif modus =="max":
            df = df_.groupby(df_["YYYY"], sort=True).max(numeric_only=True).reset_index()
        elif modus =="min":
            df = df_.groupby(df_["YYYY"], sort=True).min(numeric_only=True).reset_index()

        avg = df[what].mean()
        if modus2=="anomaly":
            df["anomaly"] = df[what] - avg
        elif modus2=="absolute_value":
            df["anomaly"] = df[what]

        years = df["YYYY"].tolist()
        anomaly = df["anomaly"].tolist()
        values = df[what].tolist()
 
        lim = max(abs(min(anomaly)), abs(max(anomaly)))
 
        if style == "heatmap":
            fig = go.Figure(
                data=go.Heatmap(
                    z=[anomaly],
                    x=years,
                    y=[what],
                    colorscale=WARMING_STRIPES_COLORSCALE,
                    zmin=-lim,
                    zmax=lim,
                    customdata=[values],
                    hovertemplate="Jaar: %{x}<br>Anomalie: %{z:.2f}°C<br>Waarde: %{customdata:.2f}°C<extra></extra>",
                    showscale=True,
                    colorbar=dict(title="Anomalie (°C)"),
                )
            )
            fig.update_yaxes(visible=False, showticklabels=False)
            fig.update_xaxes(title="Jaar")
            fig.update_layout(height=250)
        else:
            # style == "bar"
            fig = go.Figure(
                data=go.Bar(
                    x=years,
                    y=anomaly,
                    customdata=values,
                    marker=dict(
                        color=anomaly,
                        colorscale=WARMING_STRIPES_COLORSCALE,
                        cmin=-lim,
                        cmax=lim,
                        colorbar=dict(title="Anomalie (°C)"),
                        line=dict(width=0),
                    ),
                    hovertemplate="Jaar: %{x}<br>Anomalie: %{y:.2f}°C<br>Waarde: %{customdata:.2f}°C<extra></extra>",
                )
            )
            fig.update_layout(
                bargap=0,
                yaxis=dict(visible=False, showticklabels=False),
                xaxis=dict(title="Jaar"),
                height=400,
            )
 
        fig.update_layout(
            title=title or what,
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=40),
        )
 
        st.plotly_chart(fig, use_container_width=True)


def show_warmingstripes_matplotlib(df_, what):
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    st.subheader(f"Code from Matplotlib site - {what}")
    df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
    avg_temperature = df[what].mean()
    df["anomaly"] = df[what]-avg_temperature

    #stacked_temps = np.stack((temperatures, temperatures))

    # Then we define our time limits, our reference period for the neutral color and the range around it for maximum saturation.

    FIRST = int( df["YYYY"].min())
    LAST = int(df["YYYY"].max())  # inclusive

    # Reference period for the center of the color scale

    FIRST_REFERENCE = FIRST
    LAST_REFERENCE = LAST
    LIM = 2 # degrees

    anomaly = df['anomaly'].tolist()

    reference = sum(anomaly)/len(anomaly)

    # We create a figure with a single axes object that fills the full area of the figure and does not have any axis ticks or labels.

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    # Finally, we create bars for each year, assign the data, colormap and color limits and add it to the axes.

    # create a collection with a rectangle for each year

    col = PatchCollection([
        Rectangle((y, 0), 1, 1)
        for y in range(FIRST, LAST + 1)
    ])

    # set data, colormap and color limits

    col.set_array(anomaly)
    col.set_cmap(cmap)
    col.set_clim(reference - LIM, reference + LIM)
    ax.add_collection(col)
    #Make sure the axes limits are correct and save the figure.

    ax.set_ylim(0, 1)
    ax.set_xlim(FIRST, LAST + 1)

    fig.savefig('warming-stripes.png')
    st.pyplot(fig)

def show_warmingstripes(df_, what_to_show, title, mode):
    """_summary_

    Args:
        df_ (df): df
        what_to_show (list): what to show
        title (str): title of the plot
        mode (str): "matplotlib", - from matplotlib site
                    "classic", - from Sebastian Beyer
                     "new" - inspired by https://showyourstripes.info/b/europe/netherlands/amsterdam/
    """
    for what in what_to_show:
        if mode == "Matplotlib":
            show_warmingstripes_matplotlib(df_,what_to_show)
        elif mode == "Plotly":
            show_warmingstripes_plotly(df_, what_to_show, title, style="bar")  
        else:
            # mode == "classic" | mode == "new"

            df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
            # Based on code of Sebastian Beyer
            # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py
            # to SVG: https://commons.wikimedia.org/wiki/User:RCraig09/Excel_to_XML_for_SVG
            temperatures = df[what].tolist()
            df['avg'] = df[what].mean()
            df['anomaly'] = df[what]-df['avg']

            # Set values under 0 to None
            df['anomaly_blue'] = df['anomaly'].apply(lambda x: x if x >= 0 else None)
            df['anomaly_red'] = df['anomaly'].apply(lambda x: x if x >= 0 else None)

            anomaly_ = df["anomaly"].tolist()
            anomaly_blue = df["anomaly_blue"].tolist()
            anomaly_red = df["anomaly_red"].tolist()
            
            years = df['YYYY'].tolist()
            # Normalize temperatures to fit colormap range
            norm = plt.Normalize(min(temperatures), max(temperatures))
            colors = cmap(norm(temperatures))
            stacked_temps = np.stack((temperatures, temperatures))
            
            if mode == "Classic":
                fig, ax = plt.subplots()

                ax.imshow(
                    stacked_temps,
                    cmap=cmap,
                    aspect=40,
                )
                ax.set_axis_off()
                ax.set_title(what)
                ax.xaxis.set_major_locator(plt.NullLocator())
                ax.yaxis.set_major_locator(plt.NullLocator())

                st.pyplot(fig)
            else:
                # mode == new
                # inspired by https://showyourstripes.info/b/europe/netherlands/amsterdam/
            
                fig, ax = plt.subplots(figsize=(15, 5))

                # Create bar plot
                ax.bar(years, anomaly_, color=colors, edgecolor='none')


                # plt.bar(years, anomaly_blue, color = 'red')
                # plt.bar(years, anomaly_red, color = 'blue')
                # Hide axes
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                #ax.get_xaxis().set_visible(True)
                ax.get_yaxis().set_visible(False)

                ax.set_title(title, fontsize=16)

                st.pyplot(fig)
        
def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    show_warmingstripes(df, ["temp_avg"], "sample", "classic")
if __name__ == "__main__":
    # main()
    print ("")