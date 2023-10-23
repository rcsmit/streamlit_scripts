import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import ListedColormap
_lock = RendererAgg.lock
from show_knmi_functions.utils import get_data
def show_warmingstripes(df_, title):
    print (df_)
    df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
    #df_grouped = df.groupby([df[valuefield]], sort=True).sum().reset_index()
    # Based on code of Sebastian Beyer
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py

    # the colors in this colormap come from http://colorbrewer2.org
    # the 8 more saturated colors from the 9 blues / 9 reds
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    cmap = ListedColormap(
        [
            "#08306b",
            "#08519c",
            "#2171b5",
            "#4292c6",
            "#6baed6",
            "#9ecae1",
            "#c6dbef",
            "#deebf7",
            "#fee0d2",
            "#fcbba1",
            "#fc9272",
            "#fb6a4a",
            "#ef3b2c",
            "#cb181d",
            "#a50f15",
            "#67000d",
        ]
    )
    # https://github.com/sebastianbeyer/warmingstripes/blob/master/warmingstripes.py
    temperatures = df["temp_avg"].tolist()
    stacked_temps = np.stack((temperatures, temperatures))
    with _lock:
        # plt.figure(figsize=(4,18))
        fig, ax = plt.subplots()

        fig = ax.imshow(
            stacked_temps,
            cmap=cmap,
            aspect=40,
        )
        plt.gca().set_axis_off()

        plt.title(title)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.show()
        # st.pyplot(fig) - gives an error
        st.set_option("deprecation.showPyplotGlobalUse", False)
        st.pyplot()
        show_warmingstripes_matplotlib(df_, title)

def show_warmingstripes_matplotlib(df_, title):
    # https://matplotlib.org/matplotblog/posts/warming-stripes/
    st.subheader("Code from Matplotlib site")
    df = df_.groupby(df_["YYYY"], sort=True).mean(numeric_only = True).reset_index()
    avg_temperature = df["temp_avg"].mean()
    df["anomaly"] = df["temp_avg"]-avg_temperature

    #stacked_temps = np.stack((temperatures, temperatures))
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import ListedColormap
    import pandas as pd
    # Then we define our time limits, our reference period for the neutral color and the range around it for maximum saturation.

    FIRST = int( df["YYYY"].min())
    LAST = int(df["YYYY"].max())  # inclusive

    # Reference period for the center of the color scale

    FIRST_REFERENCE = FIRST
    LAST_REFERENCE = LAST
    LIM = 2 # degrees

    #Here we use pandas to read the fixed width text file, only the first two columns, which are the year and the deviation from the mean from 1961 to 1990.

    # data from

    # https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/time_series/HadCRUT.4.6.0.0.annual_ns_avg.txt

 
    anomaly = df['anomaly'].tolist()
  
    reference = sum(anomaly)/len(anomaly)
    # This is our custom colormap, we could also use one of the colormaps that come with matplotlib, e.g. coolwarm or RdBu.

    # the colors in this colormap come from http://colorbrewer2.org

    # the 8 more saturated colors from the 9 blues / 9 reds

    cmap = ListedColormap([
        '#08306b', '#08519c', '#2171b5', '#4292c6',
        '#6baed6', '#9ecae1', '#c6dbef', '#deebf7',
        '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a',
        '#ef3b2c', '#cb181d', '#a50f15', '#67000d',
    ])
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

def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    
if __name__ == "__main__":
    # main()
    print ("")