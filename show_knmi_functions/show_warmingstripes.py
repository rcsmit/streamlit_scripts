import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from show_knmi_functions.utils import get_data

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

                fig = ax.imshow(
                    stacked_temps,
                    cmap=cmap,
                    aspect=40,
                )
                plt.gca().set_axis_off()

                plt.title(what)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                # plt.show()
                # st.pyplot(fig) - gives an error
                st.set_option("deprecation.showPyplotGlobalUse", False)
                st.pyplot()
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