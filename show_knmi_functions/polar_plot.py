import pandas as pd
import numpy as np
import streamlit as st
#from streamlit import caching
import matplotlib.pyplot as plt
# import matplotlib
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.animation import FuncAnimation
try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
_lock = RendererAgg.lock
import sys # for the progressbar
import shutil # for the progressbar

import plotly.express as px
import plotly.graph_objects as go

import platform
import streamlit.components.v1 as components
import time
import imageio
import os
import webbrowser

def  polar_plot(df2,   what_to_show, how):
    """_summary_

    Args:
        df2 (_type_): _description_
        what_to_show (_type_): _description_
        how (_type_): "line" or "scatter"

    Returns:
        _type_: _description_
    """    
    # https://scipython.com/blog/visualizing-the-temperature-in-cambridge-uk/
    # import numpy as np
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # from matplotlib import cm
    # from matplotlib.colors import Normalize
    # from mpl_toolkits.mplot3d import Axes3D
    # plt.rcParams['text.usetex'] = True
 
    for w in what_to_show:   
        st.subheader(w)
        df2["YYYYMMDD_"] = pd.to_datetime(df2["YYYYMMDD"], format="%Y%m%d")
        # Convert the timestamp to the number of seconds since the start of the year.
        df2['secs'] = (df2.YYYYMMDD_ - pd.to_datetime(df2.YYYYMMDD.dt.year, format='%Y')).dt.total_seconds()
        df2['dayofyear'] =  df2["YYYYMMDD_"].dt.dayofyear
        df2['angle_rad']=((360/365)*df2['dayofyear'])*np.pi/180 # = hoek in radialen
        # Approximate the angle as the number of seconds for the timestamp divide by
        # the number of seconds in an average year.
        df2['angle_degrees'] = df2['secs'] / (365.25 * 86400) *360  #   * 2 * np.pi
        big_angle= 360/12  # How we split our polar space

        
        
        

        def plot_polar_plotly(how):
            st.subheader(f"Plotly - {how}")
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
            
            if how == "line":
                # geeft foutmelding  als number of days groter is dan 3--9-2021 and 29-05-2024 = 1047 DAGEN
                # https://plotly.com/python/reference/scatterpolargl/
                try:
                    # works locally
                    fig = px.line_polar(df2, r=w, color='YYYY', theta='angle_degrees',color_discrete_sequence=px.colors.sequential.Plasma_r, line_close=False, hover_data=['YYYYMMDD'])
                except:
                    fig = px.line_polar(df2, r=w, color='YYYY', theta='angle_degrees', line_close=False, hover_data=['YYYYMMDD'])
                
                fig.update_traces(line=dict(width=0.75))
            

            elif how == "scatter":
                fig = px.scatter_polar(df2, r=w, color='YYYY', theta='angle_degrees', hover_data=['YYYYMMDD'])

            else:
                st.error("Error in HOW")
            fig.update_layout(coloraxis={"colorbar":{"dtick":1}}) #only integers in legeenda
            labelevery = 6
            fig.update_layout(
                polar={
                    "angularaxis": {
                        "tickmode": "array",
                        "tickvals": list(range(0, 360, 180 // labelevery)),
                        "ticktext": months,
                    }
                }
            )
            st.plotly_chart(fig)

     
      
        def plot_matplotlib_line():
            def dress_axes(ax):
                #https://matplotlib.org/matplotblog/posts/animated-polar-plot/
                #inner,outer = -10,50
                # Find maximum and minimum values in column 'w'
                max_value = df2[w].max()
                min_value = df2[w].min()

                # Round the maximum value up to the nearest multiple of 5
                max_value_rounded_up = np.ceil(max_value / 5) * 5

                # Round the minimum value down to the nearest multiple of 5
                min_value_rounded_down = np.floor(min_value / 5) * 5
                #print (f"{inner} {outer}")
                ax.set_facecolor('w')
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                # Here is how we position the months labels

                middles=np.arange(big_angle/2 ,360, big_angle)*np.pi/180
                ax.set_xticks(middles)
                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
                #ax.set_yticks([-10,-5,0,5,10,15,20,25,30,35,40,45])
                # Set the y-axis ticks dynamically based on min and max values
                #ax.set_yticks(range(int(min_value_rounded_down), int(max_value_rounded_up) + 5, 5))

                #ax.set_yticklabels(['-10°C','-5°C','0°C','5°C','10°C','15°C','20°C','25°C','30°C','35°C', '40°C','45°C'])
                # Define the y-axis ticks and their corresponding labels
                # Calculate the range between min and max values
                value_range = max_value_rounded_up - min_value_rounded_down
                n_ticks = value_range / 5
                if n_ticks > 8:
                    steps = 10
                else:
                    steps = 5
                yticks = range(int(min_value_rounded_down), int(max_value_rounded_up) + 5, steps)
                yticklabels = [f'{temp}°C' for temp in yticks]
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
                ax.tick_params(axis='y', labelsize=8) 

                # Changing radial ticks angle

                ax.set_rlabel_position(359)
                ax.tick_params(axis='both',color='w')
                plt.grid(None,axis='x')
                plt.grid(axis='y',color='w', linestyle=':', linewidth=1)    
                # Here is the bar plot that we use as background

                bars = ax.bar(middles, max_value_rounded_up, width=big_angle*np.pi/180, bottom=min_value_rounded_down, color='lightgray', edgecolor='w',zorder=0)
                
            #plt.ylim([inner,outer])
            # Custom legend

            # Create a figure and polar axes
            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')
 
            def make_graph(day):
                def display_progress_bar(
                    number: int, total: int, ch: str = "█", scale: float = 0.55) -> None:
                    """Display a simple, pretty progress bar.

                    Example:
                    ~~~~~~~~
                    PSY - GANGNAM STYLE(강남스타일) MV.mp4
                    ↳ |███████████████████████████████████████| 100.0%

                    :param number:
                        step number
                    :param int total:
                        total
                    :param str ch:
                        Character to use for presenting progress segment.
                    :param float scale:
                        Scale multiplier to reduce progress bar size.

                    """
                    columns = shutil.get_terminal_size().columns
                    max_width = int(columns * scale)

                    filled = int(round(max_width * number / float(total)))
                    remaining = max_width - filled
                    progress_bar = ch * filled + "_" * remaining
                    percent = round(100.0 * number / float(total), 1)
                    text = f" ↳ |{progress_bar}| {percent}%  ({round(number)}/{round(total,1)})\r"
                    sys.stdout.write(text)
                    sys.stdout.flush()
                
                display_progress_bar(day,len(df2))
                ax.cla()
                dress_axes(ax)
                last_year = df2["YYYY"].max()
                df_last_year = df2[df2["YYYY"] == last_year]
                number_of_rows_last_year = len (df_last_year)
                treshold = number_of_rows_last_year
                if len(df2) > treshold:
                    i0 = len(df2)-treshold
                else:
                    i0 = int(len(df2)/3)

                if day<i0:
                    filtered_data_1 =df2[0:day+1]
                    filtered_data_2 = df2[0:0]
                else:
                    filtered_data_1 =df2[0:i0+1]
                    filtered_data_2 =df2[i0:day+1]
                

                theta_1 = filtered_data_1["angle_rad"].to_numpy()  
                r_1 = filtered_data_1[w].to_numpy()
                ax.plot(theta_1, r_1, color='yellow', linewidth=0.5)
                if len(filtered_data_2) == 0:
                    pass
                else:
                    theta_2 = filtered_data_2["angle_rad"].to_numpy()  
                    r_2 = filtered_data_2[w].to_numpy()
                    ax.plot(theta_2, r_2, color='orange',alpha=1.0,linewidth=1)

                
                ax.set_title(f"Day: {day}")
               
                filename= (f"polarplot_{day}")


                #plt.savefig(filename, dpi=100,)
                return filename

            # Create the animation
            days = range(0, len(df2) + 1)
            st.subheader("Matplotlib last day")
            make_graph(len(df2)+1)
            st.pyplot(fig)
            if platform.processor():
                show_animation = False # True
                prepare_for_animation = False
            else:
                show_animation = False
                prepare_for_animation = False
                st.info("Animation only available locally")

            if show_animation:
                print ("Generating animation")
                st.subheader(" Animation")
                s1 = int(time.time())
                animation = FuncAnimation(fig, make_graph, frames=days, repeat=False)
                
                #HtmlFile = line_ani.to_html5_video()
                with open("myvideo.html","w") as f:
                    print(animation.to_html5_video(), file=f)
                
                HtmlFile = open("myvideo.html", "r")
                #HtmlFile="myvideo.html"
                source_code = HtmlFile.read() 
                components.html(source_code, height = 900,width=900)
                s2 = int(time.time())
                print("")
                print (f"Needed time : {s2-s1} sec")
                # Display the animation
                #st.pyplot(fig)
            if prepare_for_animation == True:
                filenames = []
                for i in range(0, len(df2)):
                    filename = make_graph(i)
                    filenames.append(filename)

                # build gif
                with imageio.get_writer('mygif.gif', mode='I') as writer:
                    for filename_ in filenames:
                        image = imageio.imread(f"{filename_}.png")
                        writer.append_data(image)
                webbrowser.open('mygif.gif')

                # Remove files
                for filename__ in set(filenames):
                    os.remove(f"{filename__}.png")
       
        plot_polar_plotly("scatter")
        plot_polar_plotly("line")
        
        plot_matplotlib_line()
        
        
def main():
   
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)
    polar_plot(df, ["temp_avg"], None)

if __name__ == "__main__":
    main()
