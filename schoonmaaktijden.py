#import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy import stats
#from scipy.stats import weibull_min
import pandas as pd
from statistics import mean
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock
import streamlit as st
import random
from itertools import cycle
from streamlit import caching
import time
# partly derived from https://stackoverflow.com/a/37036082/4173718
def calculate_weibull_(x, scale, shape):
    if x == 0: return 0
    x_min_1 = 1-np.exp(-1*((x-1/scale)**shape))
    xx = 1-np.exp(-1*((x/scale)**shape))
    return (x_min_1 - xx)

def calculate_mean(scale,shape):
    n = (1+ (1/shape))
    gamma = math.gamma(n)

    # for t in range (1_000_000):
    #     gamma += t**(n-1)* np.exp(-t)
    return scale*gamma


def calculate_weibull(x, scale, shape):

    return (shape/scale) * ((x/scale)**(shape - 1)) * np.exp(-1*((x/scale)**shape))

    if x == 0: return 0
    x_min_1 = 1-np.exp(-1*((x-1/scale)**shape))
    xx = 1-np.exp(-1*((x/scale)**shape))
    return (x_min_1 - xx)

@st.cache(ttl=60 * 60 * 24)
def read():
    sheet_id = "1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg"
    sheet_name = "gegevens"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    #url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\schoonmaaktijden.csv",
    df = pd.read_csv(url, delimiter=',')
    #df = df[:-1]  #remove last row which appears to be a Nan

    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    return df

def lineplot(data, acco_name):
    data_serie = pd.Series(data)
    sma = data_serie.rolling(window=5, center=False).mean()

    with _lock:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data, linestyle="dotted")
        ax.plot(sma)
        title =  (f"Schoonmaaktijden door de tijd heen - {acco_name} ")
        plt.title(title)
        st.pyplot(fig)
        fig = plt.close()

def calculate_and_plot(data, acco_name, modus, animation):

    a_in = 1 # α = 1 gives the Weibull distribution;
    loc_in = 0
    if modus == "exponweib":
        a_out, Kappa_out, loc_out, Lambda_out = stats.exponweib.fit(data, f0=a_in,floc=loc_in)
    else:
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        Kappa_out = shape
        Lambda_out = scale
    #Plot
    bins_formula = range( int(max(data))+1)

    binwidth = max(data)/10


    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    with _lock:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax3 = ax.twinx()
        if modus == "exponweib":
            ax3.plot(bins_formula, stats.exponweib.pdf(bins_formula, a=a_out,c=Kappa_out,loc=loc_out,scale = Lambda_out))
        else:
            ax3.plot(bins_formula, stats.weibull_min(shape, loc, scale).pdf(bins_formula))
        ax3.plot (bins_formula, calculate_weibull(bins_formula, scale, shape))
        ax.hist(data, bins = bins , density=False, alpha=0.5)
        mediaan =Lambda_out *(np.log(2) **(1/Kappa_out))
        mean_data = mean(data)
        mean_calc =calculate_mean (scale, shape)
        title =  (f"{acco_name} (n={len(data)})\n\nShape: {round(Kappa_out,2)} - Scale: {round(Lambda_out,2)}\nMediaan : {round(mediaan,2)} - mean data : {round(mean_data,2)} -  - mean calc : {round(mean_calc,2)}")
        samenvatting = [acco_name, len(data), round(Kappa_out,2), round(Lambda_out,2), round(mediaan,2), round(mean_data,2), round(mean_calc,2)]
        plt.title(title)

        #st.write (title)
        # plt.show()
        if animation ==True:
            placeholder.pyplot(fig)
        else:
            st.pyplot(fig)
        fig = plt.close()

    return samenvatting


def show_animation(df, acco_codes, acco_names, distribution_to_use ):

    code_ =  st.selectbox("Which accotype to show", acco_names, index=0)
    code = acco_codes[acco_names.index(code_)]
    df_selection = select_data(df, code)

    global placeholder
    animations = {"None": None, "Slow": 0.4, "Medium": 0.2, "Fast": 0.05}
    animate = st.sidebar.radio("", options=list(animations.keys()), index=2)
    animation_speed = animations[animate]

    slider_placeholder = st.empty()
    placeholder = st.empty()


    if animation_speed:

        c = range(1,len(df_selection)+1)
        for i in cycle(c):
            time.sleep(animation_speed)

            #TO FIX:  stap 1 wordt overgeslagen.
            j = slider_placeholder.slider("Aantal cleans", min_value=1, max_value=len(df_selection), value=i, key = str(random.random()))
            df_to_show = df_selection.iloc[:j+1]
            data_selection = df_to_show["tijd in minuten"].tolist()
            calculate_and_plot(data_selection, code_, distribution_to_use, True)

    else:
            i = slider_placeholder.slider("Number of cleans to show", min_value=1, max_value=len(df_selection), value=len(df_selection))

            df_to_show = df_selection.iloc[:i]
            data_selection = df_to_show["tijd in minuten"].tolist()
            samenvatting_ = calculate_and_plot(data_selection,code_, distribution_to_use, True)
            st.subheader("brondata")
            st.write(df_to_show.iloc[:, : 7])

def select_data(df, code):
    """Select the rows with the right accotype

    Args:
        df (df): df
        code (str): the acco type to select

    Returns:
        list: list with the cleaning times for the given acco type
    """
    if code == "all":
        df_selection = df.copy(deep=False)
    else:
        df_selection = df[df["Type acco"] == code].copy(deep=False)

    return df_selection["tijd in minuten"].tolist()


def show_various_plots(df, acco_codes, acco_names, distribution_to_use):
    samenvatting =[]
    for code, name in zip (acco_codes, acco_names):
        #print (acco_name[acco_code.index(code)])
        data = select_data(df, code)

        samenvatting_ = calculate_and_plot(data, name, distribution_to_use, False)
        samenvatting.append(samenvatting_)

    df_samenvatting = pd.DataFrame(samenvatting, columns = ['Name', 'number', 'Shape', 'scale', 'mediaan', 'mean data', 'mean calc'])
    st.subheader("Samenvatting")
    try:
        st.write(df_samenvatting.style.format("{:.2}"))
    except:
        st.write(df_samenvatting)

    for code, name in zip (acco_codes, acco_names):
        data = select_data(df, code)
        lineplot(data, name)

    st.subheader("brondata")
    st.write(df.iloc[:, : 7])

def edit_sheet():
    html = '<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQDON7pstUaT3Ftghe6jpDmYQv8iurBHKZbhKE_EYERxIy27KnIPr4zMRmd0FmWThuFanx8HJmr9fr6/pubhtml?widget=true&amp;headers=false"></iframe>'
    html = '<iframe src="https://docs.google.com/spreadsheets/d/1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg/edit#gid=0" width"100%" height="100%"></iframe>'
    st.markdown(html, unsafe_allow_html=True)

def list_accos():
    """Return the acco numbers as list

    Returns:
        list: List with acco numbers
    """
    list_saharas= list(range(1,23))
    list_kalaharis =list(range (637,656))
    list_balis = list(range (621,627))
    list_waikikis = list(range(627,637))
    list_serengeti = list(range(659,668))
    return list_saharas + list_kalaharis+list_balis+list_waikikis+list_serengeti

def check_accos_never_cleaned(df):
    """Which acco's did Rene clean and which one didnt he clean at all?
    Args:
        df (df): the dataframe
    """
    gecleande_accos = df["acco nr"].tolist()

    list_accos = list_accos()

    never_cleaned = ""
    for i in list_accos:
        if i not in gecleande_accos:
            never_cleaned = never_cleaned + str(i) + " - "
    st.write (f"{never_cleaned} is nooit door Rene (geregistreerd) schoongemaakt")
    st.subheader("Wat heeft hij wel gedaan dan?")
    for i in list_accos:
        if i in gecleande_accos:
            aantal_keer = gecleande_accos.count(i)
            st.write (f"{i} is  {aantal_keer} keer door Rene schoongemaakt")


    st.write(df)

def main():

    df = read()

    acco_codes = ["all","w", "sa", "se", "k", "b"]
    acco_names = ["All","Waikiki", "Sahara", "Serengeti", "Kalahari", "Bali"]
    # distributions = ["weibull_min",  "exponweib"]
    # distribution_to_use =  st.sidebar.selectbox(
    #         "Which distribution to use",
    #         distributions,
    #         index=0)
    distribution_to_use = "weibull_min"
    st.title(f"Schoonmaaktijden gefit aan Weibull verdeling")
    menu_choice = st.sidebar.radio("",["ALL", "interactive", "never cleaned", "edit sheet"], index=2)
    if menu_choice == "ALL":
        show_various_plots(df, acco_codes, acco_names, distribution_to_use)
    elif menu_choice == "edit sheet":
        edit_sheet()
    elif menu_choice == "never cleaned":
        check_accos_never_cleaned(df)
    elif menu_choice == "interactive":
        show_animation(df, acco_codes, acco_names, distribution_to_use)
    else:
        st.write(ËRROR)
        st.stop()
    st.sidebar.write("Attention: Guests are supposed to leave the accomodation clean behind as they found it. These cleaning times are in fact 'make perfect'-times !")
    st.sidebar.write("Google sheet : https://docs.google.com/spreadsheets/d/1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg/edit#gid=0")
    st.sidebar.write("Broncode : https://github.com/rcsmit/streamlit_scripts/schoonmaaktijden.py")


if __name__ == "__main__":
    #caching.clear_cache()
    main()
