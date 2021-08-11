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


def read():
    sheet_id = "1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg"
    sheet_name = "gegevens"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    #url = "C:\\Users\\rcxsm\\Documents\\phyton_scripts\\in\\schoonmaaktijden.csv",
    df = pd.read_csv(url, delimiter=',')
    df = df[:-1]  #remove last row which appears to be a Nan

    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    return df

def calculate_and_plot(data, acco_name, modus):

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
        st.pyplot(fig)



    return samenvatting

def main():
    df = read()

    acco_code = ["all","w", "sa", "se", "k", "b"]
    acco_name = ["All","Waikiki", "Sahara", "Serengeti", "Kalahari", "Bali"]
    # distributions = ["weibull_min",  "exponweib"]
    # distribution_to_use =  st.sidebar.selectbox(
    #         "Which distribution to use",
    #         distributions,
    #         index=0)
    distribution_to_use = "weibull_min"
    st.title(f"Schoonmaaktijden gefit aan Weibull verdeling")

    samenvatting =[]
    for code, name in zip (acco_code, acco_name):
        #print (acco_name[acco_code.index(code)])
        if code == "all":
            df_selection = df.copy(deep=False)
        else:
            df_selection = df[df["Type acco"] == code].copy(deep=False)

        data_selection = df_selection["tijd in minuten"].tolist()


        samenvatting_ = calculate_and_plot(data_selection, name, distribution_to_use)
        samenvatting.append(samenvatting_)

    df_samenvatting = pd.DataFrame(samenvatting, columns = ['Name', 'number', 'Shape', 'scale', 'mediaan', 'mean data', 'mean calc'])
    st.subheader("Samenvatting")
    try:
        st.write(df_samenvatting.style.format("{:.2}"))
    except:
        st.write(df_samenvatting)
    st.subheader("brondata")
    st.write(df.iloc[:, : 7])

    st.write("Google sheet : https://docs.google.com/spreadsheets/d/1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg/edit#gid=0")
    st.write("Broncode : https://github.com/rcsmit/streamlit_scripts/schoonmaaktijden.py")

if __name__ == "__main__":
    #caching.clear_cache()
    main()
