# Standard library
import math
import time
import bisect
import random
from itertools import cycle

# Third-party libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# For statistics
from statistics import mean

from scipy import stats
from scipy.stats import weibull_min, norm
from scipy.special import gamma

# Streamlit
import streamlit as st

# partly derived from https://stackoverflow.com/a/37036082/4173718

def cleaning_weibull(data):
    """We estimate the cleaning times of the rooms in the hotel using a Weibull distribution, "
            "based on the data collected from the cleaning staff.

    Args:
        data (list): list with claning times (in minutes) of the rooms in the hotel in the past
    """    
    # fitting cleaning times to weibull and give info
    st.info("We estimate the cleaning times of the rooms in the hotel using a Weibull distribution, "
            "based on the data collected from the cleaning staff. "
            "The Weibull distribution is a continuous probability distribution named after Wallodi Weibull, "  )
    # https://chatgpt.com/c/5f499843-4325-4b54-95cb-ebd61f1b6ffd

    # Sample data: cleaning times in minutes
    #data = [56, 18, 15, 30, 34, 30, 7, 15, 44, 23, 50, 20, 50, 18, 19, 32, 40, 36, 26, 30, 8, 15, 19, 23, 30, 74, 25, 34, 28, 61, 22, 13, 14, 70, 38, 31, 29, 31, 42, 62, 7, 40, 56, 28, 35, 12, 13, 13, 7, 30, 23, 42, 36, 38, 30, 25, 13, 55, 40, 40, 10, 10, 16, 27, 17, 15, 43, 27, 30, 22, 10, 27, 48, 30, 53, 24, 58, 11, 17, 26, 13, 86, 26, 40, 25, 13, 17, 47, 51, 41, 9, 13, 29, 5, 22, 15, 20, 75, 54, 40, 11, 34, 35, 37, 36, 39, 41, 40, 33, 28, 57, 45, 16, 12, 33, 27, 14, 26, 16, 18, 19, 70, 15, 11, 46, 35, 20, 22, 60, 7, 67, 28, 14, 15, 49, 20, 20, 40, 26, 20, 19, 26, 83, 22, 32, 29, 20, 14, 15, 38]
    
    p_low, p_high = 0.025, 0.975

    # Fit Weibull
    k, loc, lam = weibull_min.fit(data, floc=0)
    mean_time = lam * gamma(1 + 1/k)
    std_time = np.sqrt(lam**2 * (gamma(1 + 2/k) - gamma(1 + 1/k)**2))

    st.write(f"**Shape (k):** {k:.3f}")
    st.write(f"**Scale (λ):** {lam:.3f}")
    st.write(f"**Mean:** {mean_time:.2f} min")
    st.write(f"**SD:** {std_time:.2f} min")
    st.write(f"**95% CI for 1 acco:** {mean_time - 1.96*std_time:.2f} – {mean_time + 1.96*std_time:.2f} min")

    # Percentiles for one accommodation
    for p in [p_low, p_high, 0.632]:
        t_ = lam * (-np.log(1 - p))**(1/k)
        st.write(f"**{int(p*100)}th percentile:** {t_:.2f} min")

    # Loop over number of accommodations
    results = []
    for n in range(1, 100001, 1000):
        mu_sum = n * mean_time
        sd_sum = np.sqrt(n) * std_time
        p5 = norm.ppf(p_low, loc=mu_sum, scale=sd_sum)
        p95 = norm.ppf(p_high, loc=mu_sum, scale=sd_sum)
        results.append({
            "n_acco": n,
            "5th_percentile": p5,
            "95th_percentile": p95,
            "5th_avg": p5 / n,
            "95th_avg": p95 / n
        })

    df = pd.DataFrame(results)
    st.dataframe(df)

# https://chatgpt.com/c/de742f40-3c1d-41c2-9287-76539aa6cdcf
def calculate_weibull_pdf(x, scale, shape):

    return (shape/scale) * ((x/scale)**(shape - 1)) * np.exp(-1*((x/scale)**shape))


def calculate_weibull_pdf_wrong(x, scale, shape):
    # almost same as PDF, but not the good one I think
    return (((shape / scale) * ((x)/scale))**(shape-1)) * (np.exp(-1* ((x)/scale)**shape))

def calculate_weibull_cdf(x, scale, shape):
    return 1 - (np.exp(- (x/scale)**shape))

def calculate_weibull_cumm_hazard(x, scale, shape):
    return (x/scale)**shape

def calculate_weibul_ppf (p, scale, shape):
    """ Percentual point function


    Args:
        p ([type]): percentage
        scale ([type]): [description]
        shape ([type]): [description]

    Returns:
        [type]: onder de aantal minuten
    """
    #https://www.itl.nist.gov/div898/handbook/eda/section3/eda362.htm#PPF
    q = 1-p
    return scale * (-1 * np.log(q))**(1/shape)

def calculate_weibull_pmf_step(x, scale, shape, step):
    """Probability mass function
        Discrete scale, with steps
    Args:
        x ([type]): [description]
        scale ([type]): [description]
        shape ([type]): [description]
    """
    a = np.exp(-1*(x/scale)**shape)
    b =np.exp(-1*((x+step)/scale)**shape)
    return a-b

def calculate_weibull_cdf_discr(x, scale, shape):
    """Cumulative distribution function
        Discrete scale
    Args:
        x ([type]): [description]
        scale ([type]): [description]
        shape ([type]): [description]
    """
    b =np.exp(-1*((x+1)/scale)**shape)
    return (1-b)


def calculate_weibull_mean(scale,shape):
    n = (1+ (1/shape))
    gamma = math.gamma(n)

    # for t in range (1_000_000):
    #     gamma += t**(n-1)* np.exp(-t)
    return scale*gamma
def calculate_weibull_pdf_not_used(x, scale, shape):
    #if x == 0: return 0

    x_min_1 = 1-np.exp(-1*((x-1/scale)**shape))
    xx = 1-np.exp(-1*((x/scale)**shape))
    return (x_min_1 - xx)



#@st.cache_data(ttl=60 * 60 * 24)
def read():
    # https://docs.google.com/spreadsheets/d/1vWNSJXex0jly1Xq6Mqah9REVVd3tT7v6tEfVFEw8_go/edit?usp=sharing
    sheet_id = "1vWNSJXex0jly1Xq6Mqah9REVVd3tT7v6tEfVFEw8_go"
    sheet_name = "gegevens"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

    #url = "C:\\Users\\rcxsm\\Documents\\pyhton_scripts\\in\\schoonmaaktijden.csv",
    df = pd.read_csv(url, delimiter=',')
    #df = df[:-1]  #remove last row which appears to be a Nan

    df["Datum"] = pd.to_datetime(df["Datum"], format="%d-%m-%Y")
    return df

def lineplot(data, acco_name):
    """Maak een plot Schoonmaaktijden door de tijd heen
    Args:
        data ([type]): [description]
        acco_name ([type]): [description]
    """
    data_serie = pd.Series(data)
    sma = data_serie.rolling(window=5, center=False).mean()
    if 1==1:
    # with _lock:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(data, linestyle="dotted")
        ax.plot(sma)
        title =  (f"Cleaning times through the time - {acco_name} ")
        plt.xlabel("Number of cleans")
        plt.title(title)
        st.pyplot(fig)
        fig = plt.close()

def extra_plots(what, acco_name, data, bins_formula, bins, shape, scale):
    #with st.expander(f"Extra plots {what}" , expanded = False):
    if 1==1:
    # with _lock:
        fig_extra_plot = plt.figure()
        ax = fig_extra_plot.add_subplot(1, 1, 1)
        if what =="PDF":
            ax.plot (bins_formula, calculate_weibull_pdf(bins_formula, scale, shape))
        elif what =="CDF":
            ax.plot (bins_formula, calculate_weibull_cdf(bins_formula, scale, shape))
        elif what =="CHZ":
            ax.plot (bins_formula, calculate_weibull_cumm_hazard(bins_formula, scale, shape))
        elif what =="CDF_disc":
            y = [calculate_weibull_cdf_discr(x, scale, shape) for x in list(bins_formula)]
            ax.bar (bins_formula,  y)
        # ax.hist(y, bins = bins , density=False, alpha=0.5)
        elif what =="PMF":

            y = [calculate_weibull_pmf_step(x, scale, shape,1) for x in list(bins_formula)]
            #ax.plot (bins_formula,  calculate_weibull_pmf(bins_formula, scale, shape))
            ax.bar (bins_formula,  y)

        title =  (f"{what} - {acco_name}\n\nShape: {round(shape,2)} - Scale: {round(scale,2)}")
        plt.grid()
        plt.title(title)
        st.pyplot(fig_extra_plot)
        fig_extra_plot = plt.close()

def plot_weibull_comparison(kind, data, shape, scale, step=5, acco_name="", show_real=True):
    """Generalized function to plot Weibull function vs. reality.

    suggested by chatgpt, not integrated

    https://chatgpt.com/share/685a9ee6-28f4-8004-b66f-349b71211f60

    Args:
        kind (str): 'PDF', 'CDF', 'CHZ', 'PMF', 'CDF_disc'
    """
    bins_formula = list(range(0, int(max(data)) + step + 1, step))
    n = len(data)
    sorted_data = sorted(data)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    if kind == "PDF":
        y_model = calculate_weibull_pdf(np.array(bins_formula), scale, shape)
        ax.plot(bins_formula, y_model, label="PDF (model)")

    elif kind == "CDF":
        y_model = calculate_weibull_cdf(np.array(bins_formula), scale, shape)
        ax.plot(bins_formula, y_model, label="CDF (model)")

    elif kind == "CHZ":
        y_model = calculate_weibull_cumm_hazard(np.array(bins_formula), scale, shape)
        ax.plot(bins_formula, y_model, label="Cumulative Hazard (model)")

    elif kind == "PMF":
        y_model = [calculate_weibull_pmf_step(x, scale, shape, step) * n for x in bins_formula]
        ax.bar(bins_formula, y_model, width=step, alpha=0.5, label="PMF (model)", color="red")

    elif kind == "CDF_disc":
        y_model = [calculate_weibull_cdf_discr(x, scale, shape) for x in bins_formula]
        ax.bar(bins_formula, y_model, width=step, alpha=0.5, label="CDF Discrete (model)", color="red")

    if show_real and kind in {"PMF", "CDF", "CDF_disc"}:
        # Reality curve as histogram
        hist, bin_edges = np.histogram(sorted_data, bins=bins_formula)
        if kind == "PMF":
            ax.bar(bins_formula[:-1], hist, width=step, alpha=0.5, label="Reality", color="yellow")
        else:
            cum_hist = np.cumsum(hist) / n
            ax.bar(bins_formula[:-1], cum_hist, width=step, alpha=0.5, label="Reality CDF", color="yellow")

    ax.set_title(f"{kind} comparison – {acco_name} (shape={round(shape,2)}, scale={round(scale,2)})")
    ax.legend()
    plt.grid()
    st.pyplot(fig)
    plt.close()


def extra_plot_pmf(df, acco_name, data, bins_formula, bins, shape, scale, binwidth):
    """Calculate a plot with the real data compared with the data following the formula with given shape and scale


    Args:
        what ([type]): [description]
        acco_name ([type]): [description]
        data ([type]): [description]
        bins_formula ([type]): [description]
        bins ([type]): [description]
        shape ([type]): [description]
        scale ([type]): [description]
    """
    totaal_aantal = len(df)
    px = [0.10, 0.25,0.5,0.632,0.75,0.9, 0.95,0.99]
    reeks = df["tijd in minuten"].tolist()
    reeks.sort()
    lengte_reeks = len(reeks)

    #reeks_test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23]
    #len_reeks_test = len(reeks_test)

    for p in px:
        y = round(calculate_weibul_ppf (p, scale, shape))
        df_temp = df[df["tijd in minuten"] < y ]
        temp_aantal = len(df_temp)


        st.write (f"{round(p*100)} % of the cleans are done in less than {y} minutes. (Reality {reeks[round(p * lengte_reeks)-1]} minutes)")
        # uitgegaan vd berekende minuten -> (realiteit {round(temp_aantal/totaal_aantal*100,1)} % )

        #st.write (f"TEST Realiteit onder de {reeks_test[round(p * len_reeks_test)-1]} minuten")

    xx = [10,15,30,45,60]
    for x in xx:
        df_temp = df[df["tijd in minuten"] <= x ]
        temp_aantal = len(df_temp)
        st.write (f"{round( 100 * (calculate_weibull_cdf (x, scale, shape)),1)} % of the cleans are done in less than {x} minutes. (reality: {round(temp_aantal/totaal_aantal*100,1)} % = {temp_aantal} acco's) ")
    st.write (f"{round (100 - ( 100 * (calculate_weibull_cdf (x, scale, shape))),1)} % of the cleans need more than {x} minutes. (reality: {round (100 - (temp_aantal/totaal_aantal*100),1)} % = {totaal_aantal - temp_aantal} acco's) ")

    if 1==1:
    # with _lock:
        bins_new = []
        y_new, y_reality = [],[]
        j = 0

        fig_extra_plot = plt.figure()
        if binwidth == None:
            step =  round(max(data) / 10)
        else:
            step = binwidth


        for i in range(((reeks[-1])+1+step)):

            if i % step == 0 or i== (reeks[-1]+step+1):

                y_ = len(data) * calculate_weibull_pmf_step(i, scale, shape, step)
                bins_new.append (i)
                y_new.append(y_)

                lengte_selectie = find_ge(reeks, j,i)
                cumm_y =+ lengte_selectie / lengte_reeks

                y_reality.append(cumm_y * lengte_reeks )
                j = i+1


        plt.bar(bins_new, y_new, align="center", width=step,alpha=0.5, label = "PMF_formula", color = "red")
        #plt.hist(data, bins = bins , density=False, alpha=0.5, label = "PMF_reality", color = "yellow")
        plt.bar(bins_new, y_reality, align="center", width=step, alpha=0.5, label = "PMF_reality", color = "yellow")
        title =  (f"Reality vs. PMF - {acco_name} (n={len(data)})\n\nShape: {round(shape,2)} - Scale: {round(scale,2)}")
        # secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
        # secax.set_xlabel('angle [rad]')
        #plt.show()
        plt.grid()
        plt.legend()
        plt.title(title)
        st.pyplot(fig_extra_plot)
        fig_extra_plot = plt.close()
        # correlation, p_value = stats.pearsonr(data, y_new) #first I have te rework the data in frequencies




def find_ge(a, low, high):
    i = bisect.bisect_left(a, low)
    g = bisect.bisect_right(a, high)
    if i != len(a) and g != len(a):
        # return a[i:g]
        #st.write (f"{low}  {high}  {i} {g}  {a[i:g]}")
        return len(a[i:g])
    else:
        #st.write (f"{low}  {high}  {i} {g}  {a[i:g]} {len(a)} {len(a[i:g])}")
        return len(a[i:g])


def extra_plot_cdf(df, acco_name, data, bins_formula, bins, shape, scale, binwidth):
    """Calculate a plot with the real data compared with the data following the formula with given shape and scale


    Args:
        what ([type]): [description]
        acco_name ([type]): [description]
        data ([type]): [description]
        bins_formula ([type]): [description]
        bins ([type]): [description]
        shape ([type]): [description]
        scale ([type]): [description]
    """
    totaal_aantal = len(df)

    reeks = df["tijd in minuten"].tolist()
    reeks.sort()
    lengte_reeks = len(reeks)

    #reeks_test = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,23]
    #len_reeks_test = len(reeks_test)


    if 1==1:
    # with _lock:
        bins_new = []
        y_new, y_reality = [],[]
        fig_extra_plot = plt.figure()
        cumm_y = 0
        j=0

        if binwidth == None:
            step =  round(max(data) / 10)
        else:
            step = binwidth


        for i in range((max(data)+1+binwidth)):
            temp = 0
            if i % step == 0:
                y_ = calculate_weibull_cdf_discr(i, scale, shape)

                bins_new.append (i)
                y_new.append(y_)
                temp = 0

                lengte_selectie = find_ge(reeks, j,i)
                cumm_y =+ lengte_selectie / lengte_reeks

                y_reality.append(cumm_y)

        plt.bar(bins_new, y_new, align="center", width=step,alpha=0.5, label = "CDF_formula", color = "red")
        plt.bar(bins_new, y_reality, align="center", width=step, alpha=0.5, label = "CDF_reality", color = "yellow")
        #plt.hist(data, bins = bins , density=False, alpha=0.5, label = "reality", color = "yellow")
        title =  (f"Reality vs. CDF - {acco_name} (n={len(data)})\n\nShape: {round(shape,2)} - Scale: {round(scale,2)}")

        plt.grid()
        plt.legend()
        plt.title(title)
        st.pyplot(fig_extra_plot)
        fig_extra_plot = plt.close()
        # correlation, p_value = stats.pearsonr(data, y_new) #first I have te rework the data in frequencies



def calculate_and_plot(df_selection, data, acco_name, modus, animation, binwidth):
    """[summary]

    Args:
        data (list): [description]
        acco_name (string): [description]
        modus (string): exponweib or weib_min
        animation (boolean): make animation?
        binwidth :
    Returns:
        samenvatting: list with the calculated [acco_name, len(data), shape, scale , mediaan, mean_data, mean_calc]
    """

    if modus == "exponweib":
        a_in = 1 # α = 1 gives the Weibull distribution;
        loc_in = 0
        a_out, Kappa_out, loc_out, Lambda_out = stats.exponweib.fit(data, f0=a_in,floc=loc_in)
    else:
        shape, loc, scale = stats.weibull_min.fit(data, floc=0)
        Kappa_out = shape
        Lambda_out = scale
    #Plot
    bins_formula = range( int(max(data))+1)
    #binwidth = max(data)/10

    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    if 1==1:
    # with _lock:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax3 = ax.twinx()
        if modus == "exponweib":
            ax3.plot(bins_formula, stats.exponweib.pdf(bins_formula, a=a_out,c=Kappa_out,loc=loc_out,scale = Lambda_out))
        else:

            # JUST TO TEST IF THE FORMULA IS GOOD ax3.plot(bins_formula, stats.weibull_min(shape, loc, scale).pdf(bins_formula), color = "yellow", label = "scipi", alpha = 0.5 )
            # WRONG ax3.plot (bins_formula, calculate_weibull_pdf_wrong(bins_formula, scale, shape), color = "red", label = "old/pdf_wrong", alpha = 0.5)
            ax3.plot (bins_formula, calculate_weibull_pdf(bins_formula, scale, shape), color = "blue", label = "pdf", alpha = 0.5)
            pass
        ax.hist(data, bins = bins , density=False, alpha=0.5, label = "reality")

        mediaan =Lambda_out *(np.log(2) **(1/Kappa_out))
        mean_data = mean(data)
        mean_calc =calculate_weibull_mean (scale, shape)
        title =  (f"{acco_name} (n={len(data)})\n\nShape: {round(Kappa_out,2)} - Scale: {round(Lambda_out,2)}\nMediaan : {round(mediaan,2)} - mean data : {round(mean_data,2)} -  - mean calc : {round(mean_calc,2)}")
        samenvatting = [acco_name, len(data), round(Kappa_out,2), round(Lambda_out,2), round(mediaan,2), round(mean_data,2), round(mean_calc,2)]
        plt.title(title)
        plt.grid()
        plt.legend()

        if animation ==True:
            placeholder.pyplot(fig)
            fig = plt.close()
        else:
            st.pyplot(fig)
            fig = plt.close()
            with st.expander(f"Extra plots {acco_name}" , expanded = False):
                
                extra_plot_pmf(df_selection, acco_name, data, bins_formula, bins, shape, scale, binwidth)

                extra_plot_cdf(df_selection, acco_name, data, bins_formula, bins, shape, scale, binwidth)
                # what_list = ["PDF", "CDF", "CHZ"] PDF and CHZ doesnt have an added value for now

                what_list = ["PDF", "CDF","PMF", "CDF_disc",  "CHZ",]
                for what in what_list:
                    extra_plots(what, acco_name, data, bins_formula, bins, shape, scale)
                st.write(df_selection)
    return samenvatting


def show_animation(df, acco_codes, acco_names, distribution_to_use,binwidth ):

    code_ =  st.selectbox("Which accotype to show", acco_names, index=0)
    code = acco_codes[acco_names.index(code_)]
    df_selection, data_selection = select_data(df, code)

    global placeholder
    animations = {"None": None, "Slow": 0.4, "Medium": 0.2, "Fast": 0.05}
    animate = st.sidebar.radio("", options=list(animations.keys()), index=2)
    animation_speed = animations[animate]

    slider_placeholder = st.empty()
    placeholder = st.empty()


    if animation_speed:

        c = range(1,len(data_selection)+1)
        for i in cycle(c):
            time.sleep(animation_speed)

            #TO FIX:  stap 1 wordt overgeslagen.
            j = slider_placeholder.slider("Aantal cleans", min_value=1, max_value=len(data_selection), value=i, key = str(random.random()))
            # df_to_show = df_selection.iloc[:j+1]
            # data_selection = df_to_show["tijd in minuten"].tolist()
            data_selection_ = data_selection[:j+1]
            calculate_and_plot(df_selection, data_selection_, code_, distribution_to_use, True, binwidth)

    else:
            i = slider_placeholder.slider("Number of cleans to show", min_value=1, max_value=len(data_selection), value=len(data_selection))

            # df_to_show = df_selection.iloc[:i]
            # data_selection = df_to_show["tijd in minuten"].tolist()
            data_selection_ = data_selection[:i]
            samenvatting_ = calculate_and_plot(df_selection, data_selection_,code_, distribution_to_use, True, binwidth)
            # st.subheader("brondata")
            # st.write(df_to_show.iloc[:, : 7])

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
    elif code =="no_tents":
        df_selection = df[~df["Type acco"].isin(["ecolodge", "safari"])].copy(deep=False)
    elif code =="ex_vs":
        df_selection = df[df["Type acco"].isin(["prem_az_vs", "comf_xl"])].copy(deep=False)
    elif code =="mh_non_vs":
        df_selection = df[df["Type acco"].isin(["prem_az", "prem_lounge",  "comfort"])].copy(deep=False)



    else:
        df_selection = df[df["Type acco"] == code].copy(deep=False)

    return df_selection, df_selection["tijd in minuten"].tolist()


def show_various_plots(df, acco_codes, acco_names, distribution_to_use, binwidth):
    samenvatting =[]
    for code, name in zip (acco_codes, acco_names):
        #print (acco_name[acco_code.index(code)])
        df_selection, data = select_data(df, code)

        samenvatting_ = calculate_and_plot(df_selection, data, name, distribution_to_use, False, binwidth)
        samenvatting.append(samenvatting_)

    df_samenvatting = pd.DataFrame(samenvatting, columns = ['Name', 'number', 'Shape', 'scale', 'mediaan', 'mean data', 'mean calc'])
    st.subheader("Samenvatting")
    try:
        st.write(df_samenvatting.style.format("{:.2}"))
    except:
        st.write(df_samenvatting)

    for code, name in zip (acco_codes, acco_names):
        df_selection, data = select_data(df, code)
        lineplot(data, name)

    st.subheader("brondata")
    st.write(df.iloc[:, : 7])

def edit_sheet():
    html = '<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vQDON7pstUaT3Ftghe6jpDmYQv8iurBHKZbhKE_EYERxIy27KnIPr4zMRmd0FmWThuFanx8HJmr9fr6/pubhtml?widget=true&amp;headers=false"></iframe>'
    html = '<iframe src="https://docs.google.com/spreadsheets/d/1Lqddg3Rsq0jhFgL5U-HwvDdo0473QBZtjbAp9ol8kcg/edit#gid=0" width"100%" height="100%"></iframe>'
    st.markdown(html, unsafe_allow_html=True)

def make_list_accos():
    """Return the acco numbers as list

    Returns:
        list: List with acco numbers
    """
    list_various= list(range(1,53))
    list_kalaharis =list(range (637,656))
    list_balis = list(range (621,627))
    list_waikikis = list(range(627,637))
    list_serengeti = list(range(659,668))
    return list_various + list_kalaharis+list_balis+list_waikikis+list_serengeti

def check_accos_never_cleaned(df):
    """Which acco's did Rene clean and which one didnt he clean at all?
    Args:
        df (df): the dataframe
    """
    gecleande_accos = df["acco nr"].tolist()
   
    list_accos = make_list_accos()

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
    data = df["tijd in minuten"].tolist()
    acco_codes = ["all","no_tents","ex_vs","mh_non_vs", "prem_az_vs","prem_az", "prem_lounge", "ecolodge", "comf_xl", "comfort"]
    acco_names = ["All","no_tents","ex_vs","mh_non_vs", "prem_az_vs","prem_az","prem_lounge", "ecolodge", "comf_xl", "comfort"]

    # distributions = ["weibull_min",  "exponweib"]
    # distribution_to_use =  st.sidebar.selectbox(
    #         "Which distribution to use",
    #         distributions,
    #         index=0)
    # exponweib doesnt work properly

    distribution_to_use = "weibull_min"
    # distribution_to_use = "exponweib"

    st.title(f"Schoonmaaktijden gefit aan Weibull verdeling")
    menu_choice = st.sidebar.radio("",["ALL", "animated", "never cleaned", "edit sheet","simple", "show formulas"], index=0)
    binwidth = st.sidebar.slider("Binwidth", 1, 20, 6)
    st.sidebar.write("Attention: Guests are supposed to leave the accomodation with basic (cleaning)tasks done. These cleaning times are in fact 'make perfect'-times !")
    st.sidebar.write("Google sheet :https://docs.google.com/spreadsheets/d/1vWNSJXex0jly1Xq6Mqah9REVVd3tT7v6tEfVFEw8_go/edit?gid=0#gid=0")
    st.sidebar.write("Broncode : https://github.com/rcsmit/streamlit_scripts/schoonmaaktijden.py")

    if menu_choice == "ALL":
        show_various_plots(df, acco_codes, acco_names, distribution_to_use, binwidth)
    elif menu_choice == "edit sheet":
        edit_sheet()
    elif menu_choice == "never cleaned":
        check_accos_never_cleaned(df)
    elif menu_choice == "animated":
        show_animation(df, acco_codes, acco_names, distribution_to_use, binwidth)

    elif menu_choice == "show formulas":

        st.header("Formulas")

        #st.write ("distribution: y =  (shape/scale) * ((x/scale)**(shape - 1)) * np.exp(-1*((x/scale)**shape)) ")

        st.write ("PDF - probability density function : y = (shape/scale) * ((x/scale)**(shape - 1)) * np.exp(-1*((x/scale)**shape))")


        st.write ("CDF - cummulative distribution function: y = 1 - (np.exp(- (x/scale)**shape))")
        st.subheader("From percentage to time (x % of the cleans is under y minutes)")
        st.write ("PPF - Percentual point function: q = 1-p | y =  scale * (-1 * np.log(q))**(1/shape)")
        st.subheader("Discrete / steps")
        st.write ("PMF - probability mass function :     a = np.exp(-1*(x/scale)**shape) |     b =np.exp(-1*((x+step)/scale)**shape) | y = a-b")
        st.write ("CDF - Cummulative distribution function :    b =np.exp(-1*((x+1)/scale)**shape) | y = (1-b)")
        st.subheader("Various")
        st.write ("cumm_hazard : y = (x/scale)**shape")
        st.write ("mean : n = (1+ (1/shape)) |  gamma = math.gamma(n) | y = scale*gamma")
        st.write ("pdf_not_used :    x_min_1 = 1-np.exp(-1*((x-1/scale)**shape)) |     xx = 1-np.exp(-1*((x/scale)**shape))| y = (x_min_1 - xx)")
        st.subheader("Extra info")
        st.write(" the shape parameter describes the shape of your data’s distribution. Statisticians also refer to it as the Weibull slope because its value equals the slope of the line on a probability plot. Shape value of 2 equals a Rayleigh distribution, which is equivalent to a Chi-square distribution with two degrees of freedom. Shape value near of 3 approximates the normal distribution")

        st.write ("The scale parameter represents the variability present in the distribution. The value of the scale parameter equals the 63.2 percentile in the distribution. 63.2% of the values in the distribution are less than the scale value.")
        st.write("https://statisticsbyjim.com/probability/weibull-distribution/")
        st.subheader("Links")
        st.write("https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weibull_min.html")
        st.write("https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Weibull.html")
        st.write("https://www.sciencedirect.com/topics/computer-science/weibull-distribution")
        st.write("https://www.itl.nist.gov/div898/handbook/eda/section3/eda3668.htm")
    elif  menu_choice == "simple":
        cleaning_weibull(data)
    else:
        st.write("ËRROR")
        st.stop()
if __name__ == "__main__":
    #caching.clear_cache()
    main()
