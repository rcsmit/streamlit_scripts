import streamlit as st

import numpy as np
import pandas as pd
import statsmodels.api as sm

# import scipy.stats as stats
from scipy.stats import norm

from scipy.stats import t
from scipy.stats import sem


import plotly.graph_objects as go
import plotly.express as px
from scipy.linalg import qr, pinv   
from scipy.linalg import solve_triangular

import time
from math import ceil
import numpy as np
from scipy import linalg
from skmisc.loess import loess

def show_info():
    """Show an introduction text for the page
    """    
    st.header("Yearly temperature data")
    st.subheader("Trendline with local lineair regression - LOESS")

    st.info(
            """
        The trendline can be regarded as an approximation of a 30-year average, which has a smooth appearance
        and is extended toward the beginning and end of the time-series.

        It is based on linear local regression, computed using the scikit-misc library. It uses a bicubic weight
        function over a 42-year window. In the central part of the time-series, the variance of the trendline
        estimate is approximately equal to the variance of a 30-year average.

        To test the proposition of no long-term change between the years t1 and t2, these years need to be supplied.
        The result is the p-value: the probability (under the proposition) that the estimated trendline values in
        t2 and t1 differ more than observed.
        """
        )


def show_plot_plotly(title, what_to_show, t, values_, trend, trendlb, trendub, avt=None, avy=None,  draw30=False):
    """Draw a plot with the results with plotly

    Args:
        title (string): title of the graph
        what_to_show (str): which parameter to show ("temp_avg")
        t (int): years
        trend (float): the values of the trendline
        trendlb (float): the values of the trendline 95% lower bound
        trendub (float): the values of the trendline 95% upper bound
        avt (float, optional): year values of the 30 year SMA. Defaults to None.
        avy (float, optional): values of the 30 year SMA. Defaults to None.
        draw30 (bool, optional): whether to draw the 30 year SMA. Defaults to False.
       
    """
    if draw30:
        av = go.Scatter(
            name=f"{what_to_show} avg 30 jaar",
            x=avt,
            y=avy,
            # mode='lines',
            line=dict(width=1, color="rgba(0, 0, 0, 1)"),
        )
    loess = go.Scatter(
        name=f"{what_to_show} Loess",
        x=t,
        y=trend,
        mode="lines",
        line=dict(width=1, color="rgba(255, 0, 255, 1)"),
    )
    
    loess_low = go.Scatter(
        name=f"{what_to_show} Loess low",
        x=t,
        y=trendlb,
        mode="lines",
        line=dict(width=0.7, color="rgba(255, 0, 255, 0.5)"),
    )
    loess_high = go.Scatter(
        name=f"{what_to_show} Loess high",
        x=t,
        y=trendub,
        mode="lines",
        line=dict(width=0.7, color="rgba(255, 0, 255, 0.5)"),
    )
    # Create a filled area plot for confidence interval
    confidence_trace = go.Scatter(x=np.concatenate([t, t[::-1]]),
                            y=np.concatenate([trendub, trendlb[::-1]]),
                                fill='tozeroy',
                                fillcolor='rgba(0, 128, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                showlegend=True,
                                name="CI of the trendline")

    values = go.Scatter(
        name=what_to_show,
        x=t,
        y=values_,
        mode="lines",
        line=dict(width=1, color="rgba(0, 0, 255, 0.6)"),
    )

    data = [values, loess]
   
    data.append(confidence_trace)
    if draw30:
        data.append(av)
    
    # Find the bounds for the graph. Have to be calculated since the there is a fill between the CI-limits
    # using (fill='tozeroy')
    A_1d = np.ravel(values_)
    B_1d = np.ravel(trendub)
    C_1d = np.ravel(trendlb)
   
    try:
        Y_values = A_1d.tolist() + B_1d.tolist() + C_1d.tolist()
        y_lower_bound = min(Y_values)    
        y_upper_bound = max(Y_values)
    except:
        y_lower_bound = values_.min()    
        y_upper_bound = values_.max()
    
    layout = go.Layout(
        yaxis=dict(title=what_to_show, range=[y_lower_bound, y_upper_bound ]), title=f"Year averages of {what_to_show} - {title}"
    )
 
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, use_container_width=True)
    
    
def getdata(url):
    try:
        df = pd.read_csv(
            url,
            delimiter=",",
            header=0,
            comment="#",
            low_memory=False,
        )

    except:
        st.error("FOUT BIJ HET INLADEN.")
        st.stop()
    return df

def main_skmisc(t, y, t1,t2, draw30, p=None, ybounds=None, it=1):
    """Make a plot and calculate p-value with scikit-misc

    Args:
        t : list of Years, increasing by 1.
        y : list of Annual values
        t1 : float, optional
            First year for which trendline value is compared in the test.
        t2 : float, optional
             year (see t1) for which trendline value is compared in the test. Must be >30 higher than t1
        draw30 : bool, optional
            If True, add 30-year moving averages to the plot (default: False).
        ybounds : list or array-like, optional
            Lower/upper bound on the value range of y (default: [-Inf, Inf]).
        it : number of iterations
     
    Returns:
        df : dataframe with the results

    Relevant code in R
        # fixed parameters
        width <- 42
        control <- loess.control(surface = "direct", statistics= "exact",
                           iterations= 1)
        # linear LOESS trendline computation
        span <- width/ng
        mdl <- loess(y ~ t, data= data.frame(t= tg, y=yg), span= span,
                    degree= 1, control= control)
        # mdl <- loess(y ~ t, data= data.frame(t= t, y=y), span= span, degree= 1)
        pre <- predict(mdl, newdata= data.frame(t= t), se= TRUE)
        trend <- pre$fit          # trendline
        trendsd <- pre$se.fit     # standard deviation of trendline
    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python

    st.subheader("Lowess with SciKit-Misc")

    # Set default values for p, t1, and t2
    if p is None:
        p = 0.95  # default confidence level
    if t1 is None or t2 is None:
        t1 = np.inf
        t2 = -np.inf

    # Set default value for ybounds
    if ybounds is None:
        ybounds = [-np.inf, np.inf]
    elif len(ybounds) != 2:
        ybounds = [-np.inf, np.inf]

    ybounds = sorted(ybounds)

    # Dimensions and checks
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dt = np.diff(t)[0]
    n = len(y)
    ig = ~np.isnan(y)
    yg = y[ig]
    tg = t[ig]
    ng = sum(ig)

    if ng <= 29:
        raise ValueError("Insufficient valid data (less than 30 observations).")

    # Check values of bounds
    if np.any(yg < ybounds[0]) or np.any(yg > ybounds[1]):
        raise ValueError("Stated bounds are not correct: y takes values beyond bounds.")

    # Averages over 30 time-steps
    avt, avy, avysd = None, None, None
    if ng > 29:
        avt = tg + dt / 2  # time (end of time-step, for 30-year averages)
        avy = np.convolve(yg, np.ones(30) / 30, mode="valid")
        avy2 = np.convolve(yg**2, np.ones(30) / 30, mode="valid")
        avysd = np.sqrt(avy2 - avy**2)
        ind = slice(
            15, ng - 14
        )  # was (15, ng-15) but gives an error, whether the df has an even or uneven length
        # [ValueError: x and y must have same first dimension, but have shapes (92,) and (93,)]
        avt = avt[ind]
        # avy = avy[ind]            # takes away y values, gives error
        # [ValueError: x and y must have same first dimension, but have shapes (93,) and (78,)]
        avysd = avysd[ind]


    span = 42/len(y)
    l = loess(t,y)
    
    
    # MODEL and CONTROL. Essential for replicating the results from the R script.
    #
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_model.html#skmisc.loess.loess_model
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_control.html#skmisc.loess.loess_control
   
    l.model.span = span
    l.model.degree = 1
    l.control.iterations = it # must be 1 for replicating the R-script
    l.control.surface = "direct"
    l.control.statistics = "exact"

    l.fit()
    pred = l.predict(t, stderror=True)
    conf = pred.confidence()


    ste = pred.stderr
    lowess = pred.values
    ll = conf.lower
    ul = conf.upper
    
    title,what_to_show ="scikit-misc", "temp_avg"
    show_plot_plotly(title, what_to_show, t, y, lowess, ll, ul, avt, avy,  draw30=draw30)
   
    if t2 in t and t1 in t and t2 >= t1 + 30:
        idx_t1 = np.where(t == t1)[0][0]
        idx_t2 = np.where(t == t2)[0][0]
        y1 = lowess[idx_t1]#[0]
        y2 = lowess[idx_t2]#[0]
        y1sd = ste[idx_t1]#[0]
        y2sd = ste[idx_t2]#[0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2


        if pvalue != None:
            p_txt = (f"pvalue: {round(pvalue,4)}")
            if pvalue <0.05:
                st.info(f"**{p_txt}**\n\nThe data indicates a long time change between {t1} and {t2}.")
            else:
                st.info(f"**{p_txt}**\n\nThe data does not indicate (or a little) a long time change between {t1} and {t2}.")

    data = {'YYYY': t, 'skmisc_loess': lowess, 'skmisc_low': ll, 'skmisc_high':ul}

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)
    return df

def show_footer():
          
    st.info( """
        
        References:
        * https://www.knmi.nl/kennis-en-datacentrum/achtergrond/standaardmethode-voor-berekening-van-een-trend
        * KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

        Original Source code:
        https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
        Version: 09-Mar-2021
        Author R script: Cees de Valk (cees.de.valk@knmi.nl)

        **Python version ***
        translated from R to Python by ChatGPT and adapted by Rene Smit.
        Source: https://github.com/rcsmit/streamlit_scripts/blob/main/loess_scikitmisc.py
        Various algoritms and packages have been tested (*). Only the scikit-misc gives the same results
        as the R-script, with a small deviation  (**)

        (*) See https://rcsmit-streamlit-scripts-menu-streamlit-fiaxhp.streamlit.app/?choice=22
    
        """  )

def interface():
    what_to_show = "temp_avg" 
    draw30 = st.sidebar.selectbox("Show 30 year SMA", [True, False], 1)
    test_trend= st.sidebar.selectbox("Two-sided test for absence of trend", [True, False], 1)

    if test_trend:
        t1 = st.sidebar.number_input("t1",1901,2050, 1950)
        t2 = st.sidebar.number_input("t1", 1901,2050,2000)
        if t2<(t1+30): # or t1 not in year_list  or t2 not in year_list:
            st.error("t2 must be 30 years later than t1 and both years have to be in the dataframe")
            st.stop()
    else:
        t1,t2 = None, None

    return what_to_show,draw30,t1,t2

def main():
    show_info()

    what_to_show, draw30, t1, t2,   = interface()
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_jaargem_1901_2022.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/de_bilt_jaargem_1901_2022.csv" 
    
    df = getdata(url)
    what_to_show = "temp_avg"
    X_array = df["YYYY"].values
    Y_array = df[what_to_show].values
    
    df_out = main_skmisc(X_array, Y_array,t1,t2,draw30)

    #df_out.to_csv("results.csv")
   
    show_footer()

    return 



if __name__ == "__main__":
    main()