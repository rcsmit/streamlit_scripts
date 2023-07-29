import streamlit as st

import numpy as np
import pandas as pd
import statsmodels.api as sm

# import scipy.stats as stats
from scipy.stats import norm
import matplotlib.pyplot as plt


import plotly.graph_objects as go


def show_info():
    st.header("Yearly temperature data")
    st.subheader("Trendline with local lineair regression - LOESS")

    st.info(
        """
    The trendline can be regarded as an approximation of a 30-year average, which has a smooth appearance
    and is extended toward the beginning and end of the time-series.

    It is based on linear local regression, computed using the statsmodels library. It uses a bicubic weight
    function over a 42-year window. In the central part of the time-series, the variance of the trendline
    estimate is approximately equal to the variance of a 30-year average.

    To test the proposition of no long-term change between the years t1 and t2, these years need to be supplied.
    The result is the p-value: the probability (under the proposition) that the estimated trendline values in
    t2 and t1 differ more than observed.

    Version: 09-Mar-2021

    References:
    https://www.knmi.nl/kennis-en-datacentrum/achtergrond/standaardmethode-voor-berekening-van-een-trend
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    Author: Cees de Valk (cees.de.valk@knmi.nl)

    Original Source code:
    https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
    translated from R to Python by ChatGPT and adapted by Rene Smit
    not tested 100%, p-value for trend doesnt work
    """
    )


def climatrend(
    t, y, p=None, t1=None, t2=None, ybounds=None, drawplot=False, draw30=False
):

    """
    Fit a trendline to an annually sampled time-series by local linear regression (LOESS)

    Parameters:
    t : numpy array of shape (n,)
        Years, increasing by 1.
    y : numpy array of shape (n,)
        Annual values; missing values as blanks are allowed near the beginning and end.
    p : float, optional
        Confidence level for error bounds (default: 0.95).
    t1 : float, optional
        First year for which trendline value is compared in the test.
    t2 : float, optional
        Second year (see t1) for which trendline value is compared in the test. Must be >30 higher than t1
    ybounds : list or array-like, optional
        Lower/upper bound on the value range of y (default: [-Inf, Inf]).
    drawplot : bool or str, optional
        If True, a plot will be drawn.
        // If a string is provided, it will be used as the label on the y-axis. - - Not in the Python script //
        (default: False).
    draw30 : bool, optional
        If True, add 30-year moving averages to the plot (default: False).

    Returns:
    pandas DataFrame or dictionary
        A DataFrame or dictionary with the following columns/values:
            't': years,
            'trend': trendline in y for years in t,
            'p': confidence level,
            'trendubound': lower confidence limit,
            'trendlbound': upper confidence limit,
            'averaget': central value of t in a 30-year interval,
            'averagey': 30-year average of y,
            't1': first year for which trendline value is compared in the test,
            't2': second year for which trendline value is compared in the test,
            'pvalue': p-value of the test of no long-term change,
            'ybounds': bounds on the value range of y.

    Details:
    The trendline can be regarded as an approximation of a 30-year average, which has a smooth appearance
    and is extended toward the beginning and end of the time-series.

    It is based on linear local regression, computed using the statsmodels library. It uses a bicubic weight
    function over a 42-year window. In the central part of the time-series, the variance of the trendline
    estimate is approximately equal to the variance of a 30-year average.

    To test the proposition of no long-term change between the years t1 and t2, these years need to be supplied.
    The result is the p-value: the probability (under the proposition) that the estimated trendline values in
    t2 and t1 differ more than observed.

    Version: 09-Mar-2021

    References:
    KNMI Technical report TR-389 (see http://bibliotheek.knmi.nl/knmipubTR/TR389.pdf)

    Author: Cees de Valk (cees.de.valk@knmi.nl)

    # https://gitlab.com/cees.de.valk/trend_knmi/-/blob/master/R/climatrend.R?ref_type=heads
    # translated from R to Python by ChatGPT and adapted by Rene Smit
    # not tested 100%
    # P value for trend doesnt work

    """

    # Fixed parameters
    width = 42

    # Check input -> gives error

    if t is None or y is None or len(t) < 3 or len(t) != len(y):
        raise ValueError("t and y arrays must have equal lengths greater than 2.")

    if np.isnan(t).any() or np.isnan(y).sum() > 3:
        # raise ValueError("t or y contain too many NA.")
        st.error("t or y contain too many NA.")
        st.stop()

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
        )  # was (15, ng-15) but gives an error, wheteher the df has an even or uneven length
        # [ValueError: x and y must have same first dimension, but have shapes (92,) and (93,)]
        avt = avt[ind]
        # avy = avy[ind]            # takes away y values, gives error
        # [ValueError: x and y must have same first dimension, but have shapes (93,) and (78,)]
        avysd = avysd[ind]

    # Linear LOESS trendline computation
    span = width / ng
    loess_model = sm.nonparametric.lowess(yg, tg, frac=span, return_sorted=False)
    trend = loess_model
    

    # Calculate the residuals (difference between yg and trend) and then get the standard deviation for each point
    residuals = yg - trend
    trendsd = np.std(residuals, axis=0)

    z = 1.96  # 1.96 corresponds to a 95% confidence interval (z-score)
    trendub = trend + z * trendsd
    trendlb = trend - z * trendsd

    # Apply bounds
    trend = np.clip(trend, ybounds[0], ybounds[1])
    trendub = np.clip(trendub, ybounds[0], ybounds[1])
    trendlb = np.clip(trendlb, ybounds[0], ybounds[1])

    # p-value for trend
    pvalue = None
    if t2 in t and t1 in t and t2 >= t1 + 30:
        
        idx_t1 = np.where(t == t1)[0][0]
        
        idx_t2 = np.where(t == t2)[0][0]
        y1 = trend[idx_t1]#[0]
        y2 = trend[idx_t2]#[0]
        y1sd = trendsd #[idx_t1]#[0]
        y2sd = trendsd #[idx_t2]#[0]
        # Two-sided test for absence of trend
        pvalue = (1 - norm.cdf(abs(y2 - y1), scale=np.sqrt(y1sd**2 + y2sd**2))) * 2
   

    # Plotting, left it here to stay close to the original script. More logical is to call it from main()
    if drawplot:
        drawplot_matplotlib(t, y, draw30, avt, avy, trend, trendub, trendlb)

    return t, trend, trendlb, trendub, avt, avy, p, t1, t2, pvalue


def drawplot_matplotlib(t, y, draw30, avt, avy, trend, trendub, trendlb):
    """Draws the plot with matplotlib, like in the original R script


    Args:
        t (_type_): _description_
        y (_type_): _description_
        draw30 (_type_): _description_
        avt (_type_): _description_
        avy (_type_): _description_
        trend (_type_): _description_
        trendub (_type_): _description_
        trendlb (_type_): _description_

    """
    fig = plt.figure(figsize=(8, 6))
    ylim = [np.min([np.min(y), np.min(trendlb)]), np.max([np.max(y), np.max(trendub)])]
    ylim[1] = ylim[0] + (ylim[1] - ylim[0]) * 1.0
    plt.plot(t, y, "b-", label="Temperature Data")
    plt.plot(t, trend, "r-", lw=2, label="Trendline")
    plt.fill_between(
        t, trendlb, trendub, color="grey", alpha=0.5, label="Confidence Interval"
    )

    if draw30:
        plt.plot(avt, avy, "ko", markersize=3, label="30-yr Average")

    plt.xlabel("Year")
    plt.ylabel("Temperature")
    plt.grid()
    plt.legend()
    plt.show()
    st.pyplot(fig)


def show_plot_plotly(
    df, what_to_show, t, trend, trendlb, trendub, avt, avy, p, t1, t2, pvalue, draw30
):

    """Shows the plot"""

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

    values = go.Scatter(
        name=what_to_show,
        x=df["YYYY"],
        y=df[what_to_show],
        mode="lines",
        line=dict(width=1, color="rgba(0, 0, 255, 0.6)"),
    )

    data = [values, loess, loess_low, loess_high]
    if draw30:
        data.append(av)

    layout = go.Layout(
        yaxis=dict(title=what_to_show), title=f"Jaargemiddeldes van {what_to_show}"
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, use_container_width=True)


def show_returned_values(t, trend, trendlb, trendub, avt, avy, p, t1, t2, pvalue):
    st.subheader("Returned values")

    st.write(f"p: {p}")
    st.write(f"t1: {t1}")
    st.write(f"t2: {t2}")
    if pvalue != None:
        st.write(f"pvalue: {round(pvalue,4)}")
        if pvalue <0.05:
            st.info(f"The data indicates a long time change between {t1} and {t2}.")
        else:
            st.info(f"The data does not indicate (or a little) a long time change between {t1} and {t2}.")

    with st.expander("Arrays"):
        st.write(f"t: {t} - lengte:{len(t)}")
        st.write(f"trend: {trend} - lengte:{len(trend)}")
        st.write(f"trendubound: {trendub} - lengte:{len(trendub)}")
        st.write(f"trendlbound: {trendlb} - lengte:{len(trendlb)}")
        st.write(f"averaget: {avt} - lengte:{len(avt)}")
        st.write(f"averagey: {avy} - lengte:{len(avy)}")


def getdata():
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\de_bilt_jaargem_1901_2022.csv"
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/de_bilt_jaargem_1901_2022.csv"
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


def main():
    show_info()
    df = getdata()
    year_list = df["YYYY"].to_list()
    df = df.tail(-1)

    show_options = [
        "temp_min",
        "temp_avg",
        "temp_max",
    ]

    what_to_show = st.sidebar.selectbox("Wat weer te geven", show_options, 2)
    drawplot = st.sidebar.selectbox("Show Matplotlib plot", [True, False], 1)
    draw30 = st.sidebar.selectbox("Show 30 year SMA", [True, False], 1)
    test_trend= st.sidebar.selectbox("Two-sided test for absence of trend", [True, False], 1)
    if test_trend:
        t1 = st.sidebar.number_input("t1",1901,2050, 1950)
        t2 = st.sidebar.number_input("t1", 1901,2050,2000)
        if t2<(t1+30) or t1 not in year_list  or t2 not in year_list:
            st.error("t2 must be 30 years later than t1 and both years have to be in the dataframe")
            st.stop()
    else:
        t1,t2 = None, None

    X_array = df["YYYY"].values
    Y_array = df[what_to_show].values

    t, trend, trendlb, trendub, avt, avy, p, t1_, t2_, pvalue = climatrend(
        X_array,
        Y_array,
        p=None,
        t1=t1,
        t2=t2,
        ybounds=None,
        drawplot=drawplot,
        draw30=draw30,
    )

    show_plot_plotly(
        df,
        what_to_show,
        t,
        trend,
        trendlb,
        trendub,
        avt,
        avy,
        p,
        t1_,
        t2_,
        pvalue,
        draw30,
    )
    show_returned_values(t, trend, trendlb, trendub, avt, avy, p, t1_, t2_, pvalue)


if __name__ == "__main__":
    main()
