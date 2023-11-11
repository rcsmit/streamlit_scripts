import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from skmisc.loess import loess

def show_plot_plotly(X_array,Y_array,lowess,ll,ul):
    """Draw a plot with the results with plotly

    Args:
        title (string): title of the graph
        t (int): years
        trend (float): the values of the trendline
        ll (float): the values of the trendline 95% lower bound
        trendub (float): the values of the trendline 95% upper bound
    """

    loess_line = go.Scatter(
        name=f"Loess",
        x=X_array,
        y=lowess,
        mode="lines",
        line=dict(width=1, color="rgba(255, 0, 255, 1)"),
    )
    
    # Create a filled area plot for confidence interval
    confidence_trace = go.Scatter(x=np.concatenate([X_array, X_array[::-1]]),
                            y=np.concatenate([ul, ll[::-1]]),
                                fill='tozeroy',
                                fillcolor='rgba(0, 128, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                showlegend=True,
                                name="CI of the trendline")

    values = go.Scatter(
        name="values",
        x=X_array,
        y=Y_array,
        mode="lines",
        line=dict(width=1, color="rgba(0, 0, 255, 0.6)"),
    )

    data = [values, loess_line]
   
    data.append(confidence_trace)
   
    # Find the bounds for the graph. Have to be calculated since the there is a fill between the CI-limits
    # using (fill='tozeroy')
    A_1d = np.ravel(Y_array)
    B_1d = np.ravel(ul)
    C_1d = np.ravel(ll)
   
    try:
        Y_values = A_1d.tolist() + B_1d.tolist() + C_1d.tolist()
        y_lower_bound = min(Y_values)    
        y_upper_bound = max(Y_values)
    except:
        y_lower_bound = Y_array.min()    
        y_upper_bound = Y_array.max()
    
    layout = go.Layout(
        yaxis=dict(title="year averages", range=[y_lower_bound, y_upper_bound ]), title=f"Year averages of ... - {title}"
    )
 
    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(xaxis=dict(tickformat="%d-%m-%Y"))
    st.plotly_chart(fig, use_container_width=True)
    
    
def getdata():
    """Get the data

    Returns:
        df: df with the data
    """    
    excel_file_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gasstanden95xxCN5.xlsx" # r"C:\Users\rcxsm\Documents\xls\gasstanden95xxCN5.xlsx"
    df = pd.read_excel(excel_file_path)
    df["datum"] = pd.to_datetime(df["datum"].astype(str),  format='%Y-%m-%d')
    return df


def main_skmisc(x, y, span_=42, it=1):
    """Make a plot and calculate p-value with scikit-misc. Stripped down version

    Args:
        x : list of Years, increasing by 1.
        y : list of Annual values
        span_ : Numerator of the Smoothing factor (denominator will be y),
        it : number of iterations
     
    Returns:
        lowess: the values of the lowess curve
        ll : lower bound
        ul : upper bound

    """

    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess.html
    # https://stackoverflow.com/questions/31104565/confidence-interval-for-lowess-in-python

    ybounds = [-np.inf, np.inf]
    ybounds = sorted(ybounds)

    # Dimensions and checks
    y = np.asarray(y, dtype=np.float64)
    span =span_/len(y)
    l = loess(x,y)
    
    # MODEL and CONTROL. Essential for replicating the results from the R script of KNMI.
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_model.html#skmisc.loess.loess_model
    # https://has2k1.github.io/scikit-misc/stable/generated/skmisc.loess.loess_control.html#skmisc.loess.loess_control
   
    l.model.span = span
    l.model.degree = 1              # Overall degree of locally-fitted polynomial. 1 is locally-linear fitting 
                                    # and 2 is locally-quadratic fitting. Degree should be 2 at most.
                                    #  Default is 2.
    l.control.iterations = it       # must be 1 for replicating the R-script
    l.control.surface = "direct"    # One of [‘interpolate’, ‘direct’] Determines whether the fitted surface 
                                    # is computed directly at all points (‘direct’) or whether an
                                    # interpolation method is used (‘interpolate’). The default
                                    # ‘interpolate’) is what most users should use unless special 
                                    # circumstances warrant.
    l.control.statistics = "exact"  # One of [‘approximate’, ‘exact’] Determines whether the statistical 
                                    # quantities are computed exactly (‘exact’) or approximately
                                    # (‘approximate’). ‘exact’ should only be used for testing
                                    # the approximation in statistical development and is not
                                    # meant for routine usage because computation time can be 
                                    # horrendous.

    l.fit()
    pred = l.predict(x, stderror=True)
    conf = pred.confidence()
    lowess = pred.values
    ll = conf.lower
    ul = conf.upper

    return lowess,ll,ul


def main():
    df = getdata()
    what_to_show = "verbruik"
    X_array = df["datum"].values
    Y_array = df[what_to_show].values
    lowess,ll,ul = main_skmisc(X_array, Y_array)
    show_plot_plotly(X_array,Y_array,lowess,ll,ul)
    return 

if __name__ == "__main__":
    main()