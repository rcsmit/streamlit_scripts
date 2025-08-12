import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
from scipy.stats import weibull_min, lognorm, kstest
import plotly.graph_objects as go
from schoonmaaktijden2025 import read
import warnings

# Negeer irrelevante warnings
warnings.simplefilter("ignore", category=Warning)


def to_scalar(x):
    """
    Converteer een numpy array of getal naar een float-scalar.

    Parameters
    ----------
    x : array-like of float
        Waarde die mogelijk een numpy array is.

    Returns
    -------
    float
        Scalar waarde.
    """
    return float(x.item()) if isinstance(x, np.ndarray) else float(x)


def plot_distribution_plotly(data, row):
    """
    Plot een histogram met een fitted distributie in Plotly.

    Parameters
    ----------
    data : array-like
        Observaties (bv. schoonmaaktijden in minuten).
    row : pandas.Series
        Bevat kolommen 'distribution', 'parameters', 'p_value' en 'fit_ok'.

    Returns
    -------
    tuple
        (fig, params) waarbij fig een Plotly Figure is en params de distributieparameters.
    """
    x = np.linspace(0, max(data), 1000)
    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        histnorm='probability density',
        nbinsx=40,
        name='Data',
        opacity=0.6
    ))

    dist_name = row["distribution"]
    params = row["parameters"]
    p_value = row["p_value"]
    fit_ok = row["fit_ok"]

    try:
        dist = getattr(stats, dist_name)
        y = dist.pdf(x, *params)
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='lines',
            name=f'{dist_name} fit'
        ))
        fig.update_layout(
            title=f"{dist_name} | p = {p_value:.3f} | fit ok: {fit_ok}",
            xaxis_title='Waarde',
            yaxis_title='Dichtheid',
            showlegend=True
        )
    except Exception as e:
        fig.update_layout(title=f"{dist_name} | Plotten mislukt: {e}")

    return fig, params


def plot_pdf_cdf(data):
    """
    Plot PDF and CDF of a fitted distribution on the real value axis.

    Parameters
    ----------
    data : array-like
        Observed data values (e.g., cleaning times in minutes).
    dist_name : str
        Name of the scipy.stats distribution (e.g., 'lognorm', 'weibull_min').
    params : tuple
        Parameters returned by scipy.stats.<dist>.fit(data).
    """
    
    dist_name = "lognorm"  # Example distribution name, can be parameterized
    data = np.array(data)
    params = lognorm.fit(data)
    x = np.linspace(min(data), max(data), 1000)
    dist = getattr(stats, dist_name)

    pdf = dist.pdf(x, *params)
    cdf = dist.cdf(x, *params)

    # Create figure with secondary y-axis
    fig = go.Figure()

    # Histogram as PDF approximation
    fig.add_trace(go.Histogram(
        x=data,
        histnorm='probability density',
        nbinsx=20,
        name='Histogram (PDF)',
        opacity=0.6,
        marker_color='lightblue',
        yaxis='y'
    ))

    # PDF line
    fig.add_trace(go.Scatter(
        x=x,
        y=pdf,
        mode='lines',
        name=f'{dist_name} PDF',
        line=dict(color='red'),
        yaxis='y'
    ))

    # CDF line on secondary axis
    fig.add_trace(go.Scatter(
        x=x,
        y=cdf,
        mode='lines',
        name=f'{dist_name} CDF',
        line=dict(color='green', dash='dash'),
        yaxis='y2'
    ))

    # Layout with 2 y-axes
    fig.update_layout(
        title=f"{dist_name} PDF & CDF",
        xaxis=dict(title='Value'),
        yaxis=dict(title='Probability Density (PDF)', side='left'),
        yaxis2=dict(
            title='Cumulative Probability (CDF)',
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        bargap=0.1,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    return 


def test_distributions(data, distributions):
    """
    Test een dataset tegen meerdere kansverdelingen met een KS-test.

    Parameters
    ----------
    data : array-like
        Observaties die getest worden.
    distributions : list
        Namen van scipy.stats distributies (str).

    Returns
    -------
    pandas.DataFrame
        Resultaten met kolommen: distribution, p_value, D_statistic, fit_ok, parameters
    """
    data = np.array(data)
    results = []

    for i, dist_name in enumerate(distributions):
        try:
            print(f"{i+1}/{len(distributions)} - {dist_name}")
            dist = getattr(stats, dist_name)
            params = dist.fit(data)
            D, p = stats.kstest(data, dist_name, args=params)
            fit_ok = p > 0.05
            results.append((dist_name, p, D, fit_ok, params))
        except Exception:
            results.append((dist_name, np.nan, np.nan, False, None))

    df_results = pd.DataFrame(results, columns=["distribution", "p_value", "D_statistic", "fit_ok", "parameters"])
    df_results = df_results.sort_values("p_value", ascending=False).reset_index(drop=True)
    st.write(df_results)
    return df_results


def various_distributions(data):
    """
    Test een reeks standaarddistributies en plot deze met hun parameters.

    Parameters
    ----------
    data : array-like
        Observaties die getest worden.
    """
    param_labels_map = {
        "norm": ["loc", "scale"],
        "expon": ["loc", "scale"],
        "gamma": ["a", "loc", "scale"],
        "lognorm": ["s", "loc", "scale"],
        "beta": ["a", "b", "loc", "scale"],
        "weibull_min": ["c", "loc", "scale"],
        "weibull_max": ["c", "loc", "scale"],
        "pareto": ["b", "loc", "scale"],
        "uniform": ["loc", "scale"],
        "triang": ["c", "loc", "scale"],
        "logistic": ["loc", "scale"],
        "t": ["df", "loc", "scale"],
        "f": ["dfn", "dfd", "loc", "scale"],
        "rayleigh": ["loc", "scale"],
    }

    #     "alpha": ["a", "loc", "scale"],
    #     "anglit": ["loc", "scale"],
    #     "arcsine": ["loc", "scale"],
    #     "burr": ["c", "d", "loc", "scale"],
    #     "burr12": ["c", "d", "loc", "scale"],
    #     "cauchy": ["loc", "scale"],
    #     "chi": ["df", "loc", "scale"],
    #     "chi2": ["df", "loc", "scale"],
    #     "cosine": ["loc", "scale"],
    #     "dgamma": ["a", "loc", "scale"],
    #     "dweibull": ["c", "loc", "scale"],
    #     "erlang": ["a", "loc", "scale"],
    #     "exponnorm": ["K", "loc", "scale"],
    #     "exponweib": ["a", "c", "loc", "scale"],
    #     "exponpow": ["b", "loc", "scale"],
    #     "fatiguelife": ["c", "loc", "scale"],
    #     "foldcauchy": ["c", "loc", "scale"],
    #     "foldnorm": ["c", "loc", "scale"],
    #     "genlogistic": ["c", "loc", "scale"],
    #     "genpareto": ["c", "loc", "scale"],
    #     "gumbel_r": ["loc", "scale"],
    #     "gumbel_l": ["loc", "scale"],
    #     "halfcauchy": ["loc", "scale"],
    #     "halfnorm": ["loc", "scale"],
    #     "hypsecant": ["loc", "scale"],
    #     "invgamma": ["a", "loc", "scale"],
    #     "invgauss": ["mu", "loc", "scale"],
    #     "invweibull": ["c", "loc", "scale"],
    #     "johnsonsb": ["a", "b", "loc", "scale"],
    #     "johnsonsu": ["a", "b", "loc", "scale"],
    #     "ksone": ["n"],
    #     "kstwobign": ["loc", "scale"],
    #     "laplace": ["loc", "scale"],
    #     "levy": ["loc", "scale"],
    #     "levy_l": ["loc", "scale"],
    #     "levy_stable": ["alpha", "beta", "loc", "scale"],
    #     "loggamma": ["c", "loc", "scale"],
    #     "loglaplace": ["c", "loc", "scale"],
    #     "lomax": ["c", "loc", "scale"],
    #     "maxwell": ["loc", "scale"],
    #     "mielke": ["k", "s", "loc", "scale"],
    #     "nakagami": ["nu", "loc", "scale"],
    #     "ncf": ["dfn", "dfd", "nc", "loc", "scale"],
    #     "nct": ["df", "nc", "loc", "scale"],
    #     "ncx2": ["df", "nc", "loc", "scale"],
    #     "pearson3": ["skew", "loc", "scale"],
    #     "powerlaw": ["a", "loc", "scale"],
    #     "powerlognorm": ["c", "s", "loc", "scale"],
    #     "powernorm": ["c", "loc", "scale"],
    #     "reciprocal": ["a", "b"],
    #     "rice": ["b", "loc", "scale"],
    #     "semicircular": ["loc", "scale"],
    #     "skewnorm": ["a", "loc", "scale"],
    #     "tukeylambda": ["lam", "loc", "scale"],
    #     "vonmises": ["kappa", "loc", "scale"],
    #     "wald": ["loc", "scale"],
    #     "wrapcauchy": ["c", "loc", "scale"]
    # }

    df_results = test_distributions(data, list(param_labels_map.keys()))

    for _, row in df_results.iterrows():
        fig, params = plot_distribution_plotly(data, row)
        st.plotly_chart(fig, use_container_width=True)
        if row["parameters"]:
            param_labels = param_labels_map.get(row["distribution"], [f"p{i}" for i in range(len(row["parameters"]))])
            params_str = ", ".join(f"{label} = {val:.3f}" for label, val in zip(param_labels, row["parameters"]))
            st.markdown(f"**Parameters**: {params_str}")
        else:
            st.write("No parameters")


def weibull_vs_lognorm(data):
    """
    Vergelijk Weibull en Lognormaal fit op een dataset en plot de verdelingen.

    Parameters
    ----------
    data : array-like
        Observaties (bv. schoonmaaktijden in minuten).
    """
    k, loc_w, lambda_ = weibull_min.fit(data)
    x = np.linspace(0, max(data), 1000)
    weibull_pdf = weibull_min.pdf(x, k, loc_w, lambda_)

    shape, loc_l, scale = lognorm.fit(data)
    lognorm_pdf = lognorm.pdf(x, shape, loc_l, scale)

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=40,
        histnorm='probability density',
        name='Data histogram',
        opacity=0.6,
    ))

    # Weibull
    fig.add_trace(go.Scatter(
        x=x,
        y=weibull_pdf,
        mode='lines',
        name=f'Weibull fit (k={k:.2f}, λ={lambda_:.2f})',
        line=dict(color='red')
    ))

    # Lognormaal
    fig.add_trace(go.Scatter(
        x=x,
        y=lognorm_pdf,
        mode='lines',
        name=f'Lognormal fit (shape={shape:.2f}, scale={scale:.2f})',
        line=dict(color='green')
    ))

    fig.update_layout(
        title='Cleaning Times: Histogram met Weibull vs Lognormal Fit',
        xaxis_title='Cleaning time (minutes)',
        yaxis_title='Probability Density',
        bargap=0.1
    )

    st.plotly_chart(fig, use_container_width=True)


def lognormaal(data):
    """
    Analyseer of data lognormaal verdeeld is met een Q-Q plot en een fit.

    Parameters
    ----------
    data : array-like
        Observaties (bv. schoonmaaktijden in minuten).
    """
    log_data = np.log(data)

    # Q-Q Plot
    (theoretical, observed), _ = stats.probplot(log_data, dist="norm")
    fig_qq = go.Figure()
    fig_qq.add_trace(go.Scatter(x=theoretical, y=observed, mode='markers', name='Log-data Q-Q'))

    min_val = min(theoretical.min(), observed.min())
    max_val = max(theoretical.max(), observed.max())
    fig_qq.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                mode='lines', name='y=x', line=dict(color='red', dash='dash')))
    fig_qq.update_layout(title='Log-transformed Q-Q Plot', xaxis_title='Theoretical Quantiles', yaxis_title='Observed Quantiles')
    st.plotly_chart(fig_qq, use_container_width=True)

    # Lognormale fit
    shape, loc, scale = stats.lognorm.fit(data)
    x = np.linspace(0, max(data), 1000)
    pdf = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=data, nbinsx=40, histnorm='probability density', name='Data histogram', opacity=0.6))
    fig_hist.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Lognormal fit', line=dict(color='red')))
    fig_hist.update_layout(title='Lognormal Fit on Cleaning Times', xaxis_title='Cleaning Time (minutes)', yaxis_title='Probability Density', bargap=0.1)
    st.plotly_chart(fig_hist, use_container_width=True)


def compare_lognormal_fits(data):
    """
    Vergelijk lognormale fit met en zonder verschoven locatieparameter.

    Parameters
    ----------
    data : array-like
        Observaties (bv. schoonmaaktijden in minuten).
    """
    data = np.array(data)
    x = np.linspace(min(data), max(data), 1000)

    # Fit 1: loc=0
    shape0, loc0, scale0 = lognorm.fit(data, floc=0)
    pdf0 = lognorm.pdf(x, shape0, loc0, scale0)
    _, p0 = kstest(data, "lognorm", args=(shape0, loc0, scale0))

    # Fit 2: vrije loc
    shape1, loc1, scale1 = lognorm.fit(data)
    pdf1 = lognorm.pdf(x, shape1, loc1, scale1)
    _, p1 = kstest(data, "lognorm", args=(shape1, loc1, scale1))

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=40, histnorm='probability density', name='Data', opacity=0.6))
    fig.add_trace(go.Scatter(x=x, y=pdf0, mode='lines', name=f'Lognorm (loc=0), p={p0:.3f}', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=x, y=pdf1, mode='lines', name=f'Lognorm (loc={loc1:.2f}), p={p1:.3f}', line=dict(color='green')))
    fig.update_layout(title='Lognormal Fit: Classic vs Shifted', xaxis_title='Cleaning time (minutes)', yaxis_title='Probability Density', bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Voer alle distributie-analyses en visualisaties uit."""
    st.title("Test op verschillende distributies")
    df = read()
    data = df["tijd in minuten"].tolist()

    weibull_vs_lognorm(data)
    lognormaal(data)
    plot_pdf_cdf(data)
    compare_lognormal_fits(data)
    various_distributions(data)


if __name__ == "__main__":
    main()




