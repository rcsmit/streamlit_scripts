"""
Gasverbruik analyse - vergelijkt gasverbruik met weerdata (KNMI station De Bilt/Nieuw Beerta).
Visualiseert verbruik vs. temperatuur, berekent regressielijnen voor twee periodes
(voor/na een splitsdatum, bijv. verhuizing of isolatie-ingreep).
"""

# version : 20260602-120000 - Initial typed and documented version
current_version = "20260602-120000"

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import statsmodels.api as sm
import streamlit as st
import time
from scipy.stats import linregress
from skmisc.loess import loess  # noqa: F401  (imported for future use)
from sklearn.linear_model import LinearRegression

from show_knmi_functions.wbgt_knmi import wbgt_buiten_oud
# ---------------------------------------------------------------------------
# Sidebar / interface helpers
# ---------------------------------------------------------------------------

def interface() -> tuple[str, int, float]:
    """Render sidebar controls and return the selected options.

    Returns:
        what: Name of the weather variable to analyse (e.g. 'temp_avg').
        window_size: Rolling-average window in weeks.
        afkapgrens_scatter: Upper temperature cut-off for scatter plots.
    """
    what: str = st.sidebar.selectbox(
        "What to show",
        [
            "temp_min", "temp_avg", "temp_max", "graad_dagen","wbgt_buiten",
            "T10N", "zonneschijnduur", "perc_max_zonneschijnduur",
            "glob_straling", "neerslag_duur", "neerslag_etmaalsom",
            "RH_min", "RH_max",
        ],
        index=1,
    )
    window_size: int = st.sidebar.number_input("Window size", 1, 100, 3)

  
    return what, window_size


def interface_afkapgrenzen(beste_voor,beste_na) -> tuple[str, int, float]:
    """Render sidebar controls and return the selected options.

    Returns:
        
        afkapgrens_scatter_voor: Upper temperature cut-off for scatter plots <=2024.
        afkapgrens_scatter_na: Upper temperature cut-off for scatter plots >=2025.
        
    """
 

    afkapgrens_scatter_voor: float = st.sidebar.number_input(
        "Afkapgrens scatter voor", -10.0, 999.0, beste_voor
    )
    afkapgrens_scatter_na: float = st.sidebar.number_input(
        "Afkapgrens scatter na", 1.0, 999.0, beste_na
    )
    return afkapgrens_scatter_voor,  afkapgrens_scatter_na 



def interface_multiple_linear_regression() -> tuple[str, list[str]]:
    """Render sidebar controls for multiple linear regression.

    Returns:
        y_value: Name of the dependent variable column.
        x_values: List of independent variable column names.
    """
    y_value: str = st.selectbox("Y value", ["verbruik"], index=0)
    x_options: list[str] = [
        "temp_avg", "temp_min", "temp_max", "graad_dagen", "T10N",
        "zonneschijnduur", "perc_max_zonneschijnduur", "glob_straling",
        "neerslag_duur", "neerslag_etmaalsom", "RH_min", "RH_max",
    ]
    x_default: list[str] = ["temp_min", "zonneschijnduur", "perc_max_zonneschijnduur"]
    x_values: list[str] = st.multiselect("X values", x_options, x_default)
    return y_value, x_values


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data()
def get_verbruiks_data() -> pd.DataFrame:
    """Load weekly gas-consumption data from Google Sheets (or GitHub fallback).

    The Google Sheets URL points to the public 'verbruik' sheet of the workbook
    ``1j9V-otA53UWaI7-pDS4owU_qtZtjgMK8K5DLaN9kgqk``.

    Returns:
        DataFrame with columns: datum (datetime), verbruik (float),
        week_number (int), year_number (int).

    Raises:
        SystemExit: Via ``st.stop()`` when the data source is unavailable.
    """
    google_sheets: bool = True
    if google_sheets:
        sheet_id = "1j9V-otA53UWaI7-pDS4owU_qtZtjgMK8K5DLaN9kgqk"
        sheet_name = "verbruik"
        url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
        )
        try:
            df = pd.read_csv(url, delimiter=",")
            df["datum"] = pd.to_datetime(df["datum"].astype(str), format="%d/%m/%Y")
        except Exception as exc:
            st.error(f"Error reading verbruik: {exc}")
            st.stop()
    else:
        excel_url = (
            "https://raw.githubusercontent.com/rcsmit/streamlit_scripts"
            "/main/input/gasstanden95xxCN5.xlsx"
        )
        df = pd.read_excel(excel_url)
        df["datum"] = pd.to_datetime(df["datum"].astype(str), format="%Y-%m-%d")

    df["week_number"] = df["datum"].dt.isocalendar().week
    df["year_number"] = df["datum"].dt.isocalendar().year
    return df

@st.cache_data()
def get_weather_info(what: str) -> pd.DataFrame:
    """Fetch and aggregate daily KNMI weather data to weekly totals/averages.

    Data is fetched from the KNMI daggegevens API (station 260 – De Bilt).
    Daily values for temperature columns are divided by 10 (KNMI stores them
    as tenths of a degree / mm / etc.).  Graad-dagen are *summed* per week;
    all other variables are *averaged*.

    Args:
        what: The primary weather variable that will be analysed downstream.
              Used to decide whether to plot graad_dagen raw values.

    Returns:
        DataFrame aggregated by (week_number, year_number), containing all
        weather columns plus ``graad_dagen``.
    """
    today: str = datetime.now().strftime("%Y%m%d")
    url = (
        f"https://www.daggegevens.knmi.nl/klimatologie/daggegevens"
        f"?stns=260&vars=TEMP:SQ:SP:Q:DR:RH:UN:UX:FG&start=20190202&end={today}"
    )

    df = pd.read_csv(url, delimiter=",", header=None, comment="#", low_memory=False)

    column_map: dict[int, str] = {
        0: "STN", 1: "YYYYMMDD",
        2: "temp_avg", 3: "temp_min", 4: "temp_max", 5: "T10N",
        6: "zonneschijnduur", 7: "perc_max_zonneschijnduur", 8: "glob_straling",
        9: "neerslag_duur", 10: "neerslag_etmaalsom", 11: "RH_min", 12: "RH_max",13:"wind"
    }
    df = df.rename(columns=column_map)

    to_divide_by_10: list[str] = [
        "temp_avg", "temp_min", "temp_max",
        "zonneschijnduur", "neerslag_duur", "neerslag_etmaalsom","wind"
    ]
    for col in to_divide_by_10:
        df[col] = df[col] / 10

    df["wbgt_buiten"] = df.apply(
        lambda r: wbgt_buiten_oud(r["temp_avg"], (r["RH_max"] +r["RH_min"])/2, r["wind"],r["glob_straling"]* 10000 / 3600),
        axis=1,
    ).round(1)

    
    df["YYYYMMDD"] = pd.to_datetime(df["YYYYMMDD"].astype(str))
    df["week_number"] = df["YYYYMMDD"].dt.isocalendar().week
    df["year_number"] = df["YYYYMMDD"].dt.isocalendar().year

    
    # Quick raw-data chart
    fig = px.scatter(df, x="YYYYMMDD", y=what, hover_data=["YYYYMMDD", what])
    st.plotly_chart(fig)

    agg_dict: dict[str, str] = {
        "temp_avg": "mean", "temp_min": "mean", "temp_max": "mean","wbgt_buiten": "mean",
        "graad_dagen": "sum", "T10N": "mean",
        "zonneschijnduur": "mean", "perc_max_zonneschijnduur": "mean",
        "glob_straling": "mean", "neerslag_duur": "mean",
        "neerslag_etmaalsom": "mean", "RH_min": "mean", "RH_max": "mean",
    }

    df = calculate_graad_dagen(df, what)

    if what == "graad_dagen":
        result = (
            df.groupby(["week_number", "year_number"], observed=False)["graad_dagen"]
            .sum()
            .reset_index()
        )
    else:
        result = (
            df.groupby(["week_number", "year_number"], observed=False)
            .agg(agg_dict)
            .reset_index()
        )

    return result


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def calculate_graad_dagen(df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Add a ``graad_dagen`` column to *df* using the Dutch heating-degree-day method.

    Formula (source: https://www.olino.org/blog/nl/articles/2009/12/14/het-rekenen-met-graaddagen/):
        - Base temperature: 18 °C
        - Season factor: 0.8 for April–September, 1.0 for March/October, 1.1 for other months
        - Negative values (i.e. temperatures above 18 °C) are clipped to 0

    Args:
        df: DataFrame containing a ``YYYYMMDD`` datetime column and a
            temperature column matching *what* (or ``temp_avg`` when
            *what* == ``"graad_dagen"``).
        what: Name of the temperature column to use. When equal to
              ``"graad_dagen"`` the function falls back to ``temp_avg``.

    Returns:
        The same DataFrame with a new (or overwritten) ``graad_dagen`` column.
    """
    temp_col = "temp_avg" if what == "graad_dagen" else what

    season_factors: dict[int, float] = {
        m: f
        for months, f in [
            (range(4, 10), 0.8),
            ([3, 10], 1.0),
        ]
        for m in months
    }

    def _row_factor(row: pd.Series) -> float:
        return season_factors.get(row["YYYYMMDD"].month, 1.1)

    df["graad_dagen"] = (18 - df[temp_col]).clip(lower=0)
    df["graad_dagen"] = df["graad_dagen"] * df.apply(_row_factor, axis=1)
    return df


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_dataframes(
    df: pd.DataFrame,
    what: str,
    window_size: int,
    df_nw_beerta: pd.DataFrame,
) -> pd.DataFrame:
    """Merge consumption data with weekly weather data and add rolling averages.

    Args:
        df: Weekly consumption DataFrame (must have ``year_number``, ``week_number``).
        what: Weather variable column name.
        window_size: Number of weeks for the rolling average.
        df_nw_beerta: Weekly weather DataFrame from :func:`get_weather_info`.

    Returns:
        Merged DataFrame with extra columns ``year_week``, ``verbruik_sma``,
        and ``{what}_sma``.
    """
    merged = df.merge(df_nw_beerta, on=["year_number", "week_number"], how="outer")
    merged["year_week"] = (
        merged["year_number"].astype(str) + "_" + merged["week_number"].astype(str)
    )
    merged["verbruik_sma"] = merged["verbruik"].rolling(window=window_size).mean()
    merged[f"{what}_sma"] = merged[what].rolling(window=window_size).mean()
    return merged


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def make_scatter(x: str, y: str, df: pd.DataFrame) -> None:
    """Render scatter plots with OLS trendlines and confidence-interval bands.

    Two sub-sections are shown:
    1. A simple scatter with per-year colour coding and OLS trendlines.
    2. A scatter with binned confidence intervals (95 %) overlaid as a shaded band.

    Args:
        x: Column name for the horizontal axis (weather variable).
        y: Column name for the vertical axis (``"verbruik"``).
        df: Merged DataFrame containing at least *x*, *y*, and ``year_week``.
    """
    st.subheader("Met trendlijnen")
    fig = px.scatter(
        df, x=x, y=y,
        hover_data=["year_week"],
        color="year_number",
        trendline="ols",
    )
    st.plotly_chart(fig)

    correlation: float = df[x].corr(df[y])
    slope, intercept, r_value, p_value, std_err = linregress(df[x], df[y])
    st.write(f"Correlation: {correlation:.2f}")
    st.write(f"Equation of the line: y = {slope:.2f} * x + {intercept:.2f}")

    # --- Confidence-interval plot ---
    st.subheader("Met betrouwbaarheidsintervallen")
    st.write(df)

    bin_width: float = 0.5
    df = df.copy()
    df["temp_bin"] = pd.cut(
        df["temp_min"],
        bins=np.arange(df["temp_min"].min(), df["temp_min"].max() + bin_width, bin_width),
    )

    grouped = (
        df.groupby("temp_bin", observed=False)
        .agg(mean=("verbruik", "mean"), std=("verbruik", "std"), count=("verbruik", "count"))
        .reset_index()
    )

    confidence_level: float = 0.95
    grouped["margin_error"] = (
        grouped["std"]
        / np.sqrt(grouped["count"])
        * stats.t.ppf((1 + confidence_level) / 2, grouped["count"] - 1)
    )
    grouped["lower_ci"] = grouped["mean"] - grouped["margin_error"]
    grouped["upper_ci"] = grouped["mean"] + grouped["margin_error"]
    grouped = grouped.dropna(subset=["mean", "std"]).reset_index(drop=True)
    grouped["average_temp"] = grouped["temp_bin"].apply(lambda b: (b.left + b.right) / 2)

    df_ = df.merge(grouped, left_on="temp_min", right_on="mean", how="outer")
    for col in ["mean", "lower_ci", "upper_ci"]:
        df_[f"{col}_sma"] = df_[col].rolling(window=3).mean()

    fig = px.scatter(df_, x="temp_min", y="verbruik", hover_data=["year_week"], color="year_number")
    fig.add_trace(go.Scatter(x=df_["average_temp"], y=df_["mean_sma"]))
    fig.add_trace(
        go.Scatter(
            x=df_["average_temp"], y=df_["lower_ci_sma"],
            mode="lines", fill="tonexty",
            fillcolor="rgba(120, 128, 0, 0.0)",
            line=dict(width=0.7, color="rgba(0, 0, 255, 0.5)"),
            name="lower CI",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_["average_temp"], y=df_["upper_ci_sma"],
            fill="tonexty", fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(color="dimgrey", width=0.5),
            name="upper ci",
        )
    )
    fig.update_xaxes(title_text="Temperature (°C)")
    fig.update_yaxes(title_text="Verbruik")
    st.plotly_chart(fig, key=str(time.time()))


def make_plots(what: str, afkapgrens_scatter_voor: float,afkapgrens_scatter_na: float, merged_df: pd.DataFrame) -> None:
    """Render the full suite of time-series and pivot charts.

    Charts produced:
    - Scatter of raw verbruik and weather variable over time.
    - Line chart of smoothed verbruik and weather variable.
    - Pivot line chart: verbruik per week number, coloured by year.
    - Pivot line chart: weather variable per week number, coloured by year.
    - Scatter plot of weather vs. verbruik (filtered by *afkapgrens_scatter*).
    - Same scatter for rolling-average columns.

    Args:
        what: Weather variable column name to plot alongside verbruik.
        afkapgrens_scatter: Exclude rows where *what* >= this value from scatter.
        merged_df: Merged DataFrame from :func:`merge_dataframes`.
    """
    fig = px.scatter(merged_df, x="year_week", y=[what, "verbruik"],
                     title=f"verbruik en {what} in de tijd")
    st.plotly_chart(fig)

    fig = px.line(merged_df, x="year_week", y=[f"{what}_sma", "verbruik_sma"],
                  title=f"gladgestreken verbruik en {what} in de tijd")
    st.plotly_chart(fig)

    df_pivot = merged_df.pivot(index="week_number", columns="year_number", values="verbruik")
    fig = px.line(df_pivot, labels={"week_number": "Week Number", "value": "Verbruik"},
                  title="verbruik in verschillende jaren")
    st.plotly_chart(fig)

    df_pivot = merged_df.pivot(index="week_number", columns="year_number", values=what)
    fig = px.line(df_pivot, labels={"week_number": "Week Number", "value": what},
                  title=f"{what} in verschillende jaren")
    st.plotly_chart(fig)

    merged_filtered = merged_df[
        ((merged_df["year_number"] <= 2024) & (merged_df[what] < afkapgrens_scatter_voor)) |
        ((merged_df["year_number"] >= 2025) & (merged_df[what] < afkapgrens_scatter_na))
    ]
    make_scatter(what, "verbruik", merged_filtered)

    st.subheader("Lopend gemiddelde")
    make_scatter(f"{what}_sma", "verbruik_sma", merged_filtered)


# ---------------------------------------------------------------------------
# Regression analysis
# ---------------------------------------------------------------------------
def find_best_drempeltemp(
    df: pd.DataFrame,
    what: str,
    kandidaten: list[float] | None = None,
) -> tuple[float, float]:
    """Zoek de drempeltemperatuur met de hoogste R² per periode.

    De drempeltemperatuur is de temperatuur waarbij het gasverbruik nul wordt;
    de regressielijn wordt geforceerd door het punt (drempeltemp, 0).

    Args:
        df: Merged DataFrame met kolommen ``year_number``, *what* en ``verbruik``.
        what: Weer-variabele die als x-as dient.
        kandidaten: Te testen drempeltemperaturen. Standaard 0.5-stappen van 5 t/m 25.

    Returns:
        Tuple (beste_drempel_voor, beste_drempel_na).
    """
    if kandidaten is None:
        kandidaten = list(np.arange(5, 25.5, 0.5))

    def _r2(sub: pd.DataFrame, drempel: float) -> float:
        filtered = sub[[what, "verbruik"]].dropna()
        filtered = filtered[filtered[what] < drempel]
        if len(filtered) < 3:
            return -np.inf
        X = (filtered[what].to_numpy() - drempel).reshape(-1, 1)
        y = filtered["verbruik"].to_numpy()
        model = LinearRegression(fit_intercept=False).fit(X, y)
        return float(model.score(X, y))

    df_voor = df[df["year_number"] <= 2024]
    df_na   = df[df["year_number"] >= 2025]

    beste_voor = max(kandidaten, key=lambda d: _r2(df_voor, d))
    beste_na   = max(kandidaten, key=lambda d: _r2(df_na,   d))

    return beste_voor, beste_na

def _fit_forced_slope(
    df: pd.DataFrame, what: str, drempeltemp: float
) -> float:
    """Fit a regression line forced through (drempeltemp, 0).

    Only rows where *what* < *drempeltemp* are used (heating season only).

    Args:
        df: DataFrame with columns *what* and ``verbruik``.
        drempeltemp: The temperature at which consumption is forced to zero.

    Returns:
        Fitted slope coefficient.
    """
    sub = df[[what, "verbruik"]].dropna()
    sub = sub[sub[what] < drempeltemp]
    X = (sub[what].to_numpy() - drempeltemp).reshape(-1, 1)
    y = sub["verbruik"].to_numpy()
    model = LinearRegression(fit_intercept=False).fit(X, y)
    return float(model.coef_[0])


def plot_all(
    df: pd.DataFrame,
    what: str,
    drempeltemp_voor: float,
    drempeltemp_na: float,
    split_date: str = "2024-12-14",
) -> None:
    """Compare gas consumption before and after a split date via forced-origin regression.

    Produces:
    - A combined scatter with separate coloured series and trendlines for each period.
    - Two individual regression plots (voor / na) side by side.
    - A summary of expected consumption at 0 °C and the percentage change.

    Args:
        df: Merged DataFrame containing ``datum``, ``verbruik``, and *what*.
        what: Weather variable to use as the independent variable.
        drempeltemp: Temperature above which consumption is assumed to be zero;
                     regression lines are forced through (drempeltemp, 0).
        split_date: ISO date string (YYYY-MM-DD) separating the two periods.
    """
    df = df.copy()
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    split = pd.to_datetime(split_date)
    df["groep"] = np.where(df["datum"] < split, "voor", "na")

    def _plot_combined_scatter(df: pd.DataFrame, drempeltemp_voor: float,drempeltemp_na: float, what: str) -> None:
        """Render combined scatter with per-groep trendlines."""
        fig = go.Figure()
        palette: dict[str, str] = {"voor": "lightblue", "na": "purple"}

        for groep, color in palette.items():
            d = df[df["groep"] == groep]
            fig.add_scatter(
                x=d[what], y=d["verbruik"],
                mode="markers", name=groep,
                marker=dict(color=color),
                customdata=np.stack(
                    [d["datum"].astype("object"), d[what], d["verbruik"]], axis=-1
                ),
                hovertemplate="datum: %{customdata[0]}<br>verbruik: %{customdata[2]}<extra></extra>",
            )

        for groep, color, drempeltemp in [("voor", "red", drempeltemp_voor), ("na", "darkblue", drempeltemp_na)]:
            slope = _fit_forced_slope(df[df["groep"] == groep], what, drempeltemp)
            d = df[df["groep"] == groep][[what, "verbruik"]].dropna()
            d = d[d[what] < drempeltemp]
            xs = np.linspace(d[what].min(), d[what].max(), 200)
            ys = slope * (xs - drempeltemp)
            fig.add_scatter(x=xs, y=ys, mode="lines", name=f"{groep} trend",
                            line=dict(color=color))

        fig.update_layout(xaxis_title=what, yaxis_title="verbruik")
        st.plotly_chart(fig, use_container_width=True)

        slope_voor = _fit_forced_slope(df[df["groep"] == "voor"], what, drempeltemp_voor)
        slope_na   = _fit_forced_slope(df[df["groep"] == "na"],   what, drempeltemp_na)
        intercept_voor = -drempeltemp_voor * slope_voor
        intercept_na   = -drempeltemp_na * slope_na

        st.write(f"voor:  verbruik = {slope_voor:.3f} * temperatuur + {intercept_voor:.3f}")
        st.write(f"na:    verbruik = {slope_na:.3f} * temperatuur + {intercept_na:.3f}")

        exp_voor = slope_voor * (0 - drempeltemp_voor)
        exp_na   = slope_na   * (0 - drempeltemp_na)
        bespaar_pct = 100 * (exp_voor - exp_na) / exp_voor if exp_voor != 0 else float("nan")

        st.write(f"verwacht verbruik 0 °C (voor): **{exp_voor:.1f}** m³")
        st.write(f"verwacht verbruik 0 °C (na):   **{exp_na:.1f}** m³")
        st.write(f"besparing: **{bespaar_pct:.1f}** %")

    def _plot_regression(df: pd.DataFrame, drempeltemp: float, groep: str) -> float:
        """Render individual regression plot for one period.

        Args:
            df: Full merged DataFrame (will be filtered to *groep*).
            drempeltemp: Forced-origin temperature.
            groep: ``"voor"`` or ``"na"``.

        Returns:
            Fitted slope coefficient.
        """
        sub = df[df["groep"] == groep].dropna(subset=[what, "verbruik"])
        sub = sub[sub[what] < drempeltemp]

        X_shift = (sub[[what]].values - drempeltemp)
        y = sub["verbruik"].values
        model = LinearRegression(fit_intercept=False).fit(X_shift, y)
        slope = float(model.coef_[0])
        r2 = float(model.score(X_shift, y))

        x_line = np.linspace(sub[what].min(), sub[what].max(), 200)
        y_line = slope * (x_line - drempeltemp)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub[what], y=sub["verbruik"], mode="markers", name=groep))
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"fit {groep}"))
        fig.update_layout(xaxis_title=what, yaxis_title="verbruik")
        st.plotly_chart(fig, use_container_width=True)
        st.write("slope", slope)
        st.write("r²", r2)
        return slope

    _plot_combined_scatter(df, drempeltemp_voor,drempeltemp_na, what)
    col1, col2 = st.columns(2)
    with col1:
        slope_voor = _plot_regression(df, drempeltemp_voor, "voor")
    with col2:
        slope_na = _plot_regression(df, drempeltemp_na, "na")

    if slope_voor > slope_na:
        st.info("In het nieuwe huis wordt minder gas verbruikt")
    else:
        st.success("In het nieuwe huis wordt meer gas verbruikt")
    return slope_voor, slope_na

def bereken_gemiddeld(df: pd.DataFrame, what: str) -> None:
    """Show mean weekly consumption and an OLS scatter per year.

    Args:
        df: Merged DataFrame with ``datum``, ``verbruik``, and *what*.
        what: Weather variable to include in the per-year scatter.
    """
    df = df.copy()
    df = df[["datum", "verbruik", what]]
    df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
    df["jaar"] = df["datum"].dt.year
    df["week"] = df["datum"].dt.isocalendar().week

    out = df.groupby(["jaar"], observed=False).mean().reset_index()
    out["per_jaar"] = out["verbruik"] * 52
    st.write(out)

    fig = px.scatter(
        out, x=what, y="verbruik",
        hover_data=["jaar"], color="jaar", trendline="ols",
    )
    st.plotly_chart(fig)


def multiple_lineair_regression(
    df_: pd.DataFrame, x_values: list[str], y_value: str
) -> None:
    """Fit and display an OLS multiple linear regression model.

    Rows with NaN in any of the selected columns are dropped before fitting.

    Interpretation notes shown in the Streamlit output:
    - |t| > 2  → coefficient likely significant
    - p < 0.05 → coefficient likely significant
    - F-statistic p < 0.05 → model has at least one significant predictor

    Args:
        df_: Merged DataFrame.
        x_values: Independent variable column names.
        y_value: Dependent variable column name.
    """
    df = df_.dropna(subset=x_values + [y_value])
    x = sm.add_constant(df[x_values])
    y = df[y_value]
    model = sm.OLS(y, x).fit()
    st.subheader("Multiple Lineair Regression")
    st.write("**OUTPUT ORDINARY LEAST SQUARES**")
    st.write(model.summary())


def multiple_lin_regr(merged_df: pd.DataFrame) -> None:
    """Wrapper: render interface controls and run multiple linear regression.

    Args:
        merged_df: Merged DataFrame passed through to
                   :func:`multiple_lineair_regression`.
    """
    y_value, x_values = interface_multiple_linear_regression()
    multiple_lineair_regression(merged_df, x_values, y_value)

def bereken_verschil(df, what, slope_voor, afkapgrens_scatter_voor):
    df_na = df[df["year_number"] >= 2025].copy()
    df_na["virtueel_voor"] = slope_voor * (df_na[what] - afkapgrens_scatter_voor)
    df_na["verschil"] = df_na["verbruik"] - df_na["virtueel_voor"]
    df_na["verschil_pct"] = 100 * df_na["verschil"] / df_na["virtueel_voor"]

    totaal_werkelijk = df_na["verbruik"].sum()
    totaal_virtueel  = df_na["virtueel_voor"].sum()
    totaal_verschil  = totaal_werkelijk - totaal_virtueel
    totaal_pct       = 100 * totaal_verschil / totaal_virtueel

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Werkelijk verbruik (m³)",  f"{totaal_werkelijk:.0f}")
    col2.metric("Virtueel 'oud huis' (m³)", f"{totaal_virtueel:.0f}")
    col3.metric("Verschil (m³)",            f"{totaal_verschil:.0f}")
    col4.metric("Besparing",                f"{-totaal_pct:.1f}%")

    st.dataframe(
        df_na[["datum", what, "verbruik", "virtueel_voor", "verschil", "verschil_pct"]]
        .rename(columns={"verschil_pct": "verschil_%"})
        .style.format({"verbruik": "{:.1f}", "virtueel_voor": "{:.1f}",
                    "verschil": "{:.1f}", "verschil_%": "{:.1f}"})
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the Streamlit app: load data, merge, and render all charts."""
    df = get_verbruiks_data()
    what, window_size = interface()
    df_nw_beerta = get_weather_info(what)
    merged_df = merge_dataframes(df, what, window_size, df_nw_beerta)
    beste_voor, beste_na = find_best_drempeltemp(merged_df, what)
    afkapgrens_scatter_voor, afkapgrens_scatter_na = interface_afkapgrenzen(beste_voor, beste_na)
    
    bereken_gemiddeld(merged_df, what)
    slope_voor, slope_na = plot_all(merged_df, what, afkapgrens_scatter_voor, afkapgrens_scatter_na)
    st.write(slope_voor)
    make_plots(what, afkapgrens_scatter_voor, afkapgrens_scatter_na, merged_df)
    if what != "graad_dagen":
        multiple_lin_regr(merged_df)
    bereken_verschil(merged_df,what, slope_voor, afkapgrens_scatter_voor )


if __name__ == "__main__":
    print("-------------------------")
    main()