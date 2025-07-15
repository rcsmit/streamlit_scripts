import argparse
import gzip
import logging
import os
from io import StringIO
from typing import Dict, List, Optional, Tuple

# import joblib
import matplotlib.pyplot as plt
import pandas as pd
import requests
import statsmodels.api as sm
# from IPython.display import display
# from joblib import Memory
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# # Setup logging
# logging.basicConfig(level=logging.info, format="%(asctime)s - %(levelname)s - %(message)s")

# # Caching setup
# CACHE_DIR = "./cache"
# memory = Memory(location=CACHE_DIR, verbose=0)

# -------------------- WEATHER -----------------------------
# @memory.cache





# Population split ratios (assumed; can be adjusted based on data)
SPLIT_85_89_RATIO: float = 2 / 3
SPLIT_GE90_RATIO: float = 1 / 3

# Plotting configs
LOWESS_FRAC: float = 0.4  # Fraction for LOWESS smoothing; adjustable


@st.cache_data()
def get_weather_info(country: str) -> pd.DataFrame:
    """Fetch daily max-temp and aggregate to weekly max.

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        pd.DataFrame: DataFrame with year, week, temp_max.

    Raises:
        ValueError: If country not supported.
        requests.RequestException: If API call fails.
    """
    
    # Coordinates and location names for weather data (T proxy)
    COORD: Dict[str, Dict[str, float | str]] = {
        "IT": {"lat": 41.9028, "lon": 12.4964, "loc": "Rome"},
        "ES": {"lat": 40.4168, "lon": -3.7038, "loc": "Madrid"},
        "NL": {"lat": 52.0976, "lon": 5.1790, "loc": "De Bilt"},
        "FR": {"lat": 48.8566, "lon": 2.3522, "loc": "Paris"},
        "DE": {"lat": 52.5200, "lon": 13.4050, "loc": "Berlin"},
        "BE": {"lat": 50.8503, "lon": 4.3517, "loc": "Brussels"},
        "AT": {"lat": 48.2082, "lon": 16.3738, "loc": "Vienna"},
        "PT": {"lat": 38.7223, "lon": -9.1393, "loc": "Lisbon"},
        "CH": {"lat": 46.9481, "lon": 7.4474, "loc": "Bern"},
        "SE": {"lat": 59.3293, "lon": 18.0686, "loc": "Stockholm"},
        "NO": {"lat": 59.9139, "lon": 10.7522, "loc": "Oslo"},
        "DK": {"lat": 55.6761, "lon": 12.5683, "loc": "Copenhagen"},
        "FI": {"lat": 60.1699, "lon": 24.9384, "loc": "Helsinki"},
    }

    WEATHER_START: str = "2000-01-01"
    WEATHER_END: str = "2019-12-31"
    if country not in COORD:
        raise ValueError(f"Unsupported country for weather: {country}")
    lat, lon = COORD[country]["lat"], COORD[country]["lon"]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={WEATHER_START}&end_date={WEATHER_END}"
        f"&daily=temperature_2m_max&timezone=Europe/Berlin"
    )
    print(f"Fetching weather for {country}: {url}")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json().get("daily")
    if not data or "time" not in data or "temperature_2m_max" not in data:
        raise KeyError("Invalid weather API response structure")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(data["time"]),
            "temp_max": data["temperature_2m_max"],
        }
    )
    df["year"] = df["date"].dt.isocalendar().year
    df["week"] = df["date"].dt.isocalendar().week
    df = df.dropna(subset=["year", "week"])
    return df.groupby(["year", "week"], as_index=False)["temp_max"].max()

# ---------------- DEATHS ------------------------------
# @memory.cache
@st.cache_data()
def get_mortality(country: str) -> pd.DataFrame:
    """Fetch weekly deaths via Eurostat (2000‑2019).

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        pd.DataFrame: DataFrame with year, week, age, deaths.

    Raises:
        requests.RequestException: If API call fails.
        ValueError: If data parsing fails.
    """
    base = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/"
        "ESTAT/demo_r_mwk_05/1.0/*.*.*.*.*?"
        "c[freq]=W&c[age]=TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,"
        "Y35-39,Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,"
        "Y80-84,Y85-89,Y_GE90,UNK&c[sex]=T,M,F&c[unit]=NR"
    )
    url = (
        f"{base}&c[geo]={country}&c[TIME_PERIOD]=ge:2000-W01&le:2019-W53"
        "&compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=both"
    )
    print(f"Fetching mortality for {country}: {url}")
    response = requests.get(url)
    response.raise_for_status()
    try:
        txt = gzip.decompress(response.content).decode("utf-8")
    except OSError:
        txt = response.text
    df = pd.read_csv(StringIO(txt), delimiter=",", engine="python", quoting=3)
    df.columns = [c.split(":")[0].strip() for c in df.columns]
    tp_col = next(c for c in df.columns if "TIME_PERIOD" in c.upper())
    yw = df[tp_col].astype(str).str.extract(r"(?P<y>\d{4})-W(?P<w>\d{2})")
    df["year"] = pd.to_numeric(yw["y"], errors="coerce")
    df["week"] = pd.to_numeric(yw["w"], errors="coerce")
    df["age"] = df["age"].astype(str).str.split(":").str[0].str.strip()
    df.rename(columns={"OBS_VALUE": "deaths"}, inplace=True)
    df = df.dropna(subset=["year", "week", "age"])
    aggregated = df.groupby(["year", "week", "age"], as_index=False)["deaths"].sum()
    if aggregated.empty:
        raise ValueError(f"No mortality data found for {country}")
    return aggregated

# ---------------- POPULATION -----------------------------
# @memory.cache
@st.cache_data()
def get_population(country: str) -> pd.DataFrame:
    """Fetch yearly pop (2000‑2019) via Eurostat & split 85+ into 85-89 & 90+.

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        pd.DataFrame: DataFrame with year, age, population.

    Raises:
        requests.RequestException: If API call fails.
        ValueError: If data parsing or splitting fails.
    """
    
    POP_START: str = "2000"
    POP_END: str = "2019"
    ages = (
        "TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,"
        "Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,Y_GE75,"
        "Y80-84,Y_GE80,Y_GE85,UNK"
    )
    base = (
        "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/"
        "ESTAT/demo_pjangroup/1.0/*.*.*.*.*?"
        f"c[freq]=A&c[unit]=NR&c[sex]=T&c[age]={ages}"
    )
    url = (
        f"{base}&c[geo]={country}&c[TIME_PERIOD]=ge:{POP_START}&le:{POP_END}"
        "&compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=name"
    )
    print(f"Fetching population for {country}: {url}")
    response = requests.get(url)
    response.raise_for_status()
    try:
        txt = gzip.decompress(response.content).decode("utf-8")
    except OSError:
        txt = response.text
    df = pd.read_csv(StringIO(txt), delimiter=",", engine="python", quoting=3)
    df.columns = [c.split(":")[0].strip() for c in df.columns]
    tp_col = next(c for c in df.columns if "TIME_PERIOD" in c.upper())
    df["year"] = pd.to_numeric(df[tp_col].astype(str).str[:4], errors="coerce")
    df["age"] = df["age"].astype(str).str.split(":").str[0].str.strip()
    df.rename(columns={"OBS_VALUE": "population"}, inplace=True)
    pop_raw = df[["year", "age", "population"]].dropna().copy()
    if pop_raw.empty:
        raise ValueError(f"No population data found for {country}")
    mask_ge85 = pop_raw["age"] == "Y_GE85"
    if not mask_ge85.any():
        logging.warning(f"No Y_GE85 data for {country}; skipping split ")
    pop_ge85 = pop_raw[mask_ge85]
    pop = pop_raw[~mask_ge85].copy()
    df_85_89 = pop_ge85[["year"]].copy()
    df_85_89["age"] = "Y85-89"
    df_85_89["population"] = pop_ge85["population"] * SPLIT_85_89_RATIO
    df_ge90 = pop_ge85[["year"]].copy()
    df_ge90["age"] = "Y_GE90"
    df_ge90["population"] = pop_ge85["population"] * SPLIT_GE90_RATIO
    pop = pd.concat([pop, df_85_89, df_ge90], ignore_index=True)
    print(f"Population rows after split: {len(pop)} for {country}")
    return pop
# ---------------- ESP2013 ADJUSTMENT ---------------------
def calculate_esp2013_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ESP2013 age-adjusted mortality rates (vectorized).

    Args:
        df (pd.DataFrame): Merged data with death_rate, year, week, age, temp_max.

    Returns:
        pd.DataFrame: DataFrame with year, week, temp_max, esp2013_adjusted_rate.
    """

    
    # ESP2013 standard population weights (per 100,000) for ages 55+
    ESP2013_WEIGHTS: Dict[str, int] = {
        "Y55-59": 6500,
        "Y60-64": 6000,
        "Y65-69": 5500,
        "Y70-74": 5000,
        "Y75-79": 4000,
        "Y80-84": 2500,
        "Y85-89": 1500,
        "Y_GE90": 1000,
    }
    # Filter to relevant ages and pivot for vectorization
    relevant_ages = list(ESP2013_WEIGHTS.keys())
    filtered = df[df["age"].isin(relevant_ages)].copy()
    if filtered.empty:
        logging.warning("No data for ESP2013 ages; returning empty DataFrame")
        return pd.DataFrame()

    # Add weights
    filtered["weight"] = filtered["age"].map(ESP2013_WEIGHTS)
    filtered["weighted_rate"] = filtered["death_rate"] * filtered["weight"]

    # Group and compute adjusted rate
    grouped = filtered.groupby(["year", "week"])
    aggregated = grouped.agg(
        weighted_sum=("weighted_rate", "sum"),
        total_weight=("weight", "sum"),
        temp_max=("temp_max", "first"),  # Assuming consistent per group
    ).reset_index()
    aggregated["esp2013_adjusted_rate"] = (
        aggregated["weighted_sum"] / aggregated["total_weight"]
    )
    aggregated = aggregated.dropna(subset=["esp2013_adjusted_rate"])
    return aggregated[["year", "week", "temp_max", "esp2013_adjusted_rate"]]

# ---------------- MERGE & RATE ---------------------------
def prepare_data(country: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare both age-specific and ESP2013-adjusted data.

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (age_specific_df, esp_adjusted_df)
    """
    mortality_df = get_mortality(country)
    population_df = get_population(country)
    weather_df = get_weather_info(country)
    logging.info(
        f"Data sizes for {country}: mortality={len(mortality_df)}, "
        f"population={len(population_df)}, weather={len(weather_df)}"
    )
    merged = mortality_df.merge(population_df, on=["year", "age"], how="left")
    null_pop = merged["population"].isna().sum()
    if null_pop > 0:
        logging.warning(f"{null_pop} rows with null population after merge for {country}")
    merged["death_rate"] = (merged["deaths"] / merged["population"]) * 1e5
    merged = merged.merge(weather_df, on=["year", "week"], how="inner")
    if len(merged) < len(mortality_df):
        logging.warning(
            f"Data loss after weather merge for {country}: "
            f"{len(mortality_df) - len(merged)} rows dropped"
        )
    esp_df = calculate_esp2013_adjusted(merged)
    return merged, esp_df


# ------------------------ PLOTS -------------------------
def plot_age_specific(
    df: pd.DataFrame, country: str, save_plots: bool = False
) -> None:
    """Plot age-specific mortality vs temperature.

    Args:
        df (pd.DataFrame): Age-specific data.
        country (str): Country code.
        save_plots (bool): If True, save as PNG.
    """
    COORD: Dict[str, Dict[str, float | str]] = {
        "IT": {"lat": 41.9028, "lon": 12.4964, "loc": "Rome"},
        "ES": {"lat": 40.4168, "lon": -3.7038, "loc": "Madrid"},
        "NL": {"lat": 52.0976, "lon": 5.1790, "loc": "De Bilt"},
        "FR": {"lat": 48.8566, "lon": 2.3522, "loc": "Paris"},
        "DE": {"lat": 52.5200, "lon": 13.4050, "loc": "Berlin"},
        "BE": {"lat": 50.8503, "lon": 4.3517, "loc": "Brussels"},
        "AT": {"lat": 48.2082, "lon": 16.3738, "loc": "Vienna"},
        "PT": {"lat": 38.7223, "lon": -9.1393, "loc": "Lisbon"},
        "CH": {"lat": 46.9481, "lon": 7.4474, "loc": "Bern"},
        "SE": {"lat": 59.3293, "lon": 18.0686, "loc": "Stockholm"},
        "NO": {"lat": 59.9139, "lon": 10.7522, "loc": "Oslo"},
        "DK": {"lat": 55.6761, "lon": 12.5683, "loc": "Copenhagen"},
        "FI": {"lat": 60.1699, "lon": 24.9384, "loc": "Helsinki"},
    }
    AGE_BANDS: List[str] = [
        "Y55-59",
        "Y60-64",
        "Y65-69",
        "Y70-74",
        "Y75-79",
        "Y80-84",
        "Y85-89",
        "Y_GE90",
        "TOTAL",
    ]


    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)
    cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "darkblue"])
    for i, age in enumerate(AGE_BANDS):
        ax = axes[i // 3, i % 3]
        sub = df[df["age"] == age]
        if sub.empty:
            ax.set_visible(False)
            continue
        norm = (sub["year"] - sub["year"].min()) / (
            sub["year"].max() - sub["year"].min()
        )
        ax.scatter(sub["temp_max"], sub["death_rate"], c=norm, cmap=cmap, s=20, alpha=0.7)
        trend = sm.nonparametric.lowess(
            sub["death_rate"], sub["temp_max"], frac=LOWESS_FRAC
        )
        ax.plot(trend[:, 0], trend[:, 1], "r-", linewidth=2)
        ax.set_title(age)
        if i // 3 == 2:
            ax.set_xlabel("Temp max (°C)")
        if i % 3 == 0:
            ax.set_ylabel("Deaths per 100k")
    fig.suptitle(f"Mortality vs Temp – {country}", y=1.02)
    t_proxy = COORD[country]["loc"]
    footnotes = (
        f"1) T proxy: {t_proxy} (Open-Meteo) | "
        "2) Mort: Eurostat/demo_r_mwk_05 | "
        "3) Pop: Eurostat/demo_pjangroup (85+ split)  \n"
        "4) Rate=deaths/pop×100k | "
        "5) Each dot represents a week, colored by year (light blue 2000, dark blue - 2019)." 
        "6) Red line shows smoothed trend (LOWESS)." 
        "Plot/code:@orwell2022 Code inspired by:@rcsmit"
    )
    fig.text(0.5, -0.02, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"age_specific_{country}.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)

def plot_esp2013_adjusted(
    esp_df: pd.DataFrame, country: str, save_plots: bool = False
) -> None:
    """Plot ESP2013 age-adjusted mortality vs temperature (icon size).

    Args:
        esp_df (pd.DataFrame): ESP2013 data.
        country (str): Country code.
        save_plots (bool): If True, save as PNG.
    """

    COORD: Dict[str, Dict[str, float | str]] = {
        "IT": {"lat": 41.9028, "lon": 12.4964, "loc": "Rome"},
        "ES": {"lat": 40.4168, "lon": -3.7038, "loc": "Madrid"},
        "NL": {"lat": 52.0976, "lon": 5.1790, "loc": "De Bilt"},
        "FR": {"lat": 48.8566, "lon": 2.3522, "loc": "Paris"},
        "DE": {"lat": 52.5200, "lon": 13.4050, "loc": "Berlin"},
        "BE": {"lat": 50.8503, "lon": 4.3517, "loc": "Brussels"},
        "AT": {"lat": 48.2082, "lon": 16.3738, "loc": "Vienna"},
        "PT": {"lat": 38.7223, "lon": -9.1393, "loc": "Lisbon"},
        "CH": {"lat": 46.9481, "lon": 7.4474, "loc": "Bern"},
        "SE": {"lat": 59.3293, "lon": 18.0686, "loc": "Stockholm"},
        "NO": {"lat": 59.9139, "lon": 10.7522, "loc": "Oslo"},
        "DK": {"lat": 55.6761, "lon": 12.5683, "loc": "Copenhagen"},
        "FI": {"lat": 60.1699, "lon": 24.9384, "loc": "Helsinki"},
    }
    if esp_df.empty:
        logging.warning(f"No ESP2013 data available for {country}")
        return
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))  # Icon size
    cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "darkblue"])
    norm = (esp_df["year"] - esp_df["year"].min()) / (
        esp_df["year"].max() - esp_df["year"].min()
    )
    ax.scatter(
        esp_df["temp_max"],
        esp_df["esp2013_adjusted_rate"],
        c=norm,
        cmap=cmap,
        s=8,
        alpha=0.7,
    )
    trend = sm.nonparametric.lowess(
        esp_df["esp2013_adjusted_rate"], esp_df["temp_max"], frac=LOWESS_FRAC
    )
    ax.plot(trend[:, 0], trend[:, 1], "r-", linewidth=1.5)
    ax.set_xlabel("Temp max (°C)", fontsize=9)
    ax.set_ylabel("ESP2013 Adj. Deaths/100k", fontsize=9)
    ax.set_title(f"ESP2013 Age-Adj. – {country}", fontsize=10)
    ax.tick_params(labelsize=8)
    t_proxy = COORD[country]["loc"]
    footnotes = (
        f"T proxy: {t_proxy} | ESP2013 age-standardized | "
        "Plot:@orwell2022"
    )
    fig.text(0.5, -0.05, footnotes, ha="center", fontsize=7)
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"esp2013_{country}.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)

def plot_combined_esp2013(
    all_esp_data: Dict[str, pd.DataFrame], save_plots: bool = False
) -> None:
    """Plot combined ESP2013 age-adjusted mortality for all countries.

    Args:
        all_esp_data (Dict[str, pd.DataFrame]): ESP data per country.
        save_plots (bool): If True, save as PNG.
    """

    COORD: Dict[str, Dict[str, float | str]] = {
        "IT": {"lat": 41.9028, "lon": 12.4964, "loc": "Rome"},
        "ES": {"lat": 40.4168, "lon": -3.7038, "loc": "Madrid"},
        "NL": {"lat": 52.0976, "lon": 5.1790, "loc": "De Bilt"},
        "FR": {"lat": 48.8566, "lon": 2.3522, "loc": "Paris"},
        "DE": {"lat": 52.5200, "lon": 13.4050, "loc": "Berlin"},
        "BE": {"lat": 50.8503, "lon": 4.3517, "loc": "Brussels"},
        "AT": {"lat": 48.2082, "lon": 16.3738, "loc": "Vienna"},
        "PT": {"lat": 38.7223, "lon": -9.1393, "loc": "Lisbon"},
        "CH": {"lat": 46.9481, "lon": 7.4474, "loc": "Bern"},
        "SE": {"lat": 59.3293, "lon": 18.0686, "loc": "Stockholm"},
        "NO": {"lat": 59.9139, "lon": 10.7522, "loc": "Oslo"},
        "DK": {"lat": 55.6761, "lon": 12.5683, "loc": "Copenhagen"},
        "FI": {"lat": 60.1699, "lon": 24.9384, "loc": "Helsinki"},
    }
    if not all_esp_data:
        logging.warning("No ESP2013 data available for combined plot")
        return
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))  # Larger size for public communication

    # Define unique colormaps and mid-colors for each country
    country_cmaps = {
        "IT": LinearSegmentedColormap.from_list("it_blues", ["lightblue", "darkblue"]),
        "ES": LinearSegmentedColormap.from_list("es_oranges", ["#FFDAB9", "darkorange"]),
        "NL": LinearSegmentedColormap.from_list("nl_greens", ["lightgreen", "darkgreen"]),
        "FR": LinearSegmentedColormap.from_list("fr_purples", ["lavender", "darkviolet"]),
    }
    country_mid_colors = {
        "IT": "blue",
        "ES": "orange",
        "NL": "green",
        "FR": "purple",
    }

    # For legend: Create proxy lines
    from matplotlib.lines import Line2D
    legend_handles = []

    for country, esp_df in all_esp_data.items():
        if esp_df.empty:
            continue
        norm = (esp_df["year"] - esp_df["year"].min()) / (
            esp_df["year"].max() - esp_df["year"].min()
        )
        cmap = country_cmaps.get(country, LinearSegmentedColormap.from_list("default", ["lightgray", "darkgray"]))
        ax.scatter(
            esp_df["temp_max"],
            esp_df["esp2013_adjusted_rate"],
            c=norm,
            cmap=cmap,
            s=3,
            alpha=0.6,
        )
        trend = sm.nonparametric.lowess(
            esp_df["esp2013_adjusted_rate"], esp_df["temp_max"], frac=LOWESS_FRAC
        )
        mid_color = country_mid_colors.get(country, "red")
        ax.plot(
            trend[:, 0],
            trend[:, 1],
            color=mid_color,
            linewidth=2.5,
            alpha=0.8,
        )
        
        # Add country name label near the end of the LOESS line
        label_x = trend[-1, 0] + 0.5  # Slight offset to the right
        label_y = trend[-1, 1]
        ax.text(label_x, label_y, country, color=mid_color, fontsize=10, fontweight="bold")

        # Add proxy for legend (line instead of dots)
        legend_handles.append(Line2D([0], [0], color=mid_color, linewidth=2.5, label=country))

    ax.set_xlabel("Temp max (°C)", fontsize=12)
    ax.set_ylabel("ESP2013 Adj. Deaths/100k", fontsize=12)
    ax.set_title("ESP2013 Age-Adj. – All Countries", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)  # Light grid for better readability

    t_proxies = [
        f"{country}:{COORD[country]['loc']}"
        for country in all_esp_data.keys()
        if country in COORD
    ]
    footnotes = (
        f"T proxies: {', '.join(t_proxies)} | ESP2013 age-standardized | "
        "Plot:@orwell2022"
    )
    fig.text(0.5, -0.02, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    if save_plots:
        plt.savefig("esp2013_combined.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)

# ------------------------- MAIN ---------------------------
def main_orwell_esp2013_(countries: List[str], save_plots: bool = False) -> None:
    """Main execution function.

    Args:
        countries (List[str]): List of country codes to process.
        save_plots (bool): If True, save plots as PNG files.
    """
    
    # Coordinates and location names for weather data (T proxy)
    COORD: Dict[str, Dict[str, float | str]] = {
        "IT": {"lat": 41.9028, "lon": 12.4964, "loc": "Rome"},
        "ES": {"lat": 40.4168, "lon": -3.7038, "loc": "Madrid"},
        "NL": {"lat": 52.0976, "lon": 5.1790, "loc": "De Bilt"},
        "FR": {"lat": 48.8566, "lon": 2.3522, "loc": "Paris"},
        "DE": {"lat": 52.5200, "lon": 13.4050, "loc": "Berlin"},
        "BE": {"lat": 50.8503, "lon": 4.3517, "loc": "Brussels"},
        "AT": {"lat": 48.2082, "lon": 16.3738, "loc": "Vienna"},
        "PT": {"lat": 38.7223, "lon": -9.1393, "loc": "Lisbon"},
        "CH": {"lat": 46.9481, "lon": 7.4474, "loc": "Bern"},
        "SE": {"lat": 59.3293, "lon": 18.0686, "loc": "Stockholm"},
        "NO": {"lat": 59.9139, "lon": 10.7522, "loc": "Oslo"},
        "DK": {"lat": 55.6761, "lon": 12.5683, "loc": "Copenhagen"},
        "FI": {"lat": 60.1699, "lon": 24.9384, "loc": "Helsinki"},
    }
    invalid_countries = [c for c in countries if c not in COORD]
    if invalid_countries:
        raise ValueError(f"Invalid countries: {invalid_countries}")
    all_esp_data: Dict[str, pd.DataFrame] = {}
    for country in countries:
        print(f"\n{'='*50}\nProcessing {country}\n{'='*50}")
        # try:
        if 1==1:
            age_specific_df, esp_adjusted_df = prepare_data(country)
            print(f"\n{country} sample data:")
            # if "IPython" in globals():
            #     display(age_specific_df.head())
            # else:
            print(age_specific_df.head())
            plot_age_specific(age_specific_df, country, save_plots)
            print(f"\n{country} ESP2015 sample data:")
            # if "IPython" in globals():
            #     display(esp_adjusted_df.head())
            # else:
            print(esp_adjusted_df.head())
            plot_esp2013_adjusted(esp_adjusted_df, country, save_plots)
            all_esp_data[country] = esp_adjusted_df
        # except (requests.RequestException, ValueError, KeyError) as e:
        #     st.error(f"Error processing {country}: {e}")
        #     continue
    print(f"\n{'='*50}\nCombined ESP2015 Plot\n{'='*50}")
    plot_combined_esp2013(all_esp_data, save_plots)
    
def main_orwell_esp2013():  
    # https://x.com/orwell2022/status/1945102023705203154  
    # ------------------------- CONFIG -------------------------

    countries_list = ["IT", "ES", "NL", "FR"]
    save_plots = False
    main_orwell_esp2013_(countries_list, save_plots)



if __name__ == "__main__":
    main_orwell_esp2013()
    # Detect if running in Jupyter
    # is_jupyter = False
    # try:
    #     from IPython import get_ipython
    #     if get_ipython() is not None:
    #         is_jupyter = True
    # except ImportError:
    #     pass  # Not in Jupyter

    # if is_jupyter:
    #     # Jupyter mode: Use defaults (override manually if needed)
    #     print("Running in Jupyter mode - using default arguments")
    #     countries_list = ["IT", "ES", "NL", "FR"]
    #     save_plots = False
    # else:
    #     # CLI mode: Parse arguments
    #     parser = argparse.ArgumentParser(
    #         description="Analyze mortality vs weather for countries."
    #     )
    #     parser.add_argument(
    #         "--countries",
    #         type=str,
    #         default="IT,ES,NL,FR",
    #         help="Comma-separated country codes (e.g., IT,ES)",
    #     )
    #     parser.add_argument(
    #         "--save-plots", action="store_true", help="Save plots as PNG files"
    #     )
    #     args = parser.parse_args()
    #     countries_list = args.countries.split(",")
    #     save_plots = args.save_plots



    
# # ------------------------- TESTS -------------------------
# # Run with pytest: Save this file and run `pytest <filename>.py`
# def test_esp2015_calculation():
#     # Mock data
#     mock_df = pd.DataFrame(
#         {
#             "year": [2000, 2000],
#             "week": [1, 1],
#             "age": ["Y55-59", "Y60-64"],
#             "death_rate": [10.0, 20.0],
#             "temp_max": [15.0, 15.0],
#         }
#     )
#     result = calculate_esp2015_adjusted(mock_df)
#     assert not result.empty
#     assert "esp2015_adjusted_rate" in result.columns
#     # Expected: (10*7000 + 20*6000) / (7000+6000) = 14.615...
#     assert abs(result["esp2015_adjusted_rate"].iloc[0] - 14.615) < 0.01

# # If running tests manually
# if os.environ.get("RUN_TESTS"):
#     test_esp2015_calculation()
#     print("All tests passed!")