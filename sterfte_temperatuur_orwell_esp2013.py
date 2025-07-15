# Mortality × Weather analysis, 2000‑2019, Multiple Countries
# ----------------------------------------------------
# Jupyter/CLI script – fetches:
#   • Weekly deaths    (Eurostat demo_r_mwk_05, 2000‑2019)
#   • Yearly population (Eurostat demo_pjangroup, 2015‑2019)
#   • Daily max‑temp    (Open‑Meteo archive, 2000‑2019)
# Computes age‑specific death‑rate per 1000 and plots 3×3 grids + ESP2013 adjusted.
# Includes footnotes and credits.
#
# Python Version: 3.8+ recommended
# Dependencies (save as requirements.txt):
# pandas>=1.5
# requests
# matplotlib
# statsmodels
# joblib  # For caching
# ----------------------------------------------------

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
# from joblib import Memory
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

# Setup logging
# logging.basicConfig(level=print, format="%(asctime)s - %(levelname)s - %(message)s")

# Caching setup
# CACHE_DIR = "./cache"
# memory = Memory(location=CACHE_DIR, verbose=0)

# ------------------------- CONFIG -------------------------
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

# ESP2013 standard population weights (19 age groups, totaling 100,000)
# These represent the standard population structure
# 5000 5500 5500 5500 6000 6000 6500 7000 7000 7000 7000 6500 6000 5500 5000 4000 2500 1500 1000
ESP2013_WEIGHTS: Dict[str, int] = {
    "Y_LT5": 5000,    # 0-4 (combined 0 + 1-4)
    "Y5-9": 5500,
    "Y10-14": 5500,
    "Y15-19": 5500,
    "Y20-24": 6000,
    "Y25-29": 6000,
    "Y30-34": 6500,
    "Y35-39": 7000,
    "Y40-44": 7000,
    "Y45-49": 7000,
    "Y50-54": 7000,
    "Y55-59": 6500,
    "Y60-64": 6000,
    "Y65-69": 5500,
    "Y70-74": 5000,
    "Y75-79": 4000,
    "Y80-84": 2500,
    "Y85-89": 1500,
    "Y_GE90": 1000,   # 90+ (combined 90-94 + 95+)
}

# Verify ESP2013 weights sum to 100,000
assert sum(ESP2013_WEIGHTS.values()) == 100000, f"ESP2013 weights sum to {sum(ESP2013_WEIGHTS.values())}, should be 100,000"

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
POP_START: str = "2000"
POP_END: str = "2019"

# Population split ratios (based on typical demographic patterns)
SPLIT_85_89_RATIO: float = 0.67  # About 67% of 85+ are 85-89
SPLIT_GE90_RATIO: float = 0.33   # About 33% of 85+ are 90+

# Plotting configs
LOWESS_FRAC: float = 0.4  # Fraction for LOWESS smoothing; adjustable

# -------------------- WEATHER -----------------------------
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
@st.cache_data()
def get_mortality(country: str) -> pd.DataFrame:
    """Fetch weekly deaths via Eurostat (2000‑2019).
    
    CORRECTED: Excludes UNK age group to avoid double counting.

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
        "Y80-84,Y85-89,Y_GE90&c[sex]=T&c[unit]=NR"  # Removed UNK
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
    
    # Filter out UNK and any other unwanted age groups
    valid_ages = list(ESP2013_WEIGHTS.keys()) + ["TOTAL"]
    df = df[df["age"].isin(valid_ages)]
    
    aggregated = df.groupby(["year", "week", "age"], as_index=False)["deaths"].sum()
    if aggregated.empty:
        raise ValueError(f"No mortality data found for {country}")
    
    print(f"Mortality age groups found: {sorted(aggregated['age'].unique())}")
    return aggregated

# ---------------- POPULATION (CORRECTED) -----------------------------
@st.cache_data()
def get_population(country: str) -> pd.DataFrame:
    """Fetch yearly pop (2000‑2019) via Eurostat & split 85+ into 85-89 & 90+.
    
    CORRECTED: Ensures no double counting and proper age group handling.

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        pd.DataFrame: DataFrame with year, age, population.

    Raises:
        requests.RequestException: If API call fails.
        ValueError: If data parsing or splitting fails.
    """
    ages = (
        "TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,"
        "Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,"
        "Y80-84,Y_GE85"  # Clean age groups, will split Y_GE85
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
    
    # CRITICAL: Handle 85+ split without double counting
    mask_ge85 = pop_raw["age"] == "Y_GE85"
    pop_ge85 = pop_raw[mask_ge85].copy()
    pop_base = pop_raw[~mask_ge85].copy()  # All ages EXCEPT Y_GE85
    
    if not pop_ge85.empty:
        # Split Y_GE85 into Y85-89 and Y_GE90 (these replace Y_GE85, don't add to it)
        df_85_89 = pop_ge85.copy()
        df_85_89["age"] = "Y85-89"
        df_85_89["population"] = pop_ge85["population"] * SPLIT_85_89_RATIO
        
        df_ge90 = pop_ge85.copy()
        df_ge90["age"] = "Y_GE90"
        df_ge90["population"] = pop_ge85["population"] * SPLIT_GE90_RATIO
        
        # Final population: base (without Y_GE85) + split components
        pop_final = pd.concat([pop_base, df_85_89, df_ge90], ignore_index=True)
        print(f"Split Y_GE85 for {country}: 85-89={SPLIT_85_89_RATIO:.2f}, 90+={SPLIT_GE90_RATIO:.2f}")
    else:
        print(f"No Y_GE85 data for {country}; using base population")
        pop_final = pop_base
    
    # Verify no Y_GE85 remains in final data
    if "Y_GE85" in pop_final["age"].values:
        raise ValueError(f"Y_GE85 still present in final population data for {country}")
    
    print(f"Population age groups: {sorted(pop_final['age'].unique())}")
    print(f"Population rows after processing: {len(pop_final)} for {country}")
    return pop_final

# ---------------- ESP2013 ADJUSTMENT (CORRECTED) ---------------------
def calculate_esp2013_adjusted_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ESP2013 age-adjusted mortality rates (weekly, converted to annual per 1000).
    
    CORRECTED: Proper ESP2013 standardization with validation.

    Args:
        df (pd.DataFrame): Merged data with deaths, population, year, week, age, temp_max.

    Returns:
        pd.DataFrame: DataFrame with year, week, temp_max, esp2013_adjusted_rate_per_1000.
    """
    df = df.copy()
    
    # Calculate age-specific death rates per 1000 per year
    df['death_rate_per_1000_annual'] = (df['deaths'] / df['population']) * 1000 * 52.18
    
    # Filter to ESP2013 relevant ages only
    esp_ages = list(ESP2013_WEIGHTS.keys())
    filtered = df[df['age'].isin(esp_ages)].copy()
    
    if filtered.empty:
        print("No data for ESP2013 ages; returning empty DataFrame")
        return pd.DataFrame()
    
    # Check which ESP2013 ages we have data for
    available_ages = set(filtered['age'].unique())
    missing_ages = set(esp_ages) - available_ages
    if missing_ages:
        print(f"Missing ESP2013 age groups: {missing_ages}")
    
    # Calculate proportional weights for available ages only
    available_weights = {age: ESP2013_WEIGHTS[age] for age in available_ages}
    total_available_weight = sum(available_weights.values())
    
    # Add proportional weights (sum to 1.0 for available ages)
    filtered['esp_weight'] = filtered['age'].map(available_weights) / total_available_weight
    
    # Calculate weighted death rates
    filtered['weighted_rate'] = filtered['death_rate_per_1000_annual'] * filtered['esp_weight']
    
    # Group by year, week and sum weighted rates
    esp_adjusted = filtered.groupby(['year', 'week']).agg({
        'weighted_rate': 'sum',
        'temp_max': 'first',
        'esp_weight': 'sum'  # Should sum to 1.0 for validation
    }).reset_index()
    
    esp_adjusted['esp2013_adjusted_rate_per_1000'] = esp_adjusted['weighted_rate']
    
    # Validation logging
    if not esp_adjusted.empty:
        avg_rate = esp_adjusted['esp2013_adjusted_rate_per_1000'].mean()
        weight_sum = esp_adjusted['esp_weight'].iloc[0]
        age_coverage = len(available_ages)
        weight_coverage = total_available_weight / 100000  # Fraction of ESP2013 population covered
        
        print(f"ESP2013 weekly validation:")
        print(f"  - Average rate: {avg_rate:.2f} per 1000")
        print(f"  - Weight sum: {weight_sum:.3f} (should be 1.0)")
        print(f"  - Age groups: {age_coverage}/19 ESP2013 groups")
        print(f"  - Population coverage: {weight_coverage:.1%}")
        print(f"  - Available ages: {sorted(available_ages)}")
    
    return esp_adjusted[['year', 'week', 'temp_max', 'esp2013_adjusted_rate_per_1000']]

def calculate_esp2013_adjusted_annual(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ESP2013 age-adjusted mortality rates (annual, per 1000).
    
    CORRECTED: Proper annual aggregation and ESP2013 standardization.

    Args:
        df (pd.DataFrame): Merged data with deaths, population, year, age
        
    Returns:
        pd.DataFrame: DataFrame with year, esp2013_adjusted_rate_per_1000
    """
    # Aggregate weekly deaths to annual deaths
    annual_deaths = df.groupby(['year', 'age'])['deaths'].sum().reset_index()
    
    # Get annual population (consistent across weeks)
    annual_pop = df.groupby(['year', 'age'])['population'].first().reset_index()
    
    # Merge and calculate annual death rates per 1000
    annual_data = annual_deaths.merge(annual_pop, on=['year', 'age'])
    annual_data['death_rate_per_1000'] = (annual_data['deaths'] / annual_data['population']) * 1000
    
    # Filter to ESP2013 relevant ages only
    esp_ages = list(ESP2013_WEIGHTS.keys())
    filtered = annual_data[annual_data['age'].isin(esp_ages)].copy()
    
    if filtered.empty:
        print("No data for ESP2013 ages; returning empty DataFrame")
        return pd.DataFrame()
    
    # Check age coverage
    available_ages = set(filtered['age'].unique())
    missing_ages = set(esp_ages) - available_ages
    if missing_ages:
        print(f"Missing ESP2013 age groups in annual data: {missing_ages}")
    
    # Calculate proportional weights for available ages
    available_weights = {age: ESP2013_WEIGHTS[age] for age in available_ages}
    total_available_weight = sum(available_weights.values())
    
    # Add proportional weights
    filtered['esp_weight'] = filtered['age'].map(available_weights) / total_available_weight
    
    # Calculate weighted death rates
    filtered['weighted_rate'] = filtered['death_rate_per_1000'] * filtered['esp_weight']
    
    # Group by year and sum weighted rates
    esp_adjusted = filtered.groupby('year').agg({
        'weighted_rate': 'sum',
        'esp_weight': 'sum'  # Should sum to 1.0 for validation
    }).reset_index()
    
    esp_adjusted['esp2013_adjusted_rate_per_1000'] = esp_adjusted['weighted_rate']
    
    # Validation logging
    if not esp_adjusted.empty:
        avg_rate = esp_adjusted['esp2013_adjusted_rate_per_1000'].mean()
        weight_sum = esp_adjusted['esp_weight'].iloc[0]
        age_coverage = len(available_ages)
        weight_coverage = total_available_weight / 100000
        
        print(f"Annual ESP2013 validation:")
        print(f"  - Average rate: {avg_rate:.2f} per 1000")
        print(f"  - Weight sum: {weight_sum:.3f} (should be 1.0)")
        print(f"  - Age groups: {age_coverage}/19 ESP2013 groups")
        print(f"  - Population coverage: {weight_coverage:.1%}")
    
    return esp_adjusted[['year', 'esp2013_adjusted_rate_per_1000']]

# ---------------- MERGE & RATE (CORRECTED) ---------------------------
def prepare_data(country: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare data with corrected rate calculations and validation.

    Args:
        country (str): Country code (e.g., 'IT').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (age_specific_df, esp_weekly_df, esp_annual_df)
    """
    mortality_df = get_mortality(country)
    population_df = get_population(country)
    weather_df = get_weather_info(country)
    
    print(
        f"Data sizes for {country}: mortality={len(mortality_df)}, "
        f"population={len(population_df)}, weather={len(weather_df)}"
    )
    
    # Merge mortality with population
    merged = mortality_df.merge(population_df, on=["year", "age"], how="left")
    null_pop = merged["population"].isna().sum()
    if null_pop > 0:
        print(f"{null_pop} rows with null population after merge for {country}")
        # Log which age groups have missing population
        missing_pop_ages = merged[merged["population"].isna()]["age"].unique()
        print(f"Age groups with missing population: {missing_pop_ages}")
        # Remove rows with null population
        merged = merged.dropna(subset=["population"])
    
    # Calculate weekly death rate per 1000 (for age-specific plots)
    merged["weekly_rate_per_1000"] = (merged["deaths"] / merged["population"]) * 1000
    merged["annual_equiv_rate_per_1000"] = merged["weekly_rate_per_1000"] * 52.18
    
    # Merge with weather data
    merged = merged.merge(weather_df, on=["year", "week"], how="inner")
    if len(merged) < len(mortality_df):
        print(
            f"Data loss after weather merge for {country}: "
            f"{len(mortality_df) - len(merged)} rows dropped"
        )
    
    # Log final age groups in merged data
    final_ages = sorted(merged["age"].unique())
    print(f"Final age groups in merged data: {final_ages}")
    
    # Calculate ESP2013 adjustments
    esp_weekly_df = calculate_esp2013_adjusted_weekly(merged)
    esp_annual_df = calculate_esp2013_adjusted_annual(merged)
    
    return merged, esp_weekly_df, esp_annual_df

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
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=True)
    cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "darkblue"])
    
    for i, age in enumerate(AGE_BANDS):
        ax = axes[i // 3, i % 3]
        sub = df[df["age"] == age]
        if sub.empty:
            ax.set_visible(False)
            continue
        
        # Color by year
        if len(sub) > 1:
            norm = (sub["year"] - sub["year"].min()) / max(1, (sub["year"].max() - sub["year"].min()))
        else:
            norm = [0.5] * len(sub)
        ax.scatter(sub["temp_max"], sub["annual_equiv_rate_per_1000"], c=norm, cmap=cmap, s=20, alpha=0.7)
        
        # Add LOWESS trend
        if len(sub) > 10:  # Need sufficient data for LOWESS
            try:
                trend = sm.nonparametric.lowess(
                    sub["annual_equiv_rate_per_1000"], sub["temp_max"], frac=LOWESS_FRAC
                )
                ax.plot(trend[:, 0], trend[:, 1], "r-", linewidth=2)
            except Exception as e:
                print(f"LOWESS failed for {age} in {country}: {e}")
        
        ax.set_title(age, fontsize=10)
        if i // 3 == 2:
            ax.set_xlabel("Temp max (°C)")
        if i % 3 == 0:
            ax.set_ylabel("Deaths per 1000 (annual equiv.)")
    
    fig.suptitle(f"Mortality vs Temp – {country}", y=0.98, fontsize=14)
    
    t_proxy = COORD[country]["loc"]
    footnotes = (
        f"1) T proxy: {t_proxy} (Open-Meteo) | "
        "2) Mort: Eurostat/demo_r_mwk_05 | "
        "3) Pop: Eurostat/demo_pjangroup (85+ split)  \n"
        "4) Rate=deaths/pop×1000 (annual equiv.) | "
        "5) Each dot represents a week, colored by year (light blue 2000, dark blue - 2019)." 
        "6) Red line shows smoothed trend (LOWESS)." 
        "Code/Plot:@orwell2022 Code inspired by:@rcsmit"
    )
    fig.text(0.5, -0.04, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"age_specific_{country}.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)
    plt.close()

def plot_esp2013_adjusted(
    esp_df: pd.DataFrame, country: str, save_plots: bool = False, plot_type: str = "weekly"
) -> None:
    """Plot ESP2013 age-adjusted mortality vs temperature.

    Args:
        esp_df (pd.DataFrame): ESP2013 data.
        country (str): Country code.
        save_plots (bool): If True, save as PNG.
        plot_type (str): "weekly" or "annual"
    """
    if esp_df.empty:
        print(f"No ESP2013 data available for {country}")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "darkblue"])
    
    if plot_type == "weekly":
        # Color by year for weekly data
        if len(esp_df) > 1:
            norm = (esp_df["year"] - esp_df["year"].min()) / max(1, (esp_df["year"].max() - esp_df["year"].min()))
        else:
            norm = [0.5] * len(esp_df)
        ax.scatter(
            esp_df["temp_max"],
            esp_df["esp2013_adjusted_rate_per_1000"],
            c=norm,
            cmap=cmap,
            s=8,
            alpha=0.7,
        )
        
        # Add LOWESS trend
        if len(esp_df) > 10:
            try:
                trend = sm.nonparametric.lowess(
                    esp_df["esp2013_adjusted_rate_per_1000"], esp_df["temp_max"], frac=LOWESS_FRAC
                )
                ax.plot(trend[:, 0], trend[:, 1], "r-", linewidth=1.5)
            except Exception as e:
                print(f"LOWESS failed for ESP2013 weekly {country}: {e}")
        
        ax.set_xlabel("Temp max (°C)", fontsize=10)
        ax.set_ylabel("ESP2013 Adj. Deaths/1000", fontsize=10)
        ax.set_title(f"ESP2013 Age-Adj. (Weekly) – {country}", fontsize=12)
        
    else:  # annual
        ax.plot(esp_df["year"], esp_df["esp2013_adjusted_rate_per_1000"], 'bo-', linewidth=2, markersize=6)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel("ESP2013 Adj. Deaths/1000", fontsize=10)
        ax.set_title(f"ESP2013 Age-Adj. (Annual) – {country}", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    ax.tick_params(labelsize=9)
    
    t_proxy = COORD[country]["loc"]
    footnotes = (
        f"T proxy: {t_proxy} | ESP2013 age-standardized | "
        "Plot:@orwell2022"
    )
    fig.text(0.5, -0.03, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f"esp2013_{plot_type}_{country}.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)
    plt.close()
def plot_combined_esp2013(
    all_esp_data: Dict[str, pd.DataFrame], save_plots: bool = False
) -> None:
    """Plot combined ESP2013 age-adjusted mortality for all countries.

    Args:
        all_esp_data (Dict[str, pd.DataFrame]): ESP data per country.
        save_plots (bool): If True, save as PNG.
    """
    if not all_esp_data:
        print("No ESP2013 data available for combined plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Define unique colors for each country
    country_colors = {
        "IT": "blue",
        "ES": "orange", 
        "NL": "green",
        "FR": "purple",
        "DE": "red",
        "BE": "brown",
        "AT": "pink",
        "PT": "gray",
        "CH": "olive",
        "SE": "cyan",
        "NO": "magenta",
        "DK": "yellow",
        "FI": "black",
    }

    # For legend
    from matplotlib.lines import Line2D
    legend_handles = []

    for country, esp_df in all_esp_data.items():
        if esp_df.empty:
            continue
        
        color = country_colors.get(country, "gray")
        
        # Scatter plot with low alpha
        ax.scatter(
            esp_df["temp_max"],
            esp_df["esp2013_adjusted_rate_per_1000"],
            c=color,
            s=3,
            alpha=0.3,
        )
        
        # Add LOWESS trend
        if len(esp_df) > 10:
            try:
                trend = sm.nonparametric.lowess(
                    esp_df["esp2013_adjusted_rate_per_1000"], esp_df["temp_max"], frac=LOWESS_FRAC
                )
                ax.plot(
                    trend[:, 0],
                    trend[:, 1],
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                )
                
                # Add country name label near the end of the LOESS line
                label_x = trend[-1, 0] + 0.5
                label_y = trend[-1, 1]
                ax.text(label_x, label_y, country, color=color, fontsize=10, fontweight="bold")
            except Exception as e:
                print(f"LOWESS failed for combined plot {country}: {e}")

        # Add proxy for legend
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5, label=country))

    ax.set_xlabel("Temp max (°C)", fontsize=12)
    ax.set_ylabel("ESP2013 Adj. Deaths/1000", fontsize=12)
    ax.set_title("ESP2013 Age-Adj. Mortality vs Temperature – All Countries", fontsize=14)
    ax.tick_params(labelsize=10)
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.3)

    t_proxies = [
        f"{country}:{COORD[country]['loc']}"
        for country in all_esp_data.keys()
        if country in COORD
    ]
    footnotes = (
        f"T proxies: {', '.join(t_proxies)} | ESP2013 age-standardized (per 1000) | "
        "Plot:@orwell2022"
    )
    fig.text(0.5, -0.04, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig("esp2013_combined.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)
    plt.close()

def plot_combined_esp2013_annual(
    all_esp_data: Dict[str, pd.DataFrame], save_plots: bool = False
) -> None:
    """Plot combined ESP2013 age-adjusted mortality for all countries.

    Args:
        all_esp_data (Dict[str, pd.DataFrame]): ESP data per country.
        save_plots (bool): If True, save as PNG.
    """
    if not all_esp_data:
        print("No ESP2013 data available for combined plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Define unique colors for each country
    country_colors = {
        "IT": "blue",
        "ES": "orange", 
        "NL": "green",
        "FR": "purple",
        "DE": "red",
        "BE": "brown",
        "AT": "pink",
        "PT": "gray",
        "CH": "olive",
        "SE": "cyan",
        "NO": "magenta",
        "DK": "yellow",
        "FI": "black",
    }

    # For legend
    from matplotlib.lines import Line2D
    legend_handles = []


    for country, esp_df in all_esp_data.items():
        if esp_df.empty:
            continue
        
        color = country_colors.get(country, "gray")
        ax.plot(esp_df["year"], esp_df["esp2013_adjusted_rate_per_1000"], 'bo-', linewidth=1,  c=color, markersize=2)
        
        # # Scatter plot with low alpha
        # ax.scatter(
        #     esp_df["year"],
        #     esp_df["esp2013_adjusted_rate_per_1000"],
        #     c=color,
        #     s=3,
        #     alpha=0.3,
        # )
        
        # Add LOWESS trend
        if len(esp_df) > 10:
            try:
                trend = sm.nonparametric.lowess(
                    esp_df["esp2013_adjusted_rate_per_1000"], esp_df["temp_max"], frac=LOWESS_FRAC
                )
                ax.plot(
                    trend[:, 0],
                    trend[:, 1],
                    color=color,
                    linewidth=2.5,
                    alpha=0.8,
                )
                
                # Add country name label near the end of the LOESS line
                label_x = trend[-1, 0] + 0.5
                label_y = trend[-1, 1]
                ax.text(label_x, label_y, country, color=color, fontsize=10, fontweight="bold")
            except Exception as e:
                print(f"LOWESS failed for combined plot {country}: {e}")

        # Add proxy for legend
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2.5, label=country))

    # ax.plot(esp_df["year"], esp_df["esp2013_adjusted_rate_per_1000"], 'bo-', linewidth=2, markersize=6)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel("ESP2013 Adj. Deaths/1000", fontsize=10)
    ax.set_title(f"ESP2013 Age-Adj. (Annual) – all countries", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=10, loc="upper right")
   

    t_proxies = [
        f"{country}:{COORD[country]['loc']}"
        for country in all_esp_data.keys()
        if country in COORD
    ]
    footnotes = (
        f"ESP2013 age-standardized (per 1000) | "
        "Plot:@orwell2022"
    )
    fig.text(0.5, -0.04, footnotes, ha="center", fontsize=8)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig("esp2013_combined_annual.png", bbox_inches="tight", dpi=300)
    # plt.show()
    st.pyplot(fig)
    plt.close()
# ------------------------- MAIN ---------------------------
def main_orwell_esp2013_(countries: List[str], save_plots: bool = False) -> None:
    """Main execution function.

    Args:
        countries (List[str]): List of country codes to process.
        save_plots (bool): If True, save plots as PNG files.
    """
    invalid_countries = [c for c in countries if c not in COORD]
    if invalid_countries:
        raise ValueError(f"Invalid countries: {invalid_countries}")
    
    all_esp_data: Dict[str, pd.DataFrame] = {}
    all_esp_data_annual: Dict[str, pd.DataFrame] = {}
    for country in countries:
        st.subheader(f"\n{'='*5}\n{country}\n{'='*5}")
        try:
        #if 1==1:
            age_specific_df, esp_weekly_df, esp_annual_df = prepare_data(country)
            
            print(f"\n{country} sample age-specific data:")
            # try:
            #     from IPython.display import display
            #     display(age_specific_df.head())
            # except ImportError:
            print(age_specific_df.head())
            
            # Plot age-specific mortality
            plot_age_specific(age_specific_df, country, save_plots)
            
            # Plot ESP2013 weekly
            print(f"\n{country} ESP2013 weekly sample data:")
            # try:
            #     display(esp_weekly_df.head())
            # except ImportError:
            print(esp_weekly_df.head())
            
            # Plot ESP2013 annual
            print(f"\n{country} ESP2013 annual sample data:")
            # try:
            #     display(esp_annual_df.head())
            # except ImportError:
            print(esp_annual_df.head())
            plot_esp2013_adjusted(esp_weekly_df, country, save_plots, "weekly")
            plot_esp2013_adjusted(esp_annual_df, country, save_plots, "annual")
            
            all_esp_data[country] = esp_weekly_df
            all_esp_data_annual[country] = esp_annual_df
            
        except (requests.RequestException, ValueError, KeyError) as e:
            st.error(f"Error processing {country}: {e}")
            continue
    
    # Combined plot
    st.subheader(f"\n{'='*5}\nCombined ESP2013 Plot\n{'='*5}")
    plot_combined_esp2013(all_esp_data, save_plots)
    plot_combined_esp2013_annual(all_esp_data_annual, save_plots)
    
def main_orwell_esp2013():
    countries_list = ["IT", "ES", "NL", "FR"]  # RESTORED
    save_plots = True
    
    # https://x.com/orwell2022/status/1945102023705203154  
    # ------------------------- CONFIG -------------------------
    country_codes = ["IT", "ES", "NL", "FR", "DE", "BE", "AT", "PT", "CH", "SE", "NO", "DK", "FI"]

    countries_list = st.multiselect("Countries", country_codes, ["IT", "ES", "NL", "FR"])
    save_plots = False
    if len(countries_list)>0:
        main_orwell_esp2013_(countries_list, save_plots)

    else:
        st.error("Select countries")
        st.stop()
    # main(countries_list, save_plots)


if __name__ == "__main__":
    main_orwell_esp2013()

    # # Detect if running in Jupyter
    # is_jupyter = False
    # try:
    #     from IPython import get_ipython
    #     if get_ipython() is not None:
    #         is_jupyter = True
    # except ImportError:
    #     pass

    # if is_jupyter:
    #     # Jupyter mode: Use defaults (RESTORED ALL COUNTRIES)
    #     print("Running in Jupyter mode - using default arguments")
        # countries_list = ["IT", "ES", "NL", "FR"]  # RESTORED
        # save_plots = True
    # else:
    #     # CLI mode: Parse arguments
    #     parser = argparse.ArgumentParser(
    #         description="Analyze mortality vs weather for countries."
    #     )
    #     parser.add_argument(
    #         "--countries",
    #         type=str,
    #         default="IT,ES,NL,FR",  # RESTORED
    #         help="Comma-separated country codes (e.g., IT,ES)",
    #     )
    #     parser.add_argument(
    #         "--save-plots", action="store_true", help="Save plots as PNG files"
    #     )
    #     args = parser.parse_args()
    #     countries_list = args.countries.split(",")
    #     save_plots = args.save_plots

# ------------------------- TESTS -------------------------
def test_esp2013_calculation():
    """Test ESP2013 calculation with mock data."""
    # Mock data with realistic Netherlands-like values
    mock_df = pd.DataFrame({
        "year": [2015] * 8,
        "week": [1] * 8,
        "age": ["Y_LT5", "Y55-59", "Y60-64", "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89"],
        "deaths": [2, 50, 80, 120, 180, 250, 300, 200],  # Realistic weekly deaths
        "population": [900000, 1200000, 1100000, 900000, 700000, 500000, 300000, 150000],  # Realistic populations
        "temp_max": [5.0] * 8,
    })
    
    result = calculate_esp2013_adjusted_weekly(mock_df)
    assert not result.empty
    assert "esp2013_adjusted_rate_per_1000" in result.columns
    
    # Check that the result is reasonable (should be around 9-11 per 1000 for Netherlands)
    rate = result["esp2013_adjusted_rate_per_1000"].iloc[0]
    assert 3 < rate < 20, f"Rate {rate:.2f} per 1000 seems unreasonable for Netherlands"
    
    print(f"Test passed! ESP2013 rate: {rate:.2f} per 1000")

# # Run tests if requested
# if os.environ.get("RUN_TESTS"):
#     test_esp2013_calculation()
#     print("All tests passed!")