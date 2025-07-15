# It fetches mortality via Eurostat API for IT and ES (2015-2019), with robust parsing,
# fetches country-specific weather via Open-Meteo, merges, and prints two static 3x3 Matplotlib grids with LOWESS trends.
# Fixed filtering: Now filters to sex == 'T' (Total) to remove M/F duplicates and eliminate the "3 bands" issue.


# made by Orwell2024
# https://x.com/i/grok/share/XBe56iIT7rRNBhGziNct6dSvX

from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap
#from IPython.print import print
import requests
from io import StringIO
import gzip
import streamlit as st

def get_weather_info(country):
    """Fetches daily max temperature via Open-Meteo for a country and aggregates to weekly true maximum."""
    coords = {
        'NL': {'lat': 52.1445592, 'lon': 5.17377733378846},  # De Bilt
        
        'IT': {'lat': 41.9028, 'lon': 12.4964},  # Rome
        'ES': {'lat': 40.4168, 'lon': -3.7038}   # Madrid
    }
    if country not in coords:
        raise ValueError("Unsupported country for weather.")
    start_date = "2015-01-01"
    end_date = "2019-12-31"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords[country]['lat']}&longitude={coords[country]['lon']}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max&timezone=Europe/Berlin"

    print(f"DEBUG: Fetching weather from {url}")
    response = requests.get(url)
    print(f"DEBUG: Weather response status: {response.status_code}")

    if response.status_code != 200:
        print(f"DEBUG: Weather response content (first 200 chars): {response.content[:200]}")
        return pd.DataFrame()

    data = response.json()
    df_daily = pd.DataFrame({
        'date': pd.to_datetime(data['daily']['time']),
        'temp_max': data['daily']['temperature_2m_max']
    })

    df_daily["week_number"] = df_daily["date"].dt.isocalendar().week
    df_daily["year_number"] = df_daily["date"].dt.isocalendar().year

    result = (
        df_daily.groupby(["week_number", "year_number"])
        .agg({"temp_max": "max"})
        .reset_index()
    )
    result = result[["year_number", "week_number", "temp_max"]]
    print(f"DEBUG: Aggregated {len(result)} weeks of weather data for {country}")
    return result
def read_eurostats(url):
    print(f"DEBUG: Fetching data from URL: {url}")
    response = requests.get(url)
    print(f"DEBUG: Response status code: {response.status_code}")
    print(f"DEBUG: Response headers: {response.headers}")

    if response.status_code != 200:
        print(f"DEBUG: Response content (first 200 chars): {response.content[:200]}")
        return pd.DataFrame()

    print("DEBUG: Decompressing gzipped response...")
    try:
        cleaned_text = gzip.decompress(response.content).decode('utf-8')
    except Exception as e:
        print(f"DEBUG: Decompression error: {e}")
        cleaned_text = response.text  # Fallback

    print("DEBUG: First 5 lines of cleaned response text:")
    print('\n'.join(cleaned_text.splitlines()[:5]))

    df_ = pd.read_csv(
        StringIO(cleaned_text), 
        delimiter=",", 
        on_bad_lines='skip', 
        engine='python', 
        quoting=3, 
        header=0 
    )

    print("DEBUG: Columns after parsing:", df_.columns.tolist())
    for col in df_.columns:
        if ':' in col:
            new_col = col.split(':')[0].strip()
            df_ = df_.rename(columns={col: new_col})
   

    for col in ['freq', 'age', 'sex', 'unit', 'geo']:
        df_[[col, f'{col}_desc']] = df_[col].str.split(': ', n=1, expand=True)
        

   
    print("DEBUG: Columns after renaming:", df_.columns.tolist())

    
    return df_

def get_sterfte(country):
    """Fetches weekly mortality data via Eurostat API for a country with robust parsing and debug output."""
    base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_r_mwk_05/1.0/*.*.*.*.*?c[freq]=W&c[age]=TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,Y80-84,Y85-89,Y_GE90,UNK&c[sex]=T,M,F&c[unit]=NR"
    url = f"{base_url}&c[geo]={country}&c[TIME_PERIOD]=ge:2015-W01&le:2019-W53&compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=both"
    df_= read_eurostats(url)
    time_col = next((col for col in df_.columns if 'TIME_PERIOD' in col.upper()), None)
    if time_col:
        df_["year_number"] = (df_[time_col].str[:4]).astype(int, errors='ignore')
        df_["week_number"] = (df_[time_col].str[6:]).astype(int, errors='ignore')
        print("DEBUG: Extracted year_number and week_number from", time_col)
    else:
        print("Error: No 'TIME_PERIOD' column found after parsing.")
        print("DEBUG: Sample data (first row):", df_.iloc[0].to_dict())
        return pd.DataFrame()
    df_["OBS_VALUE"] = pd.to_numeric(df_["OBS_VALUE"], errors='coerce').fillna(0)

    # Clean 'age' column by taking the code before ':' (e.g., 'Y55-59: From 55 to 59 years' -> 'Y55-59')
    if 'age' in df_.columns:
        df_['age'] = df_['age'].str.split(':').str[0].str.strip()

    df_ = df_[["year_number", "week_number", "OBS_VALUE", "age", "sex"]]
    df_ = df_.groupby(["year_number", "week_number", "age", "sex"], as_index=False)["OBS_VALUE"].sum()
    df_["age_sex"] = df_["age"].astype(str) + "_" + df_["sex"]
    df_pop = get_population(country)
    
    df_merged = df_.merge(df_pop, on=["year_number", "age_sex"], how="left")
    df_merged["death_rate"] = (df_merged["OBS_VALUE"] / df_merged["population"]) * 100_000
    
    return df_merged
def groupby_agegroups(df_bevolking):
    # Define age binning
    bins = [-1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
            50, 55, 60, 65, 70, 75, 80, 85, 90, 120]
    labels = [
        "Y_LT5", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29", "Y30-34",
        "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59", "Y60-64",
        "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89", "Y_GE90"
    ]
    # Assign Eurostat group
        # Step 1: Extract rows where age == "TOTAL"
    df_bevolking =     df_bevolking[df_bevolking["age"] != "UNK"]
    df_bevolking["age"] = df_bevolking["age"].replace({"Y_LT1": "0", "Y_OPEN": "100"})
    df_total = df_bevolking[df_bevolking["age"] == "TOTAL"].copy()
    df_total["age_group"]="TOTAL"
    # Step 2: Remove those rows from the original DataFrame
    df_bevolking = df_bevolking[df_bevolking["age"] != "TOTAL"]

    df_bevolking["age"] = df_bevolking["age"].str.replace("Y", "").astype(int)
    

    df_bevolking["age_group"] = pd.cut(df_bevolking["age"], bins=bins, labels=labels, right=False)
    
    # Step 4: Add the TOTAL rows back to the end
    df_bevolking = pd.concat([df_bevolking, df_total], ignore_index=True)

  
    df_bevolking["age_sex"] = df_bevolking["age_group"].astype(str) + "_" + df_bevolking["sex"]
    # Sum population by Eurostat group, sex, and year
    df_bevolking_grouped = (
        df_bevolking
        .groupby(["year_number",  "age_sex"], as_index=False)
        .agg({"OBS_VALUE": "sum"})
    )

   
    return df_bevolking_grouped
def get_population(country):
    """Fetches yearly population data from Eurostat for a country."""
    #url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0/*.*.*.*?c[freq]=A&c[age]=TOTAL,Y_LT5,Y5-9,...,Y85-89,Y_GE90&c[sex]=T&c[geo]={country}&c[TIME_PERIOD]=ge:2015&le:2019&format=csvdata"
    base_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0/*.*.*.*.*?c[freq]=A&c[age]=TOTAL,Y_LT5,Y5-9,Y10-14,Y15-19,Y20-24,Y25-29,Y30-34,Y35-39,Y40-44,Y45-49,Y50-54,Y55-59,Y60-64,Y65-69,Y70-74,Y75-79,Y80-84,Y85-89,Y_GE90,UNK&c[sex]=T,M,F&c[unit]=NR"
    url = f"{base_url}&c[geo]={country}&c[TIME_PERIOD]=ge:2015-W01&le:2019-W53&compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=both"
    #https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0/*.*.*.*.*?c[freq]=A&c[unit]=NR&c[age]=TOTAL&c[sex]=T&c[geo]=NL&c[TIME_PERIOD]=2015,2016,2017,2018,2019,2020,2021,2022,2023,2024&compress=true
    url=f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/3.0/data/dataflow/ESTAT/demo_pjan/1.0/*.*.*.*.*?c[freq]=A&c[unit]=NR&c[age]=TOTAL,Y_LT1,Y1,Y2,Y3,Y4,Y5,Y6,Y7,Y8,Y9,Y10,Y11,Y12,Y13,Y14,Y15,Y16,Y17,Y18,Y19,Y20,Y21,Y22,Y23,Y24,Y25,Y26,Y27,Y28,Y29,Y30,Y31,Y32,Y33,Y34,Y35,Y36,Y37,Y38,Y39,Y40,Y41,Y42,Y43,Y44,Y45,Y46,Y47,Y48,Y49,Y50,Y51,Y52,Y53,Y54,Y55,Y56,Y57,Y58,Y59,Y60,Y61,Y62,Y63,Y64,Y65,Y66,Y67,Y68,Y69,Y70,Y71,Y72,Y73,Y74,Y75,Y76,Y77,Y78,Y79,Y80,Y81,Y82,Y83,Y84,Y85,Y86,Y87,Y88,Y89,Y90,Y91,Y92,Y93,Y94,Y95,Y96,Y97,Y98,Y99,Y_OPEN,UNK&c[sex]=T,M,F&c[geo]={country}&c[TIME_PERIOD]=2024,2023,2022,2021,2020,2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000&compress=true&format=csvdata&formatVersion=2.0&lang=en&labels=both"

    
    df_pop = read_eurostats(url)
    
    
    df_pop["year_number"] = df_pop["TIME_PERIOD"]
    df_pop_grouped = groupby_agegroups(df_pop)
    # df_pop["age"] = df_pop["age"].str.split(":").str[0].str.strip()
    # df_pop["age_sex"] = df_pop["age_group"].astype(str) + "_" + df_pop["sex"]
    # df_pop = df_pop[["year_number", "age_sex", "OBS_VALUE"]].rename(columns={"OBS_VALUE": "population"})
  
    df_pop_grouped = df_pop_grouped.rename(columns={"OBS_VALUE": "population"})
    return df_pop_grouped


# Function to create and save 3x3 grid for a country
def create_grid(df, country, what):
    """_summary_

    Args:
        df (_type_): _description_
        country (_type_): _description_
        what (_type_): "OBS_VALUE" |  "per_100k"
    """    
    if df.empty:
        print(f"No data for {country} - skipping grid.")
        return
    # Filter to sex == 'T' (Total) to remove M/F duplicates and fix "3 bands"
    
    df = df[df['sex'] == 'T'].copy()
    print(f"DEBUG: Data after filtering to sex='T': {len(df)} rows")

    # Define 9 age groups (cleaned codes)
    age_groups = [
        'Y55-59', 'Y60-64', 'Y65-69',
        'Y70-74', 'Y75-79', 'Y80-84',
        'Y85-89', 'Y_GE90', 'TOTAL'
    ]
    age_titles = [
        'Age 55-59', 'Age 60-64', 'Age 65-69',
        'Age 70-74', 'Age 75-79', 'Age 80-84',
        'Age 85-89', 'Age 90+', 'All Ages (Total)'
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=False)
    if what =="OBS_VALUE":
        fig.suptitle(f'Weekly Mortality vs. Maximum Temperature by Age Group (2015-2019, {country})', fontsize=20, y=0.98)
    else:
        fig.suptitle(f'Weekly Mortality per 100k vs. Maximum Temperature by Age Group (2015-2019, {country})', fontsize=20, y=0.98)
    
    fig.text(0.5, 0.02, 'Each dot represents a week, colored by year (light blue = 2015, dark blue = 2019).\nRed line shows smoothed trend (LOWESS). Data from Eurostat (deaths) & Open-Meteo (weather).', 
            ha='center', va='bottom', fontsize=10)

    # Blue colormap for years
    blue_cmap = LinearSegmentedColormap.from_list("blues", ["lightblue", "darkblue"])

    for i, age_code in enumerate(age_groups):
        df_age = df[df['age'] == age_code]
        
        if df_age.empty:
            print(f"DEBUG: No data for age {age_code} in {country}")
            continue
        
        # Calculate row and col
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        # Normalize years for color mapping
        years_norm = (df_age['year_number'] - df_age['year_number'].min()) / (df_age['year_number'].max() - df_age['year_number'].min())
        
        # Scatter points with blue gradient by year
        ax.scatter(
            df_age['temp_max'], df_age[what],
            c=years_norm, cmap=blue_cmap, alpha=0.7, s=20
        )
        
        # Improved LOWESS trend
        lowess = sm.nonparametric.lowess(df_age[what], df_age['temp_max'], frac=0.4)
        ax.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=4, alpha=0.5)
        
        ax.set_title(age_titles[i], fontsize=14)
        ax.set_xlabel('Highest Weekly Temperature (°C)' if row == 2 else '', fontsize=12)
        if what =="OBS_VALUE":
            ax.set_ylabel('Deaths per Week' if col == 0 else '', fontsize=12)
        else:
            ax.set_ylabel('Deathrate [per 100k] per Week' if col == 0 else '', fontsize=12)
        ax.grid(True, color='lightgray')
        ax.set_xlim(-5, 45)  # Extended to 45°C as requested

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    #plt.savefig(f'mortality_grid_{country}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    st.pyplot(fig)
    print(f"DEBUG: 3x3 grid for {country} saved as 'mortality_grid_{country}.png'")
    # --- Main execution block (run this in a Jupyter cell) ---

def main_orwell():
    print("Fetching and plotting mortality data with local weather (2015-2019)...")
    # Fetch data for IT and ES
    what = st.selectbox("what to show [OBS_VALUE | per_100k]", ["OBS_VALUE",  "per_100k"],0)
        
    for country in ["NL", "IT", "ES"]:
        df_sterfte_country = get_sterfte(country)
        df_weather_country = get_weather_info(country)
        df_country = df_sterfte_country.merge(df_weather_country, on=["year_number", "week_number"])
        df_country = df_country[(df_country["year_number"] >= 2015) & (df_country["year_number"] <= 2019)]
        print(f"\nSample Merged Data for {country} (2015-2019):")
        print(df_country.head())
        # Create and save 3x3 grids
        st.subheader(f"=== {country} ===")
        create_grid(df_country,country, what)

        st.info("Code based on https://x.com/orwell2022/status/1944806639388778852")
        
if __name__ == "__main__":
    main_orwell()