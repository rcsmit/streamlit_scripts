import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import streamlit as st

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data


def normaal_verdeeld(df,what_to_show):
    df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year

    # Resultaatlijst
    results = []

    # Voor elke dag van het jaar
    for day, group in df.groupby('day_of_year'):
        temps = group[what_to_show].dropna()
        
        # Alleen testen als er genoeg data is (min. 3 punten vereist voor Shapiro)
        if len(temps) >= 3:
            stat, p = shapiro(temps)
            results.append({
                'day_of_year': day,
                'n': len(temps),
                'statistic': stat,
                'p_value': p,
                'normal': p > 0.05
            })

    # Naar dataframe
    normality_df = pd.DataFrame(results)

    # Sorteer op dagnummer
    normality_df = normality_df.sort_values('day_of_year')


    # Maak een kolom met 1 = niet normaal, 0 = normaal
    normality_df['not_normal'] = (~normality_df['normal']).astype(int)

    # Zet om naar array van 53 weken x 7 dagen
    heatmap_array = np.full((53, 7), np.nan)

    for _, row in normality_df.iterrows():
        day = int(row['day_of_year']) - 1
        week = day // 7
        weekday = day % 7
        if week < 53:
            heatmap_array[week, weekday] = row['not_normal']

    # Plot de heatmap
    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_array, cmap='Reds', cbar=False, linewidths=0.5, linecolor='gray')

    plt.title(f'Niet-normale verdeling ({what_to_show}) per dag van het jaar')
    plt.xlabel('Weekdag (0 = maandag)')
    plt.ylabel('Weeknummer')
    plt.yticks(rotation=0)
    plt.xticks(ticks=np.arange(7)+0.5, labels=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'])

    plt.tight_layout()
    st.pyplot (fig)

    st.info("Rood = niet normaal verdeeld | Wit = normaal verdeeld of geen data |Indeling is op weeknummer × weekdag")


def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result.csv" 
    df = get_data(url)

    normaal_verdeeld(df,"temp_max")
if __name__ == "__main__":
    main()