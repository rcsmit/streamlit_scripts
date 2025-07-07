import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro
import streamlit as st
from scipy.stats import norm
from scipy.stats import norm, gamma, kstest

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


    # Voeg dag-maand toe aan dataframe voor labels
    normality_df['date'] = pd.to_datetime(normality_df['day_of_year'], format='%j')
    normality_df['dag_maand'] = normality_df['date'].dt.strftime('%d-%m')

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
    fig = plt.figure()
    sns.heatmap(heatmap_array, cmap='Reds', cbar=False, linewidths=0.5, linecolor='gray')

    plt.title(f'Niet-normale verdeling ({what_to_show}) per dag van het jaar')
    plt.xlabel('Weekdag (0 = maandag)')
    plt.ylabel('Weeknummer')
    plt.yticks(rotation=0)
    plt.xticks(ticks=np.arange(7)+0.5, labels=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'])

    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

    # Histogram plots voor niet normaal verdeelde dagen
    # not_normal_days = normality_df[normality_df['not_normal'] == 1]
    days = normality_df

    #for _, row in not_normal_days.iterrows():
    for _, row in days.iterrows():
        day = int(row['day_of_year'])
        weekday = day % 7
        if weekday ==0:
            label = row['dag_maand']
            is_normal = "ja" if row['normal'] else "nee"

            temps = df[df['day_of_year'] == day][what_to_show].dropna()
            if len(temps) < 3:
                continue

            fig = plt.figure(figsize=(6, 4))
            plt.hist(temps, bins=15, density=True, alpha=0.6, label='Waarnemingen')

            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            mu, std = norm.fit(temps)
            p = norm.pdf(x, mu, std)
            
            

             # Gamma verdeling fit + test
            shape, loc, scale = gamma.fit(temps)
            gamma_y = gamma.pdf(x, shape, loc, scale)
            _, gamma_p = kstest(temps, 'gamma', args=(shape, loc, scale))
            if gamma_p < 0.05:
                is_gamma = "ja"
            
            
            plt.plot(x, p, 'k-', linewidth=2, label='Normale verdeling')
            plt.plot(x, gamma_y, 'r--', linewidth=2, label='Gamma verdeling')

            plt.title(f'Dag {label}: normaal verdeeld? {is_normal} Gamma verdeeld?{is_gamma}')
            plt.xlabel('Temp max')
            plt.ylabel('Frequentie')
            plt.legend()
            plt.tight_layout()
        
            st.pyplot(fig)



def main():
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result_1901.csv" 
    df = get_data(url)

    normaal_verdeeld(df,"temp_max")
if __name__ == "__main__":
    main()