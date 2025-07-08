import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm, gamma, kstest
import streamlit as st
import sys

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data


def normaal_verdeeld(df, what_to_show):
    df['date'] = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
    df['day_of_year'] = df['date'].dt.dayofyear
    df['year'] = df['date'].dt.year

    results = []

    # Loop over elke dag van het jaar
    for day, group in df.groupby('day_of_year'):
        sys.stdout.write(f"Processing day {day}...\r")
        sys.stdout.flush()
        temps = group[what_to_show].dropna()
        if len(temps) >= 3:
            # Normale verdeling
            stat, p_norm = shapiro(temps)
            is_norm = p_norm > 0.05

            # Gamma verdeling
            shape, loc, scale = gamma.fit(temps)
            _, p_gamma = kstest(temps, 'gamma', args=(shape, loc, scale))
            is_gamma = p_gamma > 0.05

            results.append({
                'day_of_year': day,
                'n': len(temps),
                'p_value': p_norm,
                'normal': is_norm,
                'p_gamma': p_gamma,
                'gamma_normal': is_gamma
            })

    # Resultaten naar DataFrame
    normality_df = pd.DataFrame(results)
    normality_df = normality_df.sort_values('day_of_year')
    normality_df['date'] = pd.to_datetime(normality_df['day_of_year'], format='%j')
    normality_df['dag_maand'] = normality_df['date'].dt.strftime('%d-%m')
    normality_df['not_normal'] = (~normality_df['normal']).astype(int)
    normality_df['not_gamma'] = (~normality_df['gamma_normal']).astype(int)

    # Arrays voor heatmaps
    heatmap_normal = np.full((53, 7), np.nan)
    heatmap_gamma = np.full((53, 7), np.nan)

    for _, row in normality_df.iterrows():
        day = int(row['day_of_year']) - 1
        week = day // 7
        weekday = day % 7
        if week < 53:
            heatmap_normal[week, weekday] = row['not_normal']
            heatmap_gamma[week, weekday] = row['not_gamma']

    # Streamlit layout met 2 kolommen
    col1, col2 = st.columns(2)

    # Plot normaal heatmap
    with col1:
        fig1 = plt.figure()
        sns.heatmap(heatmap_normal, cmap='Reds', cbar=False, linewidths=0.5, linecolor='gray')
        plt.title(f'Niet-normale verdeling ({what_to_show})')
        plt.xlabel('Weekdag (0 = maandag)')
        plt.ylabel('Weeknummer')
        plt.yticks(rotation=0)
        plt.xticks(ticks=np.arange(7)+0.5, labels=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'])
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    # Plot gamma heatmap
    with col2:
        fig2 = plt.figure()
        sns.heatmap(heatmap_gamma, cmap='Purples', cbar=False, linewidths=0.5, linecolor='gray')
        plt.title(f'Niet-gamma verdeling ({what_to_show})')
        plt.xlabel('Weekdag (0 = maandag)')
        plt.ylabel('Weeknummer')
        plt.yticks(rotation=0)
        plt.xticks(ticks=np.arange(7)+0.5, labels=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'])
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Histogrammen (optioneel: alleen voor maandag)
    for _, row in normality_df.iterrows():
        day = int(row['day_of_year'])
        weekday = day % 7
        if weekday == 1:
            label = row['dag_maand']
            is_normal = "ja" if row['normal'] else "nee"
            is_gamma = "ja" if row['gamma_normal'] else "nee"
            temps = df[df['day_of_year'] == day][what_to_show].dropna()
            if len(temps) < 3:
                continue

            fig = plt.figure(figsize=(6, 4))
            plt.hist(temps, bins=15, density=True, alpha=0.6, label='Waarnemingen')

            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            mu, std = norm.fit(temps)
            p = norm.pdf(x, mu, std)

            shape, loc, scale = gamma.fit(temps)

            p25_gamma = gamma.ppf(0.025, shape, loc=loc, scale=scale)
            p25_norm = norm.ppf(0.025, loc=mu, scale=std)


            p975_gamma = gamma.ppf(0.975, shape, loc=loc, scale=scale)
            p975_norm = norm.ppf(0.975, loc=mu, scale=std)
            gamma_y = gamma.pdf(x, shape, loc, scale)

            plt.plot(x, p, 'k-', linewidth=2, label='Normale verdeling')
            plt.plot(x, gamma_y, 'r--', linewidth=2, label='Gamma verdeling')
            plt.axvline(p975_gamma, color='red', linestyle=':', linewidth=1.5, label='p975 gamma')
            plt.axvline(p975_norm, color='black', linestyle=':', linewidth=1.5, label='p975 normaal')

            plt.axvline(p25_gamma, color='red', linestyle='--', linewidth=1.5, label='p25 gamma')
            plt.axvline(p25_norm, color='black', linestyle='--', linewidth=1.5, label='p25 normaal')


            plt.title(f'Dag {label}: normaal? {is_normal} | gamma? {is_gamma}')
            plt.xlabel('Temp max')
            plt.ylabel('Frequentie')
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.write(f"📅 Dag: {label}  |  Waarnemingen: {len(temps)}")
            st.write(f"Normaal:  p={row['p_value']:.3f} → {is_normal} | μ={mu:.2f}, σ={std:.2f} | p25={p25_norm:.2f} | p975={p975_norm:.2f}")
            st.write(f"Gamma:    p={row['p_gamma']:.3f} → {is_gamma} | shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f} | p25={p25_gamma:.2f} | p975={p975_gamma:.2f}")
            st.markdown("---")


def main():
    st.title("Test op normale en gamma-verdeling per dag")
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result_1901.csv"
    df = get_data(url)
    normaal_verdeeld(df, "temp_max")


if __name__ == "__main__":
    main()
