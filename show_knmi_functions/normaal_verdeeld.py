import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm, gamma, kstest
import streamlit as st
import sys
from tqdm import tqdm
import numpy as np
from scipy import stats
from joblib import Parallel, delayed
from scipy.stats import shapiro, gamma, kstest

# ignore    RuntimeWarning: invalid value encountered in sqrt
#            sk = 2*(b-a)*np.sqrt(a + b + 1) / (a + b + 2) / np.sqrt(a*b)
import warnings
warnings.simplefilter("ignore", category=Warning)

try:
    from show_knmi_functions.utils import get_data
except:
    from utils import get_data
def to_scalar(x):
    return float(x.item()) if isinstance(x, np.ndarray) else float(x)

def test_distributions(data):
    
    # Your data here
    data = np.array(data)

    # List of distributions to test
    distributions = [
        "norm", "expon", "gamma", "lognorm", "beta", "weibull_min", "weibull_max",
        "pareto", "uniform", "triang", "logistic", "t", "f", "rayleigh"
    ]

    results = []

    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        try:
            params = dist.fit(data)
            D, p = stats.kstest(data, dist_name, args=params)
            fit_ok = p > 0.05
            results.append((dist_name, p, D, fit_ok, params))
        except Exception as e:
            results.append((dist_name, np.nan, np.nan, False, str(e)))

    # Create dataframe
    df_results = pd.DataFrame(results, columns=["distribution", "p_value", "D_statistic", "fit_ok", "parameters"])
    df_results = df_results.sort_values("p_value", ascending=False).reset_index(drop=True)
    return df_results 

@st.cache_data() 
def get_results(df,what_to_show):
    results = []
    

    def process_day(day, temps):
        if len(temps) < 3:
            return None

        stat, p_norm = shapiro(temps)
        is_norm = p_norm > 0.05

        shape, loc, scale = gamma.fit(temps)
        _, p_gamma = kstest(temps, 'gamma', args=(shape, loc, scale))
        is_gamma = p_gamma > 0.05

        return {
            'day_of_year': day,
            'n': len(temps),
            'p_value': p_norm,
            'normal': is_norm,
            'p_gamma': p_gamma,
            'gamma_normal': is_gamma
        }

    df_clean = df.dropna(subset=[what_to_show])
    grouped = df_clean.groupby('day_of_year')[what_to_show]

        
    # Run with progress bar
    results = Parallel(n_jobs=-1)(
        delayed(process_day)(day, temps)
        for day, temps in tqdm(grouped, desc="Processing days")
    )

    # Remove None results
    results = [r for r in results if r is not None]
    return results

def normaal_verdeeld(df, what_to_show):
    st.write("### Normale en gamma verdeling per dag van het jaar")
    st.write("Deze pagina test of de temperatuur per dag van het jaar normaal or gamma verdeeld is.")
    
   
    results = get_results(df,what_to_show)

    # Resultaten naar DataFrame
    normality_df = make_normality_df(results)

    # Arrays voor heatmaps
    heatmap_normal = np.full((53, 7), np.nan)
    heatmap_gamma = np.full((53, 7), np.nan)
    # Collect all test_distributions
    all_distributions = []

    for _, row in normality_df.iterrows():
        day = int(row['day_of_year']) - 1
        week = day // 7
        weekday = day % 7
        # weekday = row['weekday']
        if week < 53:
            heatmap_normal[week, weekday] = row['not_normal']
            heatmap_gamma[week, weekday] = row['not_gamma']

    # Streamlit layout met 2 kolommen
    col1, col2 = st.columns(2)

    # Plot normaal heatmap
    with col1:
        plot_heatmap(what_to_show, heatmap_normal, 'Reds', 'Niet-normale verdeling')

    # Plot gamma heatmap
    with col2:
        plot_heatmap(what_to_show, heatmap_gamma, 'Purples', "Niet-gamma verdeling")

        
    # Histogrammen (optioneel: alleen voor elke 7e dag)
    for _, row in normality_df.iterrows():
        day = int(row['day_of_year'])
        weekday = day % 7
        #weekday = row['weekday']
        #if 1==1:
        if weekday == 0:
            label = row['dag_maand']
            is_normal = "ja" if row['normal'] else "nee"
            is_gamma = "ja" if row['gamma_normal'] else "nee"
            temps = df[df['day_of_year'] == day][what_to_show].dropna()
            if len(temps) < 3:
                continue

            
            make_plot(row, label, is_normal, is_gamma, temps)

            df_test_distribution = test_distributions(temps)
                
            # with st.expander("Data"):
                # st.write(df_test_distribution)
            df_test_distribution['day_of_year'] = day
            all_distributions.append(df_test_distribution)
            text= f"week {day}\r"
            sys.stdout.write(text)
            # print (text)
            sys.stdout.flush()

        sys.stdout.write("Klaar")
            #st.markdown("---")
            

    # Combine into one big DataFrame
    df_all = pd.concat(all_distributions, ignore_index=True)

    # Count number of times each distribution had fit_ok == True
    df_summary = (
        df_all[df_all['fit_ok'] == True]
        .groupby('distribution')
        .size()
        .reset_index(name='count_fit_ok')
        .sort_values('count_fit_ok', ascending=False)
    )

    # Show in Streamlit
    st.subheader("Aantal dagen dat distributies een goede fit hadden")
    st.dataframe(df_summary)

def make_normality_df(results):
    normality_df = pd.DataFrame(results)
    normality_df = normality_df.sort_values('day_of_year')
    normality_df['date'] = pd.to_datetime(normality_df['day_of_year'], format='%j')
    normality_df['dag_maand'] = normality_df['date'].dt.strftime('%d-%m')
    normality_df['not_normal'] = (~normality_df['normal']).astype(int)
    normality_df['not_gamma'] = (~normality_df['gamma_normal']).astype(int)
    return normality_df

def plot_heatmap(what_to_show, heatmap_normal, color,title):
    fig1 = plt.figure()
    sns.heatmap(heatmap_normal, cmap=color, cbar=False, linewidths=0.5, linecolor='gray')
    plt.title(f'{title} ({what_to_show})')
    plt.xlabel('Weekdag')
    plt.ylabel('Weeknummer')
    plt.yticks(rotation=0)
    plt.xticks(ticks=np.arange(7)+0.5, labels=['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'])
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close()

def make_plot(row, label, is_normal, is_gamma, temps):
    x, mu, std, p, shape, loc, scale, p25_gamma, p25_norm, p975_gamma, p975_norm, gamma_y = calculate_distribution_lines(temps)

    draw_plot(label, is_normal, is_gamma, temps, x, p, p25_gamma, p25_norm, p975_gamma, p975_norm, gamma_y)

    write_parameters(row, label, is_normal, is_gamma, temps, mu, std, shape, loc, scale, p25_gamma, p25_norm, p975_gamma, p975_norm)

def write_parameters(row, label, is_normal, is_gamma, temps, mu, std, shape, loc, scale, p25_gamma, p25_norm, p975_gamma, p975_norm):
    st.write(f"📅 Dag: {label}  |  Waarnemingen: {len(temps)}")

    try:
        st.write(f"Normaal:  p={to_scalar(row['p_value']):.3f} → {is_normal} | μ={mu:.2f}, σ={std:.2f} | p25={p25_norm:.2f} | p975={p975_norm:.2f}")
        st.write(f"Gamma:    p={to_scalar(row['p_gamma']):.3f} → {is_gamma} | shape={shape:.2f}, loc={loc:.2f}, scale={scale:.2f} | p25={p25_gamma:.2f} | p975={p975_gamma:.2f}")
    except:
                # gives an error on streamlit sharing
        pass

def calculate_distribution_lines(temps):
    #xmin, xmax = plt.xlim()

    xmin, xmax=min(temps), max(temps)
 
    x = np.linspace(xmin, xmax, 100)
    mu, std = norm.fit(temps)
    p = norm.pdf(x, mu, std)

    shape, loc, scale = gamma.fit(temps)

    p25_gamma = gamma.ppf(0.025, shape, loc=loc, scale=scale)
    p25_norm = norm.ppf(0.025, loc=mu, scale=std)


    p975_gamma = gamma.ppf(0.975, shape, loc=loc, scale=scale)
    p975_norm = norm.ppf(0.975, loc=mu, scale=std)
    gamma_y = gamma.pdf(x, shape, loc, scale)
    return x,mu,std,p,shape,loc,scale,p25_gamma,p25_norm,p975_gamma,p975_norm,gamma_y

def draw_plot(label, is_normal, is_gamma, temps, x, p, p25_gamma, p25_norm, p975_gamma, p975_norm, gamma_y):
    fig = plt.figure(figsize=(6, 4))
    plt.hist(temps, bins=15, density=True, alpha=0.6, label='Waarnemingen')

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
          
def main():
    st.title("Test op normale en gamma-verdeling per dag")
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/show_knmi_functions/result_1901.csv"
    df = get_data(url)
    normaal_verdeeld(df, "temp_max")


if __name__ == "__main__":
    main()
