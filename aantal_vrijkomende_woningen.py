import pandas as pd
import streamlit as st
import plotly.express as px

# ------------------------------------------------------------
# Kernfuncties
# ------------------------------------------------------------
def constante_geboortes(population_data, startjaar=2024, jaren=40):
    """Constante geboortes per jaar gelijk aan aantal 1-jarigen in 2024."""
    pop_2024 = population_data[population_data['jaar'] == 2024]
    geboortes_const = int(pop_2024.loc[pop_2024['leeftijd'] == 1, 'aantal'].sum())
    return {startjaar + i: geboortes_const for i in range(jaren + 1)}

def simuleer_overleving_70_120(population_data, mort_f, mort_m, jaren=15):
    """
    Simuleer 70..120 cohort vanaf 2024 met jaar-specifieke sterftekansen uit AG2024.
    Retourneert df met per leeftijd en geslacht: start, survivors, deaths, en per jaar deaths.
    """
    # Startcohort 2024
    pop24 = population_data[population_data['jaar'] == 2024].copy()
    cohort_f = pop24[(pop24['geslacht'] == 'F') & (pop24['leeftijd'].between(70, 120))][['leeftijd', 'aantal']].copy()
    cohort_m = pop24[(pop24['geslacht'] == 'M') & (pop24['leeftijd'].between(70, 120))][['leeftijd', 'aantal']].copy()
    cohort_f['geslacht'] = 'F'
    cohort_m['geslacht'] = 'M'
    st.write(mort_f)
    st.write(mort_m)
    
    
    # Zorg dat age integer is
    mort_f = mort_f.copy()
    mort_m = mort_m.copy()
    mort_f['age'] = mort_f['age'].astype(int)
    mort_m['age'] = mort_m['age'].astype(int)

    # We simuleren van 2025 t m 2024+jaren
    years = [str(y) for y in range(2025, 2024 + jaren + 1)]

    # Voeg helpers toe
    for dfc in (cohort_f, cohort_m):
        dfc['aantal_start'] = dfc['aantal']
        dfc['aantal_survivors'] = dfc['aantal']
        for y in years:
            dfc[f'deaths_{y}'] = 0.0

    # Stap per jaar
    for y in years:
        # verouder 1 jaar
        cohort_f['leeftijd'] += 1
        cohort_m['leeftijd'] += 1

        # merge met sterftekans q_x(year)
        f = cohort_f.merge(mort_f[['age', y]], left_on='leeftijd', right_on='age', how='left')
        m = cohort_m.merge(mort_m[['age', y]], left_on='leeftijd', right_on='age', how='left')

        f[y] = f[y].fillna(0.0)
        m[y] = m[y].fillna(0.0)

        # bereken deaths en survivors voor dit jaar
        f_deaths = f['aantal_survivors'] * f[y]
        m_deaths = m['aantal_survivors'] * m[y]

        # schrijf terug
        cohort_f = f.drop(columns=['age'])
        cohort_m = m.drop(columns=['age'])

        cohort_f[f'deaths_{y}'] = f_deaths
        cohort_m[f'deaths_{y}'] = m_deaths

        cohort_f['aantal_survivors'] = cohort_f['aantal_survivors'] - f_deaths
        cohort_m['aantal_survivors'] = cohort_m['aantal_survivors'] - m_deaths

        # kolom met kans weg
        cohort_f = cohort_f.drop(columns=[y])
        cohort_m = cohort_m.drop(columns=[y])

    # eindresultaat
    cohort = pd.concat([cohort_f, cohort_m], ignore_index=True)
    death_cols = [c for c in cohort.columns if c.startswith('deaths_')]
    cohort['deaths_15y'] = cohort[death_cols].sum(axis=1)

    # samenvatting per geslacht
    summary = cohort.groupby('geslacht', as_index=False).agg(
        start=('aantal_start', 'sum'),
        survivors=('aantal_survivors', 'sum'),
        deaths_15y=('deaths_15y', 'sum')
    )
    summary['survival_perc'] = summary['survivors'] / summary['start'] * 100

    total = {
        'start': summary['start'].sum(),
        'survivors': summary['survivors'].sum(),
        'deaths_15y': summary['deaths_15y'].sum()
    }

    return cohort, summary, total

def woningen_vrij_van_deaths(total_deaths_15y, woningfactor):
    """Zet overlijdens om naar woningen vrij met woningfactor."""
    return total_deaths_15y * woningfactor

# ------------------------------------------------------------
# Streamlit app
# ------------------------------------------------------------
def main():
    st.title("Woningvrijval 70–120 binnen 15 jaar met AG2024-sterftekansen")

    # Data laden
    population_data = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/bevolking_leeftijd_NL.csv",
        sep=';'
    )
    mort_f = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_female.csv"
    )
    mort_m = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_male.csv"
    )

    # UI
    colA, colB = st.columns(2)
    with colA:
        jaren = st.slider("Aantal jaren vooruit", 5, 30, 15, 1)
    with colB:
        woningfactor = st.slider("Woningen per overlijden", 0.20, 1.00, 0.60, 0.05)

    # Simulatie
    with st.spinner("Simuleren met jaar-specifieke sterftekansen"):
        cohort, summary, total = simuleer_overleving_70_120(population_data, mort_f, mort_m, jaren=jaren)

    # Kerncijfers
    st.subheader("Kerncijfers cohort 70–120 in 2024")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Start cohort", f"{int(total['start']):,}".replace(",", "."))
    with k2:
        st.metric(f"Overlijdens binnen {jaren} jaar", f"{int(total['deaths_15y']):,}".replace(",", "."))
    with k3:
        st.metric("Survivors eindjaar", f"{int(total['survivors']):,}".replace(",", "."))
    with k4:
        st.metric("Woningen vrij", f"{int(round(woningen_vrij_van_deaths(total['deaths_15y'], woningfactor))):,}".replace(",", "."))

    st.caption(f"**Woningfactor** {woningfactor:.2f} per overlijden")

    # Verdeling naar geslacht
    st.subheader("Samenvatting per geslacht")
    st.dataframe(summary)

    # Visualisatie overlijdens per leeftijd
    st.subheader(f"Overlijdens binnen {jaren} jaar per leeftijd en geslacht")
    plotdf = cohort[['leeftijd', 'geslacht', 'deaths_15y']].groupby(['leeftijd', 'geslacht'], as_index=False).sum()
    fig = px.bar(plotdf, x='leeftijd', y='deaths_15y', color='geslacht',
                 labels={'leeftijd': 'Leeftijd in eindjaar', 'deaths_15y': 'Overlijdens binnen periode'},
                 title='Cumulatieve overlijdens binnen de periode')
    st.plotly_chart(fig, use_container_width=True)

    # Constante geboortes
    st.subheader("Constante geboortes op basis van 1-jarigen in 2024")
    years_birth = st.number_input("Toon jaren", 5, 100, 40)
    geboortedict = constante_geboortes(population_data, startjaar=2024, jaren=int(years_birth))
    st.metric("Geboortes per jaar", f"{list(geboortedict.values())[0]:,}".replace(",", "."))
    with st.expander("Reeks per jaar"):
        st.json(geboortedict, expanded=False)

    st.divider()
    st.write("Bronnen  CBS bevolking per leeftijd 2024  AG2024 prognosetafel")

if __name__ == "__main__":
    main()
