import pandas as pd
import streamlit as st
import plotly.express as px

def calculate_85plus_in_2060(population_data, mortality_female, mortality_male):
    """
    Berekent het aantal 85-plussers in 2060 op basis van de 2024 bevolking
    met variërende sterftekansen per jaar
    """
    # Filter population data for the year 2024
    population_2024 = population_data[population_data['jaar'] == 2024]
    
    # Initialize surviving population
    surviving_population_female = population_2024[population_2024['geslacht'] == 'F'].copy()
    surviving_population_male = population_2024[population_2024['geslacht'] == 'M'].copy()
    
    # Add a column to track the initial age in 2024
    surviving_population_female['age_in_2024'] = surviving_population_female['leeftijd']
    surviving_population_male['age_in_2024'] = surviving_population_male['leeftijd']
    
    # Loop through each year from 2025 to 2060
    for year in range(2025, 2061):
        # Age the population by 1 year
        surviving_population_female['leeftijd'] += 1
        surviving_population_male['leeftijd'] += 1
        
        # Merge with mortality probabilities for the current year
        surviving_population_female = surviving_population_female.merge(
            mortality_female[['age', str(year)]], left_on='leeftijd', right_on='age', how='left'
        )
        surviving_population_male = surviving_population_male.merge(
            mortality_male[['age', str(year)]], left_on='leeftijd', right_on='age', how='left'
        )
        
        # Fill NaN values with 0 (for ages beyond the mortality table)
        surviving_population_female[str(year)] = surviving_population_female[str(year)].fillna(0)
        surviving_population_male[str(year)] = surviving_population_male[str(year)].fillna(0)
        
        # Apply survival probabilities
        surviving_population_female['aantal'] *= (1 - surviving_population_female[str(year)])
        surviving_population_male['aantal'] *= (1 - surviving_population_male[str(year)])
        
        # Drop unnecessary columns
        surviving_population_female = surviving_population_female.drop(columns=[str(year), 'age'])
        surviving_population_male = surviving_population_male.drop(columns=[str(year), 'age'])
    
    # Filter for 85+ in 2060
    female_85plus = surviving_population_female[surviving_population_female['leeftijd'] >= 85]['aantal'].sum()
    male_85plus = surviving_population_male[surviving_population_male['leeftijd'] >= 85]['aantal'].sum()
    
    return female_85plus, male_85plus, female_85plus + male_85plus


def calculate_85plus_in_2060_fixed_2025_rates(population_data, mortality_female, mortality_male):
    """
    Berekent het aantal 85-plussers in 2060 op basis van de 2024 bevolking
    met vaste sterftekansen van 2025 voor alle jaren
    """
    # Filter population data for the year 2024
    population_2024 = population_data[population_data['jaar'] == 2024]
    
    # Initialize surviving population
    surviving_population_female = population_2024[population_2024['geslacht'] == 'F'].copy()
    surviving_population_male = population_2024[population_2024['geslacht'] == 'M'].copy()
    
    # Add a column to track the initial age in 2024
    surviving_population_female['age_in_2024'] = surviving_population_female['leeftijd']
    surviving_population_male['age_in_2024'] = surviving_population_male['leeftijd']
    
    # Get mortality rates for 2025 (fixed rates)
    mortality_2025_female = mortality_female[['age', '2025']].copy()
    mortality_2025_male = mortality_male[['age', '2025']].copy()
    
    # Loop through each year from 2025 to 2060
    for year in range(2025, 2061):
        # Age the population by 1 year
        surviving_population_female['leeftijd'] += 1
        surviving_population_male['leeftijd'] += 1
        
        # Merge with mortality probabilities for 2025 (fixed rates)
        surviving_population_female = surviving_population_female.merge(
            mortality_2025_female, left_on='leeftijd', right_on='age', how='left'
        )
        surviving_population_male = surviving_population_male.merge(
            mortality_2025_male, left_on='leeftijd', right_on='age', how='left'
        )
        
        # Fill NaN values with 0 (for ages beyond the mortality table)
        surviving_population_female['2025'] = surviving_population_female['2025'].fillna(0)
        surviving_population_male['2025'] = surviving_population_male['2025'].fillna(0)
        
        # Apply survival probabilities using 2025 rates
        surviving_population_female['aantal'] *= (1 - surviving_population_female['2025'])
        surviving_population_male['aantal'] *= (1 - surviving_population_male['2025'])
        
        # Drop unnecessary columns
        surviving_population_female = surviving_population_female.drop(columns=['2025', 'age'])
        surviving_population_male = surviving_population_male.drop(columns=['2025', 'age'])
    
    # Filter for 85+ in 2060
    female_85plus = surviving_population_female[surviving_population_female['leeftijd'] >= 85]['aantal'].sum()
    male_85plus = surviving_population_male[surviving_population_male['leeftijd'] >= 85]['aantal'].sum()
    
    return female_85plus, male_85plus, female_85plus + male_85plus

def main():
    st.title("Nederlandse Bevolking Overlevingsanalyse")
    
    # Load the population data
    aantal_jaar = st.number_input("Aantal jaar", 1, 100, 40)
    population_data = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/bevolking_leeftijd_NL.csv", sep=';')

    # Load the mortality tables
    mortality_female = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_female.csv")
    mortality_male = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_male.csv")




    # Originele analyse
    st.header(f"Overlevingsanalyse voor {aantal_jaar} jaar")
    
    # Filter population data for the year 2024
    population_2024 = population_data[population_data['jaar'] == 2024]

    # Initialize surviving population
    surviving_population_female = population_2024[population_2024['geslacht'] == 'F'].copy()
    surviving_population_male = population_2024[population_2024['geslacht'] == 'M'].copy()

    # Add a column to track the initial age in 2024
    surviving_population_female['age_in_2024'] = surviving_population_female['leeftijd']
    surviving_population_male['age_in_2024'] = surviving_population_male['leeftijd']

    # Add a column to track the initial age in 2024
    surviving_population_female['aantal_in_2024'] = surviving_population_female['aantal']
    surviving_population_male['aantal_in_2024'] = surviving_population_male['aantal']

    # Loop through each year from 2025 to 2066
    for year in range(2025, 2025+aantal_jaar+1):
        # Age the population by 1 year
        surviving_population_female['leeftijd'] += 1
        surviving_population_male['leeftijd'] += 1

        # Merge with mortality probabilities for the current year
        surviving_population_female = surviving_population_female.merge(
            mortality_female[['age', str(year)]], left_on='leeftijd', right_on='age', how='left'
        )
        surviving_population_male = surviving_population_male.merge(
            mortality_male[['age', str(year)]], left_on='leeftijd', right_on='age', how='left'
        )

        # Fill NaN values with 0 (for ages beyond the mortality table)
        surviving_population_female[str(year)] = surviving_population_female[str(year)].fillna(0)
        surviving_population_male[str(year)] = surviving_population_male[str(year)].fillna(0)

        # Apply survival probabilities
        surviving_population_female['aantal'] *= (1 - surviving_population_female[str(year)])
        surviving_population_male['aantal'] *= (1 - surviving_population_male[str(year)])

        # Drop unnecessary columns
        surviving_population_female = surviving_population_female.drop(columns=[str(year), 'age'])
        surviving_population_male = surviving_population_male.drop(columns=[str(year), 'age'])

    # Calculate the total surviving population in target year
    total_population_target = surviving_population_female['aantal'].sum() + surviving_population_male['aantal'].sum()

    # Calculate the total population in 2024
    total_population_2024 = surviving_population_female['aantal_in_2024'].sum() + surviving_population_male['aantal_in_2024'].sum()

    # Calculate the percentage of people still alive in target year
    percentage_alive_target = (total_population_target / total_population_2024) * 100
    surviving_population_male['survival_perc'] = surviving_population_male['aantal'] / surviving_population_male['aantal_in_2024'] * 100
    surviving_population_female['survival_perc'] = surviving_population_female['aantal'] / surviving_population_female['aantal_in_2024'] * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("MALES")
        st.dataframe(surviving_population_male)
    with col2:
        st.write("FEMALES")
        st.dataframe(surviving_population_female)

    st.write(f"Total population in {2024+aantal_jaar} who were alive in 2024: {total_population_target:.0f}")
    st.write(f"Percentage of the 2024 population still alive in {2024+aantal_jaar}: {percentage_alive_target:.2f}%")

    # Combine male and female data for plotting
    surviving_population = pd.concat([surviving_population_male, surviving_population_female])

    # Create the line plot
    fig = px.line(
        surviving_population,
        x='age_in_2024',
        y='survival_perc',
        color='geslacht',
        labels={'age_in_2024': 'Age in 2024', 'survival_perc': 'Survival Percentage'},
        title=f'Survival Percentage in {2024+aantal_jaar} by Age in 2024'
    )

    # Show the plot
    st.plotly_chart(fig)
    st.divider()

    # Bereken 85-plussers in 2060
    st.header("85-plussers in 2060")
    
    tab1, tab2 = st.tabs(["Variërende sterftekansen", "Vaste sterftekansen 2025"])
    
    with tab1:
        st.subheader("Met variërende sterftekansen per jaar")
        with st.spinner("Berekenen van 85-plussers in 2060 (variërende rates)..."):
            female_85plus_2060, male_85plus_2060, total_85plus_2060 = calculate_85plus_in_2060(
                population_data, mortality_female, mortality_male
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vrouwen 85+", f"{female_85plus_2060:,.0f}")
        with col2:
            st.metric("Mannen 85+", f"{male_85plus_2060:,.0f}")
        with col3:
            st.metric("Totaal 85+", f"{total_85plus_2060:,.0f}")
        
        st.info(f"Met variërende sterftekansen: **{total_85plus_2060:,.0f}** mensen van 85+ jaar in 2060")
    
    with tab2:
        st.subheader("Met vaste sterftekansen van 2025")
        with st.spinner("Berekenen van 85-plussers in 2060 (vaste rates 2025)..."):
            female_85plus_2060_fixed, male_85plus_2060_fixed, total_85plus_2060_fixed = calculate_85plus_in_2060_fixed_2025_rates(
                population_data, mortality_female, mortality_male
            )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Vrouwen 85+", f"{female_85plus_2060_fixed:,.0f}")
        with col2:
            st.metric("Mannen 85+", f"{male_85plus_2060_fixed:,.0f}")
        with col3:
            st.metric("Totaal 85+", f"{total_85plus_2060_fixed:,.0f}")
        
        st.info(f"Met vaste sterftekansen van 2025: **{total_85plus_2060_fixed:,.0f}** mensen van 85+ jaar in 2060")
    
    # Vergelijking
    st.subheader("Vergelijking")
    verschil = total_85plus_2060 - total_85plus_2060_fixed
    verschil_perc = (verschil / total_85plus_2060_fixed) * 100
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Verschil (absoluut)", f"{verschil:,.0f}")
    with col2:
        st.metric("Verschil (%)", f"{verschil_perc:+.1f}%")
    
    if verschil > 0:
        st.success(f"De variërende sterftekansen leiden tot **{abs(verschil):,.0f} meer** 85-plussers in 2060 dan met vaste sterftekansen van 2025.")
    else:
        st.warning(f"De variërende sterftekansen leiden tot **{abs(verschil):,.0f} minder** 85-plussers in 2060 dan met vaste sterftekansen van 2025.")

    
    st.info("Geinspireerd door https://x.com/BonneKlok/status/1941172543832146010")
    st.divider()
    st.info("BRONNEN:")
    st.info("Bevolking 2024: CBS")
    st.info("Overlevingstabel: https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2")

if __name__ == "__main__":
    main()