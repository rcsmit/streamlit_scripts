import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Load the population data
population_data = pd.read_csv(r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\bevolking_leeftijd_NL.csv", sep=';')

# Load the mortality tables
mortality_female = pd.read_csv(r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\AG2024DefinitiefGevalideerd_female.csv")
mortality_male = pd.read_csv(r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\AG2024DefinitiefGevalideerd_male.csv")

# Filter population data for the year 2024
population_2024 = population_data[population_data['jaar'] == 2024]

# Initialize surviving population
surviving_population_female = population_2024[population_2024['geslacht'] == 'F']
surviving_population_male = population_2024[population_2024['geslacht'] == 'M']

# Add a column to track the initial age in 2024
surviving_population_female['age_in_2024'] = surviving_population_female['leeftijd']
surviving_population_male['age_in_2024'] = surviving_population_male['leeftijd']

# Add a column to track the initial age in 2024
surviving_population_female['aantal_in_2024'] = surviving_population_female['aantal']
surviving_population_male['aantal_in_2024'] = surviving_population_male['aantal']

# Loop through each year from 2025 to 2066
for year in range(2025, 2066):
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

    # Apply survival probabilities
    surviving_population_female['aantal'] *= (1 - surviving_population_female[str(year)])
    surviving_population_male['aantal'] *= (1 - surviving_population_male[str(year)])

    # Drop unnecessary columns
    surviving_population_female = surviving_population_female.drop(columns=[str(year), 'age'])
    surviving_population_male = surviving_population_male.drop(columns=[str(year), 'age'])


# Calculate the total surviving population in 2066
total_population_2066 = surviving_population_female['aantal'].sum() + surviving_population_male['aantal'].sum()

# Calculate the total population in 2024
total_population_2024 = surviving_population_female['aantal_in_2024'].sum() + surviving_population_male['aantal_in_2024'].sum()

# Calculate the percentage of people still alive in 2066
percentage_alive_2066 = (total_population_2066 / total_population_2024) * 100
surviving_population_male['survival_perc'] = surviving_population_male['aantal'] / surviving_population_male['aantal_in_2024'] *100
surviving_population_female['survival_perc'] = surviving_population_female['aantal'] / surviving_population_female['aantal_in_2024'] *100


st.write (surviving_population_male)

st.write (surviving_population_female)


st.write(f"Total population in 2066 who were alive in 2024: {total_population_2066:.0f}")
st.write(f"Percentage of the 2024 population still alive in 2066: {percentage_alive_2066:.2f}%")

import plotly.express as px

# Combine male and female data for plotting
surviving_population = pd.concat([surviving_population_male, surviving_population_female])

# Create the line plot
fig = px.line(
    surviving_population,
    x='age_in_2024',
    y='survival_perc',
    color='geslacht',
    labels={'age_in_2024': 'Age in 2024', 'survival_perc': 'Survival Percentage'},
    title='Survival Percentage by Age in 2024'
)

# Show the plot
st.plotly_chart(fig)