import random
import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np


# House class to represent each house
class House:
    def __init__(self, house_id):
        self.house_id = house_id
        self.clean_count = 0
    
    def clean(self):
        self.clean_count += 1

# Rene class to represent Rene and her cleaning schedule
class Rene:
    def __init__(self, num_houses):
        self.houses = [House(i) for i in range(1, num_houses+1)]
    
    def clean_houses(self, houses_per_day):
        houses_to_clean = random.sample(self.houses, houses_per_day)
        for house in houses_to_clean:
            house.clean()

# Function to simulate the cleaning process over a specified period
def simulate_cleaning(days,num_houses,houses_per_day):
    rene = Rene(num_houses)
    for _ in range(days):
        rene.clean_houses(houses_per_day)
    return rene.houses

def simulation_cleaning(num_days, num_houses, houses_per_day):
    houses = simulate_cleaning(num_days, num_houses, houses_per_day)

    # Collect the data for plotting
    house_ids = [house.house_id for house in houses]
    clean_counts = [house.clean_count for house in houses]

    # Create a DataFrame for plotting
    df = pd.DataFrame({'House ID': house_ids, 'Clean Count': clean_counts})

    # Plot the data using Plotly
    fig = px.bar(df, x='House ID', y='Clean Count', title='Number of Times Each House Has Been Cleaned by Rene')
    st.subheader("House Cleaning  by Rene Simulation")
    st.plotly_chart(fig)



# Probability calculation function
def calculate_prob_not_cleaned_any(days, total_houses, houses_cleaned_per_day):
    p_house_x_not_chosen = (100-houses_cleaned_per_day)/100
    prob_not_cleaned = p_house_x_not_chosen ** days
    prob_not_cleaned_any = 1 - (1 - prob_not_cleaned) ** total_houses
    return prob_not_cleaned_any

def prob_not_cleaned_any(num_days, num_houses, houses_per_day):
    # Calculate probabilities for days from 0 to 200
    days_range = np.arange(0, 200+1)
    prob_not_cleaned_any_list = [calculate_prob_not_cleaned_any(d, num_houses, houses_per_day) for d in days_range]



    # Find the first day where the probability reaches 0.5
    threshold = 0.5
    day_reaching_threshold  = next((d for d, p in zip(days_range, prob_not_cleaned_any_list) if p <= threshold), None)
    threshold_  = next((p for d, p in zip(days_range, prob_not_cleaned_any_list) if p <= threshold), None)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({'Days': days_range, 'Probability Not Cleaned': prob_not_cleaned_any_list})
    # Filter out rows where the probability is smaller than 0.0001
    df = df[df['Probability Not Cleaned'] >= 0.001]
    # Plot the data using Plotly
    fig = px.line(df, x='Days', y='Probability Not Cleaned', title='Probability that at Least One House is Not Cleaned by Rene Over Time')
    fig.add_scatter(x=[day_reaching_threshold], y=[threshold_], mode='markers', name=f'Threshold 0.5 @ {int(day_reaching_threshold)}', marker=dict(color='red', size=5))

    # Display the plot using Streamlit
    st.subheader("Probability of Not Being Cleaned by Rene")

    st.plotly_chart(fig)
    st.write(day_reaching_threshold)
def main():
    """Simulation of cleaning by Rene. 

    https://chatgpt.com/c/5ec72df1-c475-42ee-aaf9-052a2d7f4281

    """

    st.title("House Cleaning Simulation")
    num_days = st.sidebar.number_input("Number of days", 0, 1000,60)
    num_houses = st.sidebar.number_input("Number of houses", 0, 1000,100)
    houses_per_day = st.sidebar.number_input("Houses per day", 0, 1000,6)
    simulation_cleaning(num_days, num_houses, houses_per_day)
    prob_not_cleaned_any(num_days, num_houses, houses_per_day)
    
    if st.sidebar.button("Rerun"):
        st.rerun()

main()