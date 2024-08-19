import random
import plotly.express as px
import pandas as pd
import streamlit as st
import numpy as np
from scipy.stats import binom

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

    # Filter the DataFrame to find houses not cleaned
    houses_not_cleaned = df[df['Clean Count'] == 0]

    # Calculate the percentage of houses not cleaned
    percentage_not_cleaned = (len(houses_not_cleaned) / len(df)) * 100

    # Print the result
    st.write(f"Percentage of houses not cleaned by Rene: {percentage_not_cleaned:.2f}%")



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
    st.write(f"The treshold is reached on day {day_reaching_threshold}")

def days_to_reach_target_cleaning_probability(total_houses=100, houses_cleaned_per_day=6, target_cleaned_percentage=0.95, target_probability=0.95):
    """
    Calculate the number of days required for a given probability that a target percentage of houses 
    have been cleaned at least once.

    Parameters:
    -----------
    total_houses : int, optional
        The total number of houses to be cleaned. Default is 100.
    houses_cleaned_per_day : int, optional
        The number of houses Rene cleans each day. Default is 6.
    target_cleaned_percentage : float, optional
        The target percentage of houses that need to be cleaned at least once. 
        Must be between 0 and 1. Default is 0.95 (95% of houses).
    target_probability : float, optional
        The desired probability of reaching the target_cleaned_percentage. 
        Must be between 0 and 1. Default is 0.95 (95% probability).

    Returns:
    --------
    int
        The number of days required to have the target_probability that at least 
        target_cleaned_percentage of the houses have been cleaned at least once.

    Example:
    --------
    >>> days_to_reach_target_cleaning_probability(total_houses=100, houses_cleaned_per_day=6, target_cleaned_percentage=0.95, target_probability=0.95)
    156
    """
    st.subheader ("Days to reach target")
    

    # Define the probability of not cleaning a specific house in one day
    prob_not_cleaned_one_day = (total_houses - houses_cleaned_per_day) / total_houses

    # Define the probability of cleaning a house at least once after d days
    def prob_house_cleaned_at_least_once(days):
        # For each house, the probability that it has been cleaned at least once after d days is
        return 1 - prob_not_cleaned_one_day ** days

    # Find the number of days required
    d = 1
    probabilities = []
    target_reached= False
    while True:
        # Calculate the probability of cleaning at least 95% of the houses
        prob_cleaned_houses = prob_house_cleaned_at_least_once(d)
        
        prob_cleaned_95_percent = 1- binom.cdf(int(total_houses * target_cleaned_percentage), total_houses, prob_cleaned_houses)
        
        if (prob_cleaned_95_percent >= target_probability) and  (target_reached==False):
            answer = d
            target_reached = True
        if prob_cleaned_95_percent >= 0.995:
            break
        probabilities.append(prob_cleaned_95_percent)    
        d += 1

    st.write(f"""The number of days required for a probability of {target_probability} 
            that a {target_cleaned_percentage*100}% of the {total_houses} houses 
            have been cleaned at least once while cleanining {houses_cleaned_per_day} 
            houses per day : {answer}""")

    df = pd.DataFrame({
        'Index': range(1, len(probabilities) + 1),  # X-axis values (could be days, steps, etc.)
        'Probability': probabilities
        })
    # Create a line graph using Plotly
    fig = px.line(df, x='Index', y='Probability', title='Probability Line Graph', labels={'Index': 'Index', 'Probability': 'Probability'})
    st.plotly_chart(fig)

def main():
    """Simulation of cleaning by Rene. 

    https://chatgpt.com/c/5ec72df1-c475-42ee-aaf9-052a2d7f4281

    """

    st.title("House Cleaning Simulation")
    num_days = st.sidebar.number_input("Number of days", 0, 1000,60)
    num_houses = st.sidebar.number_input("Number of houses", 0, 1000,100)
    houses_per_day = st.sidebar.number_input("Houses per day", 0, 1000,6)
    target_cleaned_percentage= st.sidebar.number_input("Target cleaned percentage", 0.0,1.0,.95)
    target_probability= st.sidebar.number_input("Target Probability", 0.0,1.0,.95)
    simulation_cleaning(num_days, num_houses, houses_per_day)
    prob_not_cleaned_any(num_days, num_houses, houses_per_day)
    days_to_reach_target_cleaning_probability(num_houses, houses_per_day, target_cleaned_percentage, target_probability)
    if st.sidebar.button("Rerun"):
        st.rerun()


main()