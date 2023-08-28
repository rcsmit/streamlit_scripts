"""
Created on Tue Aug 11 15:34:30 2020

@author: dan

https://github.com/hsma-programme/3a_introduction_to_discrete_event_simulation/blob/main/3A_Introduction_to_Discrete_Event_Simulation/Lecture_Examples/simple_simpy.py
License : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

reproducing https://medium.com/towards-data-science/simulating-a-theme-park-understanding-queue-times-with-r-100b12d97cd3

"""
import simpy
import random
import streamlit as st

#import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def calculate_mean(x):
    return sum(x)/len(x)


def make_plot(values,x_label,y_label, run, number_of_runs):
    """Make a plot, with an average value in time (from 0 to n) and the standard deviation of it 

    Args:
        values (list): list with the values to plot
        x_label (str): x label
        y_label (str): y label
    """    
    averages = []
    confidence_intervals = []
    total_value = 0

    for i, value in enumerate(values, start=1):
        if i==1:
            # avoid error:"Degrees of freedom <= 0 for slice"
            confidence_intervals.append(0.0)
        else:
            total_value += value
            average = total_value / i
            averages.append(average)
            confidence_level = 0.95
            standard_error = np.std(values[:i], ddof=1) / np.sqrt(i)
            margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=i-1) * standard_error
            confidence_intervals.append(margin_of_error)

    
    # Create traces for each line
    x_range = list(range(1, len(values) + 1))
    values_ = go.Scatter(x=x_range, y=values, mode='lines', line=dict(color='blue'), name='Time Queue')
    averages_ = go.Scatter(x=x_range, y=averages, mode='lines', line=dict(color='red'), name='Averages')
    
    # Create confidence interval band
    confidence_interval_upper = [avg + ci for avg, ci in zip(averages, confidence_intervals)]
    confidence_interval_lower = [avg - ci for avg, ci in zip(averages, confidence_intervals)]

    conf_interv_high = go.Scatter(x=x_range, y=confidence_interval_upper,
                                    line=dict(color='dimgrey', width=.5),)# fill down to xaxis
    conf_interv_low = go.Scatter(x=x_range, y=confidence_interval_lower, fill='tonexty',fillcolor='rgba(0, 128, 0, 0.2)',
                                    line=dict(color='dimgrey', width=.5),) # fill to trace0 y

    layout = go.Layout(title=f'{y_label} Evolution [{run} / {number_of_runs}]', xaxis=dict(title=x_label), yaxis=dict(title=y_label))
    fig = go.Figure(data=[values_, averages_, conf_interv_high, conf_interv_low], layout=layout)
    st.plotly_chart(fig)
  
# Class to store global parameter values.  We don't create an instance of this
# class - we just refer to the class blueprint itself to access the numbers
# inside.  Therefore, we don't need to specify a constructor.
class g:
   
    number_of_runs = st.sidebar.number_input("Number of runs (#)", 1, 10,3) 
    
    mean_inter_arrival_time = st.sidebar.number_input("Mean inter arrival time (min)", 0, 60,8) # 8
    st.sidebar.write(f"[{round(60/mean_inter_arrival_time,1)} guests per hour]")
    mean_time_spent = st.sidebar.number_input("Mean time spent (min)", 0, 60,5) # 5
    number_of_characters = st.sidebar.number_input("Number of characters (#)", 1, 10,2) 
    runtime =  st.sidebar.number_input("Runtime (hours)", 1, 100,15) *60 # 900
    
# Class representing our guests coming in for the meet and greet.
# Here, we only have a constructor method, that sets up the guest's ID
class guest:
    def __init__(self, g_id):
        self.id = g_id
        

class meet_and_greet_model:
  
    def __init__(self, run_number):
        self.env = simpy.Environment()
        self.guest_counter = 0  
        self.character = simpy.Resource(self.env, capacity=g.number_of_characters)

        self.run_number = run_number
        
        self.mean_q_time_character = 0
        self.time_queue = []
        self.start_visits = []
        self.consumption_time=[]
        self.results_df = pd.DataFrame()
        # self.results_df["G_ID"] = []
        # self.results_df["Start_Q_character"] = []
        # self.results_df["End_Q_character"] = []
        # self.results_df["Q_Time_character"] = []
        # self.results_df.set_index("G_ID", inplace=True)
        
    def guest_generator(self):
        # Keep generating indefinitely (until the simulation ends)
        while True:
            self.guest_counter += 1
            wp = guest(self.guest_counter)
            self.env.process(self.see_character(wp))
            sampled_interarrival = random.expovariate(1.0 / g.mean_inter_arrival_time)
            
            # Freeze this function until that time has elapsed
            yield self.env.timeout(sampled_interarrival)
            
    # A method that models the processes for attending the weight loss clinic.
    # The method needs to be passed a guest who will go through these
    # processes
    def see_character(self, guest):
        time_entered_queue_for_character = self.env.now
        with self.character.request() as req:
            yield req
            time_left_queue_for_character = self.env.now
            time_in_queue_for_character = (time_left_queue_for_character -
                                    time_entered_queue_for_character)
            meet_and_greet_duration = random.normalvariate( g.mean_time_spent, 1)

            self.time_queue.append(time_in_queue_for_character)
            self.start_visits.append(time_left_queue_for_character)
            self.consumption_time.append(meet_and_greet_duration)

            df_to_add = pd.DataFrame({"G_ID":[guest.id],
                                      "Start_Q_character":[time_entered_queue_for_character],
                                      "End_Q_character":[time_left_queue_for_character],
                                      "Q_Time_character":[time_in_queue_for_character]})
            df_to_add.set_index("G_ID", inplace=True)
            self.results_df = pd.concat([self.results_df, df_to_add])

            yield self.env.timeout(meet_and_greet_duration)
            
    
    # The run method starts up the entity generator(s), and tells SimPy to
    # start running the environment for the duration specified in the g class.
    def run(self):
        self.env.process(self.guest_generator())
        self.env.run(until=g.runtime)
def show_info():
    st.info("Reproducing https://medium.com/towards-data-science/simulating-a-theme-park-understanding-queue-times-with-r-100b12d97cd3")
    st.info("Based on https://github.com/hsma-programme/3c_simpy_part_2/tree/main/3C_SimPy_for_Discrete_Event_Simulation_Part_2 \nLicense : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License")
    st.info("Source: https://github.com/rcsmit/streamlit_scripts/blob/main/waiting_times_disney_OO.py")
def main():
    st.header("Wait times for characters in a Disney Park")
    show_info()
    end_results_df = pd.DataFrame()
    for run in range(g.number_of_runs):
        st.write(f"**Run {run+1} of {g.number_of_runs}**")
        m = meet_and_greet_model(run)
        m.run()
        st.write()

        m_time_queue = calculate_mean(m.time_queue)
        sd_time_queue = np.std(m.time_queue)
        m_cons_time = calculate_mean(m.consumption_time)
        char_occ = round(sum(m.consumption_time)/(g.runtime*g.number_of_characters)*100,1)
        m.results_df['interval_guests'] =m.results_df['Start_Q_character'].diff()
        m_interval_g = m.results_df['interval_guests'].mean()

        df_to_add = pd.DataFrame({"run_id":[run+1],
                                      "mean_time_q":[m_time_queue],
                                      "sd_time_q":[sd_time_queue],
                                      "mean_cons_time":[m_cons_time],
                                      "characters_occupied":[char_occ],
                                      "mean_interval_guests":[m_interval_g]})
        end_results_df = pd.concat([end_results_df, df_to_add])

        st.write(f"mean time queue: {round(m_time_queue,2)} min")
        st.write(f"mean consumptation time: {round(m_cons_time,2)} min")
        st.write(f"Characters occupied {round(char_occ,1)}% of the time")  
        st.write(f"mean interval guests: {round(m_interval_g,2)} min")
        
        make_plot(m.time_queue,"Simulation time", "Waiting time", run+1, g.number_of_runs)
        make_plot(m.consumption_time,"Simulation time", "consumption time", run+1, g.number_of_runs)
    st.subheader("End Results of all runs")
    st.write(end_results_df)

    
if __name__ == "__main__":
    main()