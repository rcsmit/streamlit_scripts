"""
Created on Tue Aug 11 15:34:30 2020

@author: dan

https://github.com/hsma-programme/3a_introduction_to_discrete_event_simulation/blob/main/3A_Introduction_to_Discrete_Event_Simulation/Lecture_Examples/simple_simpy.py
License : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License

reproducing https://medium.com/towards-data-science/simulating-a-theme-park-understanding-queue-times-with-r-100b12d97cd3

"""

import simpy
import random
#import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def calculate_mean(x):
    return sum(x)/len(x)

def guest_generator_see_character(env, mean_inter_arrival_time, mean_time_spent, character):
    """
        Generator function for our guest generator (that will bring new guests
        into the model).  We pass into the function the simulation environment along
        with any parameter values and resources we'll need to use or pass on here.
        In this example, we pass in mean inter-arrival time for guests coming in,
        the mean time guests spend in the meet and greet and the character resource.
    Args:
        env (_type_): _description_
        mean_inter_arrival_time (_type_): _description_
        mean_time_spent (_type_): _description_
        character (_type_): _description_

    Yields:
        _type_: _description_
    """    
    g_id = 0 # We'll set this up to give each guest created a unique ID

    while True:
        wp = activity_generator_see_character(env, mean_time_spent, character, g_id)
        env.process(wp)
        t = random.expovariate(1.0 / mean_inter_arrival_time)
        yield env.timeout(t)
        g_id += 1
        
def activity_generator_see_character(env, mean_time_spent, character, g_id):
    """
    Generator function for the activities that our entities (guests here) will
    queue for.  Think of this as the function that describes the guest's
    journey through the system.  It needs to be passed the environment, along
    with any parameter values and resources it needs - here, the mean
    time the guest spends with the character, the character resource and the guest ID
    Args:
        env (_type_): _description_
        mean_time_spent (_type_): _description_
        character (_type_): _description_
        g_id (_type_): _description_

    Yields:
        _type_: _description_
    """   

    time_entered_queue_for_character = env.now
    # st.write (f"guest {g_id} joined queue @ {time_entered_queue_for_character:.1f}")
    
    with character.request() as req:
        yield req
        time_left_queue_for_character = env.now
        #st.write (f"guest {g_id} left queue @ {time_left_queue_for_character:.1f}")
        time_in_queue_for_character = (time_left_queue_for_character -
                                   time_entered_queue_for_character)
        #st.write (f"guest {g_id} queued for {time_in_queue_for_character:.1f} mins")
        time_queue.append(time_in_queue_for_character)
        start_visits.append(time_left_queue_for_character)
        #sampled_consultation_time = random.expovariate(1.0 / mean_time_spent)
        sampled_consultation_time = random.normalvariate( mean_time_spent, 1)
        consumption_time.append(sampled_consultation_time)
        yield env.timeout(sampled_consultation_time)
        #st.write (f"***guest {g_id} finished at {env.now:.1f}")
        #st.write (f"***guest {g_id} finished at {env.now:.1f}")


def make_plot(time_queue,x_label,y_label):
    """_summary_

    Args:
        time_queue (list): _description_
        x_label (str): _description_
        y_label (str): _description_
    """    
    averages = []
    confidence_intervals = []
    total_wait = 0

    for i, wait in enumerate(time_queue, start=1):
        total_wait += wait
        average = total_wait / i
        averages.append(average)
        confidence_level = 0.95
        standard_error = np.std(time_queue[:i], ddof=1) / np.sqrt(i)
        margin_of_error = stats.t.ppf((1 + confidence_level) / 2, df=i-1) * standard_error
        confidence_intervals.append(margin_of_error)

        
    # Create traces for each line
    x_range = list(range(1, len(time_queue) + 1))
    trace1 = go.Scatter(x=x_range, y=time_queue, mode='lines', line=dict(color='blue'), name='Time Queue')
    trace2 = go.Scatter(x=x_range, y=averages, mode='lines', line=dict(color='red'), name='Averages')
    
    # Create confidence interval band
    confidence_interval_upper = [avg + ci for avg, ci in zip(averages, confidence_intervals)]
    confidence_interval_lower = [avg - ci for avg, ci in zip(averages, confidence_intervals)]

    trace_3 = go.Scatter(x=x_range, y=confidence_interval_upper,
                                    line=dict(color='dimgrey', width=.5),)# fill down to xaxis
    trace_4 = go.Scatter(x=x_range, y=confidence_interval_lower, fill='tonexty',fillcolor='rgba(0, 128, 0, 0.2)',
                                    line=dict(color='dimgrey', width=.5),) # fill to trace0 y

    layout = go.Layout(title=f'{y_label} Evolution', xaxis=dict(title=x_label), yaxis=dict(title=y_label))
    fig = go.Figure(data=[trace1, trace2, trace_3, trace_4], layout=layout)
    st.plotly_chart(fig)
  
 
def gantt_chart(time_queue, consumption_time):

    """Make a gantt chart

    GIVES AN ERROR
    TypeError: Object of type timedelta is not JSON serializable
    """

    guest_ids = [f'Guest {i}' for i in range(1, len(time_queue) + 1)]
    # Create a DataFrame for the Gantt chart
    data = {'Task': guest_ids, 'Start': time_queue, 'Finish': [queue + cons for queue, cons in zip(time_queue, consumption_time)]}
    df = pd.DataFrame(data)
    
    # Convert start and finish times to datetime-like objects
    df['Start'] = pd.to_datetime(df['Start'], unit='d', origin='unix')
    df['Finish'] = pd.to_datetime(df['Finish'], unit='d', origin='unix')
    # df['Start'] = df['Start']*10 #pd.to_datetime(df['Start'], unit='m', origin='unix')
    # df['Finish'] =df['Finish']*10 # pd.to_datetime(df['Finish'], unit='m', origin='unix')

    st.write(df)
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", title='character Occupancy Gantt Chart')
    fig.update_yaxes(categoryorder='total ascending')  # Display guests from top to bottom
    fig.update_layout(showlegend=False)

    st.plotly_chart(fig)
  
def footer():
    st.info("Reproducing https://medium.com/towards-data-science/simulating-a-theme-park-understanding-queue-times-with-r-100b12d97cd3")
    st.info("Based on https://github.com/hsma-programme/3a_introduction_to_discrete_event_simulation/blob/main/3A_Introduction_to_Discrete_Event_Simulation/Lecture_Examples/simple_simpy.py \nLicense : Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License")


def main():
    st.header("Wait times for characters in a Disney Park")
    mean_inter_arrival_time = st.sidebar.number_input("Mean inter arrival time (min)", 0, 100,8) # 8
    mean_time_spent =  st.sidebar.number_input("Mean time spent (min)", 0, 100,5) # 5
    number_of_characters =  st.sidebar.number_input("Number of characters", 1, 100,1) 
    runtime =  st.sidebar.number_input("Runtime (hours)", 1, 100,15) *60 # 900
    env = simpy.Environment()
    character = simpy.Resource(env, capacity=number_of_characters)
    env.process(guest_generator_see_character(env, mean_inter_arrival_time, mean_time_spent, character))
    
    env.run(until=runtime)

    st.write (f"mean time queue: {calculate_mean(time_queue)}")
    st.write (f"mean consumptation time: {calculate_mean(consumption_time)}")
    st.write (f"Characters occupied {round(sum(consumption_time)/(runtime*number_of_characters)*100,1)}% of the time")


    make_plot(time_queue,"Simulation time", "Waiting time")
    make_plot(consumption_time,"Simulation time", "consumption time")
    # gantt_chart(start_visits, consumption_time)
    # TO ADD> Fast passes and renenging guests

    # https://pythonhosted.org/SimPy/Tutorials/TheBank2.html
    # https://pythonhosted.org/SimPy/Tutorials/TheBank2OO.html
    footer()



if __name__ == "__main__":
    time_queue = []
    start_visits = []
    consumption_time=[]

    main()