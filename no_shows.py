# https://github.com/gswarge/Monte_carlo_simulation_airline_overbooking/blob/master/Airline_Overbooking_Simulation.ipynb
# https://medium.com/@gaurang.swarge/airline-ticket-overbooking-monte-carlo-simulation-9e276cc2bd8a

import random as rd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import math
from scipy.stats import binom
from scipy.stats import gaussian_kde
from scipy.stats import norm
import matplotlib


def show_up(prob_showup):
    """Does the person show up

    Args:
        prob_showup (float): prob pax shows up

    Returns:
        bool: True if shows up.
    """  
    if rd.random() <= prob_showup :
        return True; #person showed up
    else:
        return False; #person didnt show up
    
def estimate_n_binomial(p,max_n=1000, tolerance=0.05):
    """
    Estimates the original number of trials (n) in a binomial distribution.

    Args:
        p: The probability of success in a single trial.
        proportion: The proportion of successes (x/n).
        max_n: The maximum value of n to consider (optional).

    Returns:
        The estimated value of n, or None if no suitable n is found.
    """
       # Assuming x/n is also 0.9

    for n in range(100,max_n + 1):
        x =round(n*p)
        # proportion = proportion = 96 / n
        # x = round(proportion * n)  
        
        cumulative_prob = binom.cdf(x , n, p)  # Check cumulative prob up to x-1

        # Check if the cumulative probability is close enough to 0.5
        if abs(cumulative_prob - 0.5) < tolerance:
            return n

    return None  # No suitable n found within the given range

def make_box_plot(data,xlabel,ylabel,y_low,y_high, title):
    """Make box plot

    Args:
        data (?): The data
        xlabel (str): x label
        ylabel (str): y label
        x_low (int): lowest value x axis - not used
        x_high (int): highest value x axis - not used
        title (str): title
    """
    # Create a box plot with Plotly
    fig = go.Figure()

    for i, col in enumerate(data.T):
        fig.add_trace(go.Box(y=col, name=str(i)))

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title=title
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


def density_plot(data,xlabel,ylabel,x_low,x_high, title):
    """Make density plot

    Args:
        data (?): The data
        xlabel (str): x label
        ylabel (str): y label
        x_low (int): lowest value x axis
        x_high (int): highest value x axis
        title (str): title
    """
    
    # Create a density plot with Plotly
    fig = go.Figure()

    # Generate KDE for each overbooking level and plot
    for i, col in enumerate(data.T):
        try:
            kde = gaussian_kde(col)
            x_vals = np.linspace(x_low, x_high)
            y_vals = kde(x_vals)
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'{i}'))
        except np.linalg.LinAlgError:
            print(f"Skipping overbooked level {i} due to singular data covariance matrix")

    # Update layout
    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        # xaxis=dict(range=ylimit),
        # yaxis=dict(range=xlimit),
        title=title
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    
    # st.write(title)
    # # Plotting the Density Plots turned 90 degrees for each overbooking level
    # sns.set()
    # fig, axes = plt.subplots(1, max_overbooking + 1, figsize=(15, 5),  sharey=True)

    # for tickets_overbooked in range(max_overbooking + 1):
    #     sns.kdeplot(revenue[:, tickets_overbooked], ax=axes[tickets_overbooked], )
    #     axes[tickets_overbooked].set_xlabel("_")
    #     axes[tickets_overbooked].set_ylabel("Net Revenue")
    #     axes[tickets_overbooked].set_title(f"{tickets_overbooked}")
    #     #axes[tickets_overbooked].set_ylim(30000,50000)
   
    # # Displaying the density plots in Streamlit
    # st.pyplot(fig)


def make_lineplot(x, y, title):
    """Make line plot

    Args:
        x (list): x values
        y (list): y values
        title (str): title
    """    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                            x=x,
                            y=y,
                            mode='lines+markers',
                            ))
    fig.update_layout(
            title=title ,)
    st.plotly_chart(fig)



def vedant_sanil(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking):
    """
    https://vedant-sanil.github.io/science/2021/02/13/airplane-overbooking-problem.html
    
     Args:
        prob_showup (float): probability for show up
        remburse_percentage (int): Percentage of the no-show gets reimbursed
        voucher_cost (int): Amoount of money the turned away pax gets
        seat_capacity (int): Amount of people in the plane
        revenue_per_seat (int): Average ticket price
        no_simulations (int): number of simulations
        max_overbooking (int): max overbooked pax
    """    
   
    exp_revenue = []
    extra_tickets=[]
    for n in range(seat_capacity, seat_capacity+max_overbooking):
        rev_ls = []

        # Running 100000 iterations for deterministic results 
        for i in range(no_simulations):

            # As disucssed earlier, the probability of a single passenger
            # showing up, according to airline data, is around 95%. Therefore,
            # we can sample the number of passengers who show up from a binomial distribution.
            show = np.random.binomial(n, prob_showup)

            # Computing total revenue
            if show <=seat_capacity:
                rev = revenue_per_seat * show 
            else: 
                rev = revenue_per_seat * seat_capacity - voucher_cost * (show-seat_capacity)
                #rev = (tickets_sold * revenue_per_seat - rembursed - upset_customers* revenue_per_seat) - (voucher_cost * upset_customers)
            rev_ls.append(rev)
        extra_tickets.append(n-seat_capacity)
        rev_mean = np.mean(np.array(rev_ls))
        exp_revenue.append(rev_mean)
        
    make_lineplot(extra_tickets,exp_revenue,  "Vedant Sanil")


def cory_simon(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking):
    """https://corysimon.github.io/articles/by-how-many-flights-should-an-airline-overbook/

    Args:
        prob_showup (float): probability for show up
        remburse_percentage (int): Percentage of the no-show gets reimbursed
        voucher_cost (int): Amoount of money the turned away pax gets
        seat_capacity (int): Amount of people in the plane
        revenue_per_seat (int): Average ticket price
        no_simulations (int): number of simulations
        max_overbooking (int): max overbooked pax
    """    
    plt.style.use('bmh')
   
    matplotlib.rc('lines',linewidth=3)
    matplotlib.rc('font',size=16)

    # revenue we make from each ticket sold ($)
    revenue_per_ticket = revenue_per_seat
    # cost of a voucher ($)
    cost_per_voucher = voucher_cost

    # probability any given passenger who bought a ticket will show up for his/her flight
    p = prob_showup

    # total number of seats on the airplane.
    nb_total_seats = seat_capacity

    # Goal: find expected net revenue per flight as a function of `x`, the number of tickets sold beyond capaacity.
    #    i.e. we are selling `nb_total_seats` + `x` tickets.
    #   net revenue = (revenue from tickets) - (cost of voucher payoffs to overbook customers)
    #  We will find net revenue for `x` = 0, 1, 2, ..., N_x
    #  (Note we only consider `x` >= 0 b/c we at least sell a ticket for each seat!) 
    N_x = max_overbooking

    # pre-allocate here. net_revenue[i] := net revenue for x = i.
    expected_net_revenue = np.zeros((N_x, ))

    ## expected net revenue as a function of x
    for x in range(N_x):
        # mean and variance in binomial distribution for this $x$.
        # e.g. mean is referring to the # of customers we expect to show up given we sold (nb_total_seats+x) tickets
        mean = (nb_total_seats + x) * p
        sig2 = (nb_total_seats + x) * p * (1 - p)
        
        # pre-allocate expected voucher payoffs and ticket revenue we expect for this `x`
        expected_voucher_payoffs = 0.0
        expected_ticket_revenue = 0.0
        
        # consider the probability that $k$ customers show up to the flight
        # anywhere from 0, 1, 2, ..., nb_total_seats+x customers could show up
        #    ... since we sold nb_total_seats+x tickets!
        for k in range(nb_total_seats + x + 1):
            # to calculate Pr(N=k| x), governed by binomial dist'n, use normal approximation to binomial
            # let Z ~ Normal(0, 1)
            #   Pr(N=k|x) ~ Prob(l < Z < h)
            #   subtract cumulative distribution (cdf) functions for this
            h = (k + 0.5 - mean) / math.sqrt(sig2) # -0.5 is for continuity correction
            l = (k - 0.5 - mean) / math.sqrt(sig2)
            prob_k_show_up = norm.cdf(h) - norm.cdf(l)
            
            # calculate ticket revenue given `k` customers show up
            ticket_revenue = revenue_per_ticket * np.min([nb_total_seats, k])
        
            expected_ticket_revenue += prob_k_show_up * ticket_revenue

            # calculate voucher payoffs
            voucher_payoffs = cost_per_voucher * np.max([0, k - nb_total_seats])

            expected_voucher_payoffs += prob_k_show_up * voucher_payoffs
        expected_net_revenue[x] = expected_ticket_revenue - expected_voucher_payoffs

        #expected_net_revenue[x] = (tickets_sold * revenue_per_seat - rembursed - upset_customers* revenue_per_seat) - (voucher_cost * upset_customers)
    make_lineplot(list(range(N_x)), expected_net_revenue, "Cory Simon")

def simulate_flight(tickets_sold,prob_showup):
    """Generate the number of show ups

    Args:
        tickets_sold (int): tickets sold
        prob_showup (float): prob of show up

    Returns:
        n (int): people showing up
    """  
    n=0;
    for i in range(1,tickets_sold):
        if(show_up(prob_showup)):
            n = n+1; 
    return n

def estimate(prob_showup):
    estimated_n = estimate_n_binomial(prob_showup)

    if estimated_n:
        st.write(f"The estimated original number (n) is: {estimated_n}")
    else:
        st.write("No suitable n found within the given range.")
#simulating the net Revenue per flight
def simulate_net_revenue(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking, tickets_sold):
    """_summary_

     Args:
        prob_showup (float): probability for show up
        remburse_percentage (int): Percentage of the no-show gets reimbursed
        voucher_cost (int): Amoount of money the turned away pax gets
        seat_capacity (int): Amount of people in the plane
        revenue_per_seat (int): Average ticket price
        no_simulations (int): number of simulations
        max_overbooking (int): max overbooked pax
        tickets_sold (int) : tickets sold
    

    Returns:
        _type_: _description_
    """   
    total_showups = simulate_flight(tickets_sold,prob_showup);
    #total_showups = np.random.poisson(tickets_sold, prob_showup)

    # no one bumped from flight if less or equal folks show up than for the number of seats we have
    no_show = tickets_sold- total_showups
    rembursed = no_show*  revenue_per_seat * (remburse_percentage/100)
    if (total_showups <= seat_capacity):
        return revenue_per_seat * tickets_sold - rembursed, total_showups;
    else:
        upset_customers = total_showups - seat_capacity;
        return (tickets_sold * revenue_per_seat - rembursed - upset_customers* revenue_per_seat) - (voucher_cost * upset_customers) , total_showups;
   

def gaurang(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking):
    """ https://medium.com/@gaurang.swarge/airline-ticket-overbooking-monte-carlo-simulation-9e276cc2bd8a

    Args:
        prob_showup (float): probability for show up
        remburse_percentage (int): Percentage of the no-show gets reimbursed
        voucher_cost (int): Amoount of money the turned away pax gets
        seat_capacity (int): Amount of people in the plane
        revenue_per_seat (int): Average ticket price
        no_simulations (int): number of simulations
        max_overbooking (int): max overbooked pax
    """    

    revenue = np.zeros(shape = (no_simulations,max_overbooking+1));
    people_showing_up = np.zeros(shape = (no_simulations,max_overbooking+1));
    #Running the simulation
    extra_tickets, exp_revenue = [],[]
    for tickets_overbooked in range(0,max_overbooking):
        rev_1=[]
        tickets_sold = seat_capacity + tickets_overbooked;
        for i in range(0,no_simulations):
            revenue[i,tickets_overbooked],  people_showing_up[i,tickets_overbooked] = simulate_net_revenue(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking, tickets_sold)
            
            rev_1.append(revenue[i,tickets_overbooked])
        extra_tickets.append(tickets_overbooked)
        rev_mean = np.mean(np.array(rev_1))
        exp_revenue.append(rev_mean)

    make_box_plot(revenue,"Tickets oversold","Revenue",30000,50000, "Gaurang - Revenue vs tickets oversold")
    make_lineplot(extra_tickets, exp_revenue, "Gaurang - mean")
    make_box_plot(people_showing_up,"Tickets oversold","People", seat_capacity-max_overbooking,seat_capacity+max_overbooking, "Gaurang - How many people show up vs tickets oversold")
    density_plot(people_showing_up,"People showing up","Probability", seat_capacity-max_overbooking,seat_capacity+max_overbooking, "Gaurang - People showing up")
    
    #density_plot(max_overbooking, revenue, "Revenue vs tickets oversold")


def main():
    
    prob_showup = 1- st.sidebar.number_input("Percentage no show",0,100,10)/100
    remburse_percentage = st.sidebar.number_input("Percentage of no show who get money back (100 to compare with others)",0,100,100)
    voucher_cost = st.sidebar.number_input("Voucher cost per person",0,10_000,800);

    with st.sidebar.expander("Advanced"):
        seat_capacity = st.sidebar.number_input("Number of seats",0,1000,100);
        revenue_per_seat = st.sidebar.number_input("Revenue per seat",0,1000,400);
        no_simulations = st.sidebar.number_input("Number of simulations",0,1_000_000,1000);
        max_overbooking = st.sidebar.number_input("Max overbooking",0,100,20);
    
    estimate(prob_showup)
    
    gaurang(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking)
    vedant_sanil(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking)
    cory_simon(prob_showup, remburse_percentage, voucher_cost, seat_capacity, revenue_per_seat, no_simulations, max_overbooking)

    st.info("Based on https://medium.com/@gaurang.swarge/airline-ticket-overbooking-monte-carlo-simulation-9e276cc2bd8a")
    st.info("https://vedant-sanil.github.io/science/2021/02/13/airplane-overbooking-problem.html")
    st.info("https://corysimon.github.io/articles/by-how-many-flights-should-an-airline-overbook/")
    st.info("https://www.nytimes.com/2007/05/30/business/30bump.html?pagewanted=all&_r=0")


if __name__ == "__main__":
    main()

#107 x 0,9 = 96.3
 # binominal - n=107, x=96 (X<=96)=0.5