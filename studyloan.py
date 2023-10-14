"""SCRIPT TO CALCULATE THE REPAYMENTS OF THE STUDYLOAN
tested against/reproducing https://duo.nl/particulier/rekenhulp-studiefinanciering.jsp#/nl/terugbetalen/resultaat

Returns:
    _type_: _description_
"""
# 2024 50000
# 2026 52593
# 2061 0
#TOTALE RENTE € 29.327,90 2.56%
# Totale schuld € 79.327,90

# aflossing per maand 188.88
#2024 25000
# 2026 26296
# TOTALE RENTE € 14.663,95
# aflossing per maand 94.44

import pandas as pd
import streamlit as st
import plotly.express as px

import plotly.graph_objects as go
# Constants


def plot_stack_line_graph(df,first,second):
    """Plot a stacked linegraph

    Args:
        df (df): dataframe
        first (str): fieldname of the first values
        second (str): fieldname of the second values
    """
    first_ = go.Scatter(x=df['Year'], y=df[first] , 
                                mode='lines', fill='tonexty', 
                                            fillcolor='rgba(120, 128, 0, 0.2)',
                                            line=dict(width=0.7, 
                                            color="rgba(0, 0, 255, 0.5)"), 
                                            name=first )

    second_ = go.Scatter(x=df['Year'],
                                y=df[second],
                                fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.2)',
                                line=dict(color='dimgrey', width=.5),
                                name="second",)
    
    second_['y'] = [a + b for a, b in zip( first_['y'],second_['y'])]
    fig2 = go.Figure()

    fig2.add_trace(first_)    
    fig2.add_trace(second_) 
    
    # Update layout settings
    fig2.update_layout(
        title=f'{first} and {second}',
        xaxis_title='Years',
        yaxis_title='Euros',
        showlegend=True,
        hovermode='x'
    )

    st.plotly_chart(fig2)
        
def calculate_totals(LEGAL_REPAYMENT,annual_interest_rate,initial_debt,repayment_time_in_years, custom_repayment_amount,start_repaying):
    """calculate the total amount of repayment and interest

    Args:
        LEGAL_REPAYMENT (float): monthly amount of repayment as calculated
        annual_interest_rate (float): interest
        initial_debt (int): initial debt in euros
        repayment_time_in_years (int): repayment time in years
        custom_repayment_amount (bool) : custom repayment amount
        start_repaying(int) : in which month we start repaying

    """
    if custom_repayment_amount:
        REPAYMENT = st.sidebar.number_input("How much do you want to repay", 0.0,initial_debt/1.0,LEGAL_REPAYMENT)
    else:
        REPAYMENT = LEGAL_REPAYMENT
    #REPAYMENT = 188.875 # 94.44 # Monthly payment amount
    # Calculate monthly interest rate
    monthly_interest_rate = (1 + (annual_interest_rate / 100)) ** (1 / 12) - 1

    # Initialize lists to store data
    start_year = 2023
    years = []
    debts_end_year = []
    debts_start_year = []
    y_repayments = []
    y_interests = []
    total_debts = []
    cumm_repayments =[]
    cumm_interests =[]
    total_interests = 0
    total_repayments = 0
    yearly_interests = 0
    yearly_repayments = 0
    # Initial year and debt values
    year = start_year +1
    current_debt = initial_debt
    debt_start_year = initial_debt
    loan_completed = False
    # Loop through months
    for month in range(1, (repayment_time_in_years +2) * 12 + 1):
        
        if month > 0:
            interest = current_debt * monthly_interest_rate
            current_debt += interest
            total_interests += interest
            yearly_interests +=interest
        else:
            interest = 0

        if month > start_repaying:
            if current_debt - REPAYMENT > 0:
                current_debt -= REPAYMENT
                yearly_repayments +=  REPAYMENT
                total_repayments+=  REPAYMENT
            else:
                if current_debt == 0:
                    loan_completed = True
                elif current_debt > 0:
                    
                    yearly_repayments +=  current_debt
                    total_repayments+=  current_debt
                    current_debt = 0

        if month % 12 == 0:
            if month > start_repaying:
                y_repayments.append(yearly_repayments)
                cumm_repayments.append(total_repayments)
            else:
                y_repayments.append(0)
                cumm_repayments.append(0)
            y_interests.append(yearly_interests)
            total_debts.append(current_debt)
            cumm_interests.append(total_interests)
            years.append(year)

            debts_start_year.append(debt_start_year)
            debt_start_year = current_debt
            debts_end_year.append(current_debt)
            year += 1
            yearly_interests = 0
            yearly_repayments = 0
            if loan_completed:
                break
            
    st.write(f"Start schuld : €  {round(initial_debt,2)}")
    st.write(f"Wettelijk maandbedrag : € {round(LEGAL_REPAYMENT,2)}")
    st.write(f"Einddatum : 01-01-{year-1}") 
    st.write(f"Rentebedrag ({annual_interest_rate} %) : € {round(total_interests)} / € {round(total_interests/repayment_time_in_years)} per jaar / € {round(total_interests/(repayment_time_in_years*12))} per maand")
    st.write(f"Totale aflossing : € {round(total_repayments,2)}")
    st.write(f"Eindschuld : € {round(current_debt,2)}")
    # Create a DataFrame
    data = {
        "Year": years,
        "Debt_start_of_year": debts_start_year,
        "Yearly Repayment": y_repayments,
        "Debt_end_of_year": debts_end_year,
        "Yearly Interest": y_interests,
        "Total Debt": total_debts,
        "Cumm Repayment": cumm_repayments,
        "Cumm Interest": cumm_interests,
        
    }

    df = pd.DataFrame(data)

    rolling_interest = 0
    rolling_debt = 0
    paid_to_debt = 0
    df["rolling_debt"] = None
    df["rolling_interest"] = None
    df["paid_to_debt"] = None
    df["yearly_repayment_hoofdsom"] = None

    # Define a function to calculate interest paid and update the rolling interest and debt
    def calculate_interest_and_debt(row):
        nonlocal rolling_interest, rolling_debt
        repayment = row["Yearly Repayment"]
        interest = row["Yearly Interest"]
       
        # Calculate rolling interest
        rolling_interest += interest 

        # Calculate interest paid for this row based on the conditions you provided
        if repayment == 0:
            interest_paid = 0
        elif repayment > 0 and repayment < rolling_interest:
            interest_paid = repayment
            rolling_interest -= interest_paid
        else:  # Repayment >= rolling_interest
            interest_paid = rolling_interest
            rolling_interest = 0

        # Update the rolling debt
        rolling_debt = repayment - interest_paid
        paid_to_debt = repayment - interest_paid

        # # Update the rolling debt and debt left
        # row["rolling_debt"] = rolling_debt
        # row["rolling_interest"] = rolling_interest

        return pd.Series([interest_paid,  rolling_debt, rolling_interest, paid_to_debt])

    # Use the apply method to calculate interest paid and debt paid for each row and update the DataFrame
    df[["Interest Paid", "rolling_debt", "rolling_interest", "paid_to_debt"]] = df.apply(calculate_interest_and_debt, axis=1)
    #In this code, we modify the calculate_interest_and_debt function to return a Pandas 
    df["paid_to_debt_cumm"] = df["paid_to_debt"].cumsum()
    df["Interest Paid_cumm"] = df["Interest Paid"].cumsum()
    df["zero"] = 0
    # Create a line chart using Plotly
    fig = px.line(df, x='Year', y=["Debt_end_of_year", 'Yearly Repayment', 'Yearly Interest', 'Total Debt','Cumm Interest', 'Cumm Repayment'],
                labels={'value': 'Amount'},
                title='Loan Repayment and Interest Over Time',
                markers=True)

    # Customize the layout
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Amount',
        legend_title_text='Legend',
    )

    # Show the plot
    st.plotly_chart(fig)
    
    plot_stack_line_graph(df,"paid_to_debt_cumm","Interest Paid_cumm")
    plot_stack_line_graph(df,"paid_to_debt","Interest Paid")


    st.write(df)

def find_amount_repayment(annual_interest_rate,initial_debt,repayment_time_in_years,start_repaying):
    """ Binary search approach to find the monthly repayment amount that would result
        in a final debt of 0. 
    Args:
        annual_interest_rate (_type_): _description_
        initial_debt (_type_): _description_
        repayment_time_in_years (_type_): _description_
        start_repaying (_type_): _description_

    Returns:
        _type_: _description_
    """    

    def calculate_final_debt(repayment,annual_interest_rate,initial_debt,repayment_time_in_years,start_repaying):
        # Calculate monthly interest rate
        monthly_interest_rate = (1 + (annual_interest_rate / 100)) ** (1 / 12) - 1
        
        current_debt = initial_debt
        for month in range(1, (repayment_time_in_years +2) * 12 + 1):
            interest = current_debt * monthly_interest_rate
            current_debt += interest
            if month > start_repaying:
                current_debt -= repayment
            if current_debt <= 0:
                return current_debt
        return current_debt

    # Use binary search to find the repayment amount
    lower_bound = 0
    upper_bound = initial_debt
    epsilon = 0.01  # A small threshold for the final debt

    while upper_bound - lower_bound > epsilon:
        repayment = (upper_bound + lower_bound) / 2
        final_debt = calculate_final_debt(repayment,annual_interest_rate,initial_debt,repayment_time_in_years,start_repaying)
        
        if final_debt < 0:
            upper_bound = repayment
        else:
            lower_bound = repayment

    # Calculate the monthly repayment amount
    monthly_repayment = (upper_bound + lower_bound) / 2

    # Display the result
    st.write(f"The monthly repayment amount that results in a final debt of 0 is: {monthly_repayment:.3f}")
    return monthly_repayment

def interface():
    annual_interest_rate = st.sidebar.number_input("Yearly interest rate (%)",0.0,99.9,2.56) # 2.56  # Annual interest rate in percentage
    initial_debt = st.sidebar.number_input("Initial debt", 0,None,50000) # 25000  # Initial debt
    repayment_time_in_years = st.sidebar.number_input("Repayment in years",0,100,35)
    start_repaying = st.sidebar.number_input("Start repaying after... (months)",0,repayment_time_in_years*12,24)
    custom_repayment_amount = st.sidebar.selectbox("Custom repayment amount",[True,False], 1)
    return annual_interest_rate,initial_debt,repayment_time_in_years,start_repaying,custom_repayment_amount

def main():
    st.header("Calculate repayment plan studyloan")
    annual_interest_rate, initial_debt, repayment_time_in_years, start_repaying, custom_repayment_amount = interface()
    monthly_repayment = find_amount_repayment(annual_interest_rate,initial_debt,repayment_time_in_years,start_repaying)
    calculate_totals(monthly_repayment,annual_interest_rate,initial_debt,repayment_time_in_years, custom_repayment_amount,start_repaying)

if __name__ == "__main__":
    main()
    