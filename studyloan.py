"""SCRIPT TO CALCULATE THE REPAYMENTS OF THE STUDYLOAN
tested against/reproducing https://duo.nl/particulier/rekenhulp-studiefinanciering.jsp#/nl/terugbetalen/resultaat

Returns:
    _type_: _description_
"""

# 2024 50000
# 2026 52593
# 2061 0
#TOTALE RENTE € 29.327,90
# Totale schuld € 79.327,90

# aflossing per maand 188.88
#2024 25000
# 2026 26296
# TOTALE RENTE € 14.663,95
# aflossing per maand 94.44

import pandas as pd
import streamlit as st
import plotly.express as px

# Constants


def calculate_totals(LEGAL_REPAYMENT,ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS, custom_repayment_amount):
    """calculate the total amount of repayment and interest

    Args:
        LEGAL_REPAYMENT (float): monthly amount of repayment as calculated
        ANNUAL_INTEREST_RATE (float): interest
        INITIAL_DEBT (int): initial debt in euros
        REPAYMENT_TIME_IN_YEARS (int): repayment time in years
        custom_repayment_amount (bool) : custom repayment amount

    """
    if custom_repayment_amount:
        REPAYMENT = st.sidebar.number_input("How much do you want to repay", 0.0,INITIAL_DEBT/1.0,LEGAL_REPAYMENT)
    else:
        REPAYMENT = LEGAL_REPAYMENT
    #REPAYMENT = 188.875 # 94.44 # Monthly payment amount
    # Calculate monthly interest rate
    monthly_interest_rate = (1 + (ANNUAL_INTEREST_RATE / 100)) ** (1 / 12) - 1

    # Initialize lists to store data
    start_year = 2024
    years = [start_year]
    debts = [INITIAL_DEBT]
    y_repayments = [0]
    y_interests = [0]
    total_debts = [INITIAL_DEBT]
    cumm_repayments =[0]
    cumm_interests =[0]
    total_interests = 0
    total_repayments = 0
    yearly_interests = 0
    yearly_repayments = 0
    # Initial year and debt values
    year = start_year +1
    current_debt = INITIAL_DEBT
    loan_completed = False
    # Loop through months
    for month in range(1, (REPAYMENT_TIME_IN_YEARS +2) * 12 + 1):
        
        if month > 0:
            interest = current_debt * monthly_interest_rate
            current_debt += interest
            total_interests += interest
            yearly_interests +=interest
        else:
            interest = 0

        if month > 24:
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
                # else:
                #     current_debt = 0
                    
                   

        if month % 12 == 0:
            if month > 24:
                y_repayments.append(yearly_repayments)
                cumm_repayments.append(total_repayments)
            else:
                y_repayments.append(0)
                cumm_repayments.append(0)
            y_interests.append(yearly_interests)
            total_debts.append(current_debt)
            cumm_interests.append(total_interests)
            years.append(year)
            debts.append(current_debt)
            year += 1
            yearly_interests = 0
            yearly_repayments = 0
            if loan_completed:
                break
            

    st.write(f"Start schuld : €  {round(INITIAL_DEBT,2)}")
    st.write(f"Wettelijk maandbedrag : € {round(LEGAL_REPAYMENT,2)}")
    st.write(f"Einddatum : 01-01-{year-1}") 
    st.write(f"Rentebedrag ({ANNUAL_INTEREST_RATE} %) : € {round(total_interests,2)}")
    
    st.write(f"Totale aflossing : € {round(total_repayments,2)}")
    
    st.write(f"Eindschuld : € {round(current_debt,2)}")
    # Create a DataFrame
    data = {
        "Year": years,
        "Debt": debts,
        "Yearly Repayment": y_repayments,
        "Yearly Interest": y_interests,
        "Total Debt": total_debts,
        "Cumm Repayment": cumm_repayments,
        "Cumm Interest": cumm_interests,
    }


    df = pd.DataFrame(data)

    # Display the DataFrame
    
    
        
    # Create a line chart using Plotly
    fig = px.line(df, x='Year', y=['Debt', 'Yearly Repayment', 'Yearly Interest', 'Total Debt','Cumm Interest', 'Cumm Repayment'],
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

    st.write(df)

def find_amount_repayment(ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS):
    # Define a function to calculate the final debt for a given repayment amount
    def calculate_final_debt(repayment,ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS):
        # Calculate monthly interest rate
        monthly_interest_rate = (1 + (ANNUAL_INTEREST_RATE / 100)) ** (1 / 12) - 1
        
        current_debt = INITIAL_DEBT
        for month in range(1, (REPAYMENT_TIME_IN_YEARS +2) * 12 + 1):
            interest = current_debt * monthly_interest_rate
            current_debt += interest
            if month > 24:
                current_debt -= repayment
            if current_debt <= 0:
                return current_debt
        return current_debt

    # Use binary search to find the repayment amount
    lower_bound = 0
    upper_bound = INITIAL_DEBT
    epsilon = 0.01  # A small threshold for the final debt

    while upper_bound - lower_bound > epsilon:
        repayment = (upper_bound + lower_bound) / 2
        final_debt = calculate_final_debt(repayment,ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS)
        
        if final_debt < 0:
            upper_bound = repayment
        else:
            lower_bound = repayment

    # Calculate the monthly repayment amount
    monthly_repayment = (upper_bound + lower_bound) / 2

    # Display the result
    st.write(f"The monthly repayment amount that results in a final debt of 0 is: {monthly_repayment:.3f}")
    # This script uses a binary search approach to find the monthly repayment amount that would result
    # in a final debt of 0. The result is st.writeed in a user-friendly format.
    return monthly_repayment


def main():
    st.header("Calculate repayment plan studyloan")
    ANNUAL_INTEREST_RATE = st.sidebar.number_input("Yearly interest rate (%)",0.0,99.9,2.95) # 2.56  # Annual interest rate in percentage
    INITIAL_DEBT = st.sidebar.number_input("Initial debt", 0,None,50000) # 25000  # Initial debt
    #REPAYMENT_TIME_IN_YEARS = st.sidebar.selectbox("Repayment in years",[15,35],1)
    REPAYMENT_TIME_IN_YEARS = st.sidebar.number_input("Repayment in years",0,100,35)
    custom_repayment_amount = st.sidebar.selectbox("Custom repayment amount",[True,False], 1)
    monthly_repayment = find_amount_repayment(ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS)
    calculate_totals(monthly_repayment,ANNUAL_INTEREST_RATE,INITIAL_DEBT,REPAYMENT_TIME_IN_YEARS, custom_repayment_amount)


if __name__ == "__main__":
    main()
    