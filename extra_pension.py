import plotly.express as px
import pandas as pd
import streamlit as st

def main():
    """
    This script endeavors to estimate the monthly financial requirement for bridging 
    the deficit in saved pension funds, commonly referred to as the 'AOW gat' 
    (gap in state pension), for individuals who have spent portions of their life 
    outside the Netherlands. (2% is deducted per year). It takes into consideration 
    factors such as inflation and prevailing interest rates. 
    
    Additionally, it allows for the inclusion of an initial contribution 
    and accommodates up to three windfall amounts ('meevallers')."  
    """
    st.title('Pension Calculation')

    i_values = []
    interest_values = []
    balance_values = []
    annual_contribution_values = []
    annual_shortfall= []
    initial_one_time_contribution = st.sidebar.number_input("Initial One-Time Contribution:", value=0)
    monthly_contribution_original = st.sidebar.number_input("Monthly Contribution (current pricelevel):", value=200)
    st.sidebar.subheader("--- The person ---")
    current_age = st.sidebar.number_input("Current Age:", value=46)
    retirement_age = st.sidebar.number_input("Retirement Age:", value=69) #https://www.svb.nl/nl/aow/aow-leeftijd/uw-aow-leeftijd
    expected_life_expectancy =  st.sidebar.number_input("Expected Life Expectancy:", value=81.4) # https://www.berekenhet.nl/pensioen/resterende-levensverwachting.html#calctop
    
    st.sidebar.write(f"Years to go to pension: {retirement_age - current_age}")
    st.sidebar.write(f"Number of years to sustain {round(expected_life_expectancy - retirement_age,1)}")
    st.sidebar.write(f"Ratio : {round((expected_life_expectancy - retirement_age) / (retirement_age - current_age),1)}")
    
    st.sidebar.subheader("--- Rates ---")
    annual_return_rate = st.sidebar.number_input("Annual Interest Rate (%):", value=2.0)
    inflation =  st.sidebar.number_input("Average Annual inflation Rate (%):", value=1.9) # 1.9 volgens https://descryptor.org/opinie/2019/07/25/Donald-Duck-Index
     
    st.sidebar.subheader("--- Pension data ---")                                                                                       # 5.38 volgens https://indeflatie.nl/inflatie-in-nederland-afgelopen-25-jaar
    monthly_pension_without_reduction_original = st.sidebar.number_input("Monthly Pension without Reduction (current pricelevel):", value=1458)
    years_shortfall = st.sidebar.number_input("Years of Shortfall:", value=7)
    st.sidebar.write(f"Shortfall per month (current price level): {round(years_shortfall * 0.02 * monthly_pension_without_reduction_original)}")

    additional_monthly_need = st.sidebar.number_input("Additional Monthly Need (current price level):", value=200)
    st.sidebar.subheader("--- Windfalls ---")
    # windfall = "meevaller"
    windfall_1_year =   st.sidebar.number_input("Windfall 1 (Year):", value=5)
    windfall_1_amount = st.sidebar.number_input("Windfall 1 (Amount):", value=0)

    windfall_2_year =  st.sidebar.number_input("Windfall 2 (Year):", value=15)
    windfall_2_amount = st.sidebar.number_input("Windfall 2 (Amount):", value=0)

    windfall_3_year =  st.sidebar.number_input("Windfall 3 (Year):", value=25)
    windfall_3_amount =  st.sidebar.number_input("Windfall 3 (Amount):", value=0)

    special_years = [windfall_1_year, windfall_2_year, windfall_3_year]
    special_amounts = [windfall_1_amount, windfall_2_amount, windfall_3_amount]
    annual_contribution_original = monthly_contribution_original * 12
    annual_pension = monthly_pension_without_reduction_original * 12
    annual_shortfall_original = ((years_shortfall * 0.02 * annual_pension)) + (additional_monthly_need * 12)

    balance = initial_one_time_contribution
    years_until_retirement = retirement_age - current_age
    
    annual_contribution_corrected = annual_contribution_original
    for i in range(0, years_until_retirement+1):
        interest = round( balance * ((annual_return_rate) / 100),2)

        balance += interest + annual_contribution_corrected
        
        # add windfalls ("meevallers")
        if i in special_years:
            special_index = special_years.index(i)
            special_addition = special_amounts[special_index]
            balance += special_addition
        i_values.append(i)
        interest_values.append(interest)
        balance_values.append(int(balance))
        annual_contribution_values.append(annual_contribution_corrected)
        annual_shortfall.append(None)
        annual_contribution_corrected = round(annual_contribution_corrected * (100+inflation)/100,2)
  
    annual_shortfall_corrected = annual_shortfall_original * (((100+inflation)/100) ** years_until_retirement)
   
    for j in range(years_until_retirement+1,100-current_age+1):# years_until_retirement+years_after_retirement):
        interest = balance * ((annual_return_rate) / 100)
        balance += interest 
        balance -= annual_shortfall_corrected
        annual_shortfall_corrected = annual_shortfall_corrected  * ((100+inflation)/100) 
        i_values.append(j)
        interest_values.append(interest)
        balance_values.append(int(balance))  
        annual_contribution_values.append(0)
        annual_shortfall.append(annual_shortfall_corrected)

    data = pd.DataFrame({'i': i_values,  'annual_contribution': annual_contribution_values, 'balance': balance_values, 'interest': interest_values,'annual_shortfall':annual_shortfall, })
    data["age"] = data["i"] + current_age
    data["monthly_contribution"] = round(data["annual_contribution"] / 12,2)
    
    fig = px.line(data, x='age', y=[ 'balance'], title='Balance Over Time')
    fig.update_traces(mode='lines+markers')
    fig.add_hline(y=0,  line_color="orange")
    
    fig.add_vline(x=retirement_age, line_dash="dash", line_color="green", annotation_text="Retirement Age", annotation_position="bottom left")
    fig.add_vline(x=expected_life_expectancy, line_dash="dash", line_color="black", annotation_text="Expected Life Expectancy", annotation_position="bottom left")
    st.plotly_chart(fig)

    # Find the year when the balance passes 0
    zero_balance_year = data[data['balance'] > 0]['age'].max()
    st.info (f"With your end balance, you can sustain for  until age {zero_balance_year}.") 
    if zero_balance_year > expected_life_expectancy:
        st.success (f"You have money for {round(zero_balance_year - expected_life_expectancy)} years more based on expected life expectancy.")
    else:
        st.warning (f"You have money for {round(expected_life_expectancy - zero_balance_year )} years less based on expected life expectancy.")
    st.subheader("DISCLAIMER")
    st.warning('''Cautious Consideration Required: Please exercise prudence when utilizing this tool. 
            The numerical outcomes it generates are constructed based on hypothetical and 
            theoretical assumptions. Actual results will be contingent upon real-world 
            interest rates and inflation rates. Past Results Do Not Guarantee Future Outcomes."
            ''')
    
    with st.expander("Data"):
        st.write(data)
if __name__ == "__main__":
    main()
