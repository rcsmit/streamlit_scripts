import plotly.express as px
import pandas as pd
import streamlit as st

def main():

    st.title('Pension Calculation')

    i_values = []
    interest_values = []
    balance_values = []
    annual_contribution_values = []
    annual_shortfall= []
    initial_one_time_contribution = st.sidebar.number_input("Initial One-Time Contribution:", value=0)
    monthly_contribution_original = st.sidebar.number_input("Annual Contribution:", value=200)
    
    annual_return_rate = st.sidebar.number_input("Annual Return Rate (%):", value=2)
    current_age = st.sidebar.number_input("Current Age:", value=46)
    retirement_age = st.sidebar.number_input("Retirement Age:", value=69)
    st.sidebar.write(f"Years to Go: {retirement_age - current_age}")
    inflation =  st.sidebar.number_input("Average Annual inflation Rate (%):", value=1.9) # 1.9 volgens https://descryptor.org/opinie/2019/07/25/Donald-Duck-Index
                                                                                            # 5.38 volgens https://indeflatie.nl/inflatie-in-nederland-afgelopen-25-jaar
    monthly_pension_without_reduction_original = st.sidebar.number_input("Monthly Pension without Reduction:", value=1458)
    years_shortfall = st.sidebar.number_input("Years of Shortfall:", value=7)
    st.sidebar.write(f"Shortfall per month orignal: {round(years_shortfall * 0.02 * monthly_pension_without_reduction_original)}")

    additional_monthly_need = st.sidebar.number_input("Additional Monthly Need (current price level):", value=200)
    # opgebouwd_per_jaar = st.sidebar.number_input("Pensioen opgebouwd per jaar bruto:", value = 749) #62 per maand
    expected_life_expectancy =  st.sidebar.number_input("Expected Life Expectancy:", value=81.4)
    st.sidebar.write(f"Number of years to sustain {round(expected_life_expectancy - retirement_age,1)}")
    st.sidebar.write(f"ratio : {round((expected_life_expectancy - retirement_age) / (retirement_age - current_age),1)}")
    windfall_1_year =   st.sidebar.number_input("Windfall 1 (Year):", value=5)
    windfall_1_amount = st.sidebar.number_input("Windfall 1 (Amount):", value=5000)

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
    for i in range(0, years_until_retirement):
        interest = balance * ((annual_return_rate) / 100)

        balance += interest + annual_contribution_corrected
        annual_contribution_corrected = annual_contribution_corrected * (100+inflation)/100
        # Check if the current index is in the special_years list
        if i in special_years:
            # Find the corresponding amount to add based on the year
            special_index = special_years.index(i)
            special_addition = special_amounts[special_index]
            balance += special_addition
        i_values.append(i)
        interest_values.append(interest)
        balance_values.append(int(balance))
        annual_contribution_values.append(annual_contribution_corrected)
        annual_shortfall.append(None)
    end_balance_at_retirement = balance
    annual_shortfall_corrected = annual_shortfall_original * (((100+inflation)/100) ** years_until_retirement)
    years_after_retirement = int(end_balance_at_retirement / annual_shortfall_corrected)

    st.write (years_after_retirement)
    for j in range(years_until_retirement,100-current_age+1):# years_until_retirement+years_after_retirement):
        interest = balance * ((annual_return_rate) / 100)
        balance += interest 
        balance -= annual_shortfall_corrected
        annual_shortfall_corrected = annual_shortfall_corrected  * ((100+inflation)/100) 
        i_values.append(j)
        interest_values.append(interest)
        balance_values.append(int(balance))  
        annual_contribution_values.append(0)
        annual_shortfall.append(annual_shortfall_corrected)

    data = pd.DataFrame({'i': i_values, 'interest': interest_values, 'annual_contribution': annual_contribution_values, 'annual_shortfall':annual_shortfall, 'balance': balance_values})
    data["age"] = data["i"] + current_age
    st.write(data)
    fig = px.line(data, x='age', y=[ 'balance'], title='Balance Over Time')
    fig.update_traces(mode='lines+markers')
    fig.add_hline(y=0,  line_color="orange")
    
    fig.add_vline(x=retirement_age, line_dash="dash", line_color="green", annotation_text="Retirement Age", annotation_position="bottom left")
    fig.add_vline(x=expected_life_expectancy, line_dash="dash", line_color="black", annotation_text="Expected Life Expectancy", annotation_position="bottom left")
    st.plotly_chart(fig)

    years_of_sustainability = round(end_balance_at_retirement / annual_shortfall_corrected, 1)
    # Find the year when the balance passes 0
    zero_balance_year = data[data['balance'] > 0]['age'].max()
    st.info (f"With your end balance, you can sustain for  until age {zero_balance_year}.") 
    if zero_balance_year > expected_life_expectancy:
        st.success (f"You have money for {round(zero_balance_year - expected_life_expectancy)} years more based on expected life expectancy.")

    else:
        st.warning (f"You have money for {round(expected_life_expectancy - zero_balance_year )} years less based on expected life expectancy.")

if __name__ == "__main__":
    main()
