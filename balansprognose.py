# https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/
from inkomstenbelasting_helpers import *
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly
# plotly.offline.init_notebook_mode(connected=True)

      
def calculate_year_delta(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,
                        salary_gross_month, number_of_month_working_nl, output, what_to_return):
    """Calculate the difference of income and expenses in a year.


    Args:
        fixed_monthly_costs (_type_): _description_
        monthly_costs_nl (_type_): _description_
        various_nl (_type_): _description_
        monthly_costs_asia (_type_): _description_
        insurance_asia (_type_): _description_
        various_asia (_type_): _description_
        flight_tickets_asia (_type_): _description_
        visas_asia (_type_): _description_
        return_flighttickets (_type_): _description_
        flighttickets_visa_run (_type_): _description_
        salary_gross (int): What is my gross salary per month?
        number_of_month_working_nl (int): How many months in the year do I work?
        output (bool): _description_
        what_to_return (str): "delta" or "row"

    Returns:
        int : the amount of money that I gain or loose in a year given the arguments
            or
        list[int] = list with the various values  
                    [number_of_month_working_nl,number_of_months_in_asia,number_of_months_in_nl,
                    salary_gross_year,transition_payment,extras,  total_income_gross ,pensioen_bijdrage ,
                    nettoloon ,total_income_netto,  expenses_fix,expenses_nl,expenses_asia,  expenses_total, delta]

    """

    
    number_of_months_in_asia = 12 - (number_of_month_working_nl+1) if number_of_month_working_nl<12 else 0 
    number_of_months_in_nl = number_of_month_working_nl+1 if number_of_month_working_nl<12 else 12
    salary_gross_year = number_of_month_working_nl * salary_gross_month
            
    # calculate "vakantiedagen" + "vakantietoeslag" + "transitievergoeding"
    transition_payment = (number_of_month_working_nl/12)*(1/3)*salary_gross_month if number_of_month_working_nl>0 else 0
    extras = salary_gross_year * 0.1 + salary_gross_year *0.08 + transition_payment

    total_income_gross = salary_gross_year + extras


    pensioen_bijdrage = calculate_pensioenbijdrage(salary_gross_month, number_of_month_working_nl)
    if number_of_month_working_nl == 0:
        total_income_netto = 0
    else:
        total_income_netto = calculate_nettoloon_simpel ((total_income_gross- pensioen_bijdrage)/number_of_month_working_nl, number_of_month_working_nl)
    belastingen = total_income_gross - pensioen_bijdrage-  total_income_netto 
    #total_income_netto =  nettoloon - pensioen_bijdrage
    expenses_fix = (fixed_monthly_costs*12)
    expenses_nl =  (monthly_costs_nl*number_of_months_in_nl) + various_nl
    expenses_asia =((monthly_costs_asia+insurance_asia)*number_of_months_in_asia)
    if number_of_months_in_asia>0:
        expenses_asia_extra =return_flighttickets+flighttickets_visa_run  + various_asia + (int(number_of_months_in_asia/3)*flight_tickets_asia) + visas_asia
        expenses_asia +=expenses_asia_extra
    else:
        expenses_asia_extra = 0
        expenses_asia = 0
    expenses_total = expenses_fix + expenses_nl + expenses_asia
    delta = total_income_netto-expenses_total
    
    if output:
        n=12
        print(f"number_of_months_in_NL : {number_of_months_in_nl}")
        print(f"number_of_months_in_asia : {number_of_months_in_asia}")
        print(f"{salary_gross_month} x {number_of_month_working_nl} = {salary_gross_month * number_of_month_working_nl/n}")
        print(f"Extras : {extras/n}")
        print(f"Totaal bruto inkomen : {int(total_income_gross/n)}")
        
        print(f"belastingen : {int(total_income_gross) - int(pensioen_bijdrage/n) -int(nettoloon/n)}")
        print(f"Pensioenbijdrage : {int(pensioen_bijdrage/n)}")
        print(f"netto loon : {int(total_income_netto/n)}")
        print(f"uit fixed : {int(expenses_fix/n)}")
        print(f"Montly costs NL : {(monthly_costs_nl*number_of_months_in_nl/n)}")
        print(f"Montly costs asia : {(monthly_costs_asia*number_of_months_in_asia/n)}")
        
        print(f"uit nl : {int(expenses_nl/n)}")
        print(f"uit asia : {int(expenses_asia/n)}")
        print (f"Vliegtickets asia : {(int(number_of_months_in_asia/3)*flight_tickets_asia/n)}")
        print(f"uit totaal : {int(expenses_total/n)}")
        print(f"delta : {int((total_income_netto-expenses_total)/n)}")
        print(f"")
    row = [number_of_month_working_nl,number_of_months_in_asia,number_of_months_in_nl,salary_gross_year,transition_payment,extras,  total_income_gross ,pensioen_bijdrage , total_income_netto,  expenses_fix,expenses_nl,expenses_asia,expenses_asia_extra,  expenses_total, delta]
    if what_to_return == "delta":
        to_return = delta
    else:
        to_return = row
    return to_return

def calculate_pensioenbijdrage(salary_gross_month, number_of_month_working_nl):
    """Calculates pensioenbijdrage

    Args:
        salary_gross_month (int): salary
        number_of_month_working_nl (int): how many months do I work

    Returns:
        int : amount of pensioenbijdrage
    """

    pensioen_bijdrage_ = (((salary_gross_month*12*1.08)-15935)*0.1265*number_of_month_working_nl/12)
    pensioen_bijdrage = 0 if  pensioen_bijdrage_ < 0 else pensioen_bijdrage_
    return pensioen_bijdrage

def main_solver_how_much_salary(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,min_delta, max_delta):
    """ Solves how many months I have to work for which salary to have a delta in between 0 and 100
    """    
    list_total=[]
    row=[]
    salaries = list(range (100,25000,1))
    for number_of_month_working_nl in range (0,13):
        for salary_gross_month in salaries:
            delta = calculate_year_delta(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,
                                         salary_gross_month=salary_gross_month,number_of_month_working_nl=number_of_month_working_nl, output=False, what_to_return="delta")
            if delta>=min_delta and delta <=max_delta: 
                row =[number_of_month_working_nl, salary_gross_month, delta]
                list_total.append(row)
                break 
    columns = ["months_working","grensinkomen", "delta"] 
    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    st.subheader("What do I have to earn when I want to work x months?")
    fig = px.line(total_df, x="months_working", y="grensinkomen", title = "Minimal salary when working a certain amount of months")
    #plotly.offline.plot(fig)
    st.plotly_chart(fig)
def make_graph_values(income,fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run):
    """Makes graph of Various values in relation of number of months working. 

    Args:
        income (int): income
    """    
    list_total = []
    for number_of_month_working_nl in range (0,13):
        row = calculate_year_delta(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,
                                    income, number_of_month_working_nl, False, what_to_return="row")
        list_total.append(row)
    columns = ["number_of_month_working_nl", "number_of_months_in_asia","number_of_months_in_nl","salary_gross_year_excl_extras","transition_payment","extras","  total_income_gross ","pensioen_bijdrage ","total_income_netto","expenses_fix","expenses_nl","expenses_asia","expenses_asia_extra","expenses_total"," delta"]
    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    fig = px.line(total_df, x="number_of_month_working_nl", y=columns, title = f"Various values in relation of number of months working with a monthly income of {income}, year total")
    fig.add_hline(y=0)
    st.plotly_chart(fig)
    #plotly.offline.plot(fig)

    total_df_diff = total_df.diff().reset_index()
    fig2 = px.line(total_df_diff, x="index", y=columns, title = f"Various values in relation of number of months working with a monthly income of {income}, difference")
    fig2.add_hline(y=0)
    st.plotly_chart(fig2)
   
    
    #     plotly.offline.plot(fig2)


def main_solver_how_many_months(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,min_delta, max_delta):
    """How many months do I have to work?
    """     
    list_total=[]
    salaries = list(range (100,10000,1))
    for salary_gross_month in salaries:
    
        row=[]
        
        for number_of_month_working_nl_ in range (0,122):
            number_of_month_working_nl = number_of_month_working_nl_/10
            delta = calculate_year_delta(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,
                                         salary_gross_month=salary_gross_month,number_of_month_working_nl=number_of_month_working_nl, output=False, what_to_return="delta")
            if delta>min_delta and delta <=max_delta:        
                row =[number_of_month_working_nl, salary_gross_month, delta]
                list_total.append(row)
                break
        
    columns = ["months_working","grensinkomen", "delta"] 

    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    
    st.subheader("How many months do I have to work with a certain income?")
    fig = px.line(total_df, x="grensinkomen", y="months_working", title = "Minimal number of months to work with a certain salary")
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)
   

def calculate_delta_main(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run):
    """_summary_

    Args:
        fixed_monthly_costs (_type_): _description_
        monthly_costs_nl (_type_): _description_
        various_nl (_type_): _description_
        monthly_costs_asia (_type_): _description_
        insurance_asia (_type_): _description_
        various_asia (_type_): _description_
        flight_tickets_asia (_type_): _description_
        visas_asia (_type_): _description_
        return_flighttickets (_type_): _description_
        flighttickets_visa_run (_type_): _description_
    """    
    
    list_total=[]
    for number_of_month_working_nl_ in range (0,130,10):
        number_of_month_working_nl = number_of_month_working_nl_/10
        row=[number_of_month_working_nl]
        salaries = list(range (1500,2300,100))
        for salary_gross_month in salaries:
    
            total_capital = 12250
            for y in range(0,1):  # later we want to know what the total capital is after x years 
                delta = calculate_year_delta(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run, salary_gross_month=salary_gross_month,number_of_month_working_nl=number_of_month_working_nl, output=False, what_to_return="delta")
                total_capital = total_capital +delta
                row.append(int(delta))
        list_total.append(row)
     
    columns = ["months_working"] + salaries

    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    st.subheader ("Yearly change in total capital")
    # total_df_diff = total_df.diff()
    
    # print (total_df_diff)

    fig = px.line(total_df, x="months_working", y=salaries, title = "Change in total capital vs months working with different monthly wages")
    fig.add_hline(y=0)
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)
    #st.write (total_df)
    
def main():
    st.header("Various calculations about the total amount of money gained/lost vs. number of months working")
    st.write("Calculation to support the article <a href='https://rcsmit.medium.com/how-to-be-on-part-time-retirement-with-a-minimum-wage-job-355e675322c5'>'How to be on a part time retirement'</a> ", unsafe_allow_html=True)
    
    st.warning("Tax calculation and extras is based on my personal situation. This calculation is only a rough estimate. Use with care!")
    proposed_salary_month =st.sidebar.number_input("Proposed salary per month",0,10000,2150)
    fixed_monthly_costs = st.sidebar.number_input("Fixed monthly costs ('vaste lasten')",0,10000,80)
    monthly_costs_nl =  st.sidebar.number_input("Monthly costs NL",0,10000,350)
    various_nl =  st.sidebar.number_input("Various NL (Total)",0,10000,200)
    monthly_costs_asia =  st.sidebar.number_input("Monthly costs Asia",0,10000,int(350 + (7000/36.5)))
    insurance_asia =  st.sidebar.number_input("Insurance Asia (per month)",0,10000,60)
    various_asia = st.sidebar.number_input("Various Asia",0,10000,200)
    flight_tickets_asia = st.sidebar.number_input("Flight tickets Asia (per 3 mnd)",0,10000,200) # per 3 mnd  = (int(i/3)) * flight_tickets_asia
    visas_asia =  st.sidebar.number_input("Visas Asia",0,10000,100)
    return_flighttickets =  st.sidebar.number_input("Return Flights NL-Asia",0,10000,1100)
    flighttickets_visa_run =  st.sidebar.number_input("Flight tickets Visa Run",0,10000,0) # integrated in montly costs asia 400 / flightickets asia
    min_delta = 0 # st.sidebar.number_input("min_delta",None,None,0) 
    max_delta = 100# st.sidebar.number_input("max_delta",None,None,100) 

    make_graph_values(proposed_salary_month,fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run)
    
    calculate_delta_main(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run)

    #
    # print(calculate_nettoloon_simpel (28563))
    # print(calculate_nettoloon_simpel (31160))
    # print(calculate_pensioenbijdrage(2150, 11))
    # print(calculate_pensioenbijdrage(2150, 12))
    # print(calculate_year_delta(salary_gross_month=2150,number_of_month_working_nl=10, output=True, what_to_return="delta"))
    # # 
    # print(calculate_year_delta(salary_gross_month=2150,number_of_month_working_nl=11, output=True, what_to_return="delta"))
    # print(calculate_year_delta(salary_gross_month=2150,number_of_month_working_nl=12, output=True, what_to_return="delta"))
    # print(calculate_year_delta(salary_gross_month=2150,number_of_month_working_nl=6, output=True))
    #main()
    main_solver_how_much_salary(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,min_delta, max_delta)

    main_solver_how_many_months(fixed_monthly_costs, monthly_costs_nl,various_nl, monthly_costs_asia, insurance_asia, various_asia, flight_tickets_asia, visas_asia,return_flighttickets, flighttickets_visa_run,min_delta, max_delta)


if __name__ == "__main__":
    main()
    
