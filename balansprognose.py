# https://dziganto.github.io/classes/data%20science/linear%20regression/machine%20learning/object-oriented%20programming/python/Understanding-Object-Oriented-Programming-Through-Machine-Learning/
from inkomstenbelasting_helpers import *
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly
# plotly.offline.init_notebook_mode(connected=True)

      
def calculate_year_delta(x, what_to_return):
    """Calculate the difference of income and expenses in a year.


    Args:
        x.fixed_x.monthly_costs (_type_): _description_
        x.monthly_costs_nl (_type_): _description_
        x.various_nl (_type_): _description_
        x.monthly_costs_asia (_type_): _description_
        x.insurance_asia (_type_): _description_
        x.various_asia (_type_): _description_
        x.flight_tickets_asia (_type_): _description_
        x.visas_asia (_type_): _description_
        x.return_flighttickets (_type_): _description_
        x.flighttickets_visa_run (_type_): _description_
        x.salary_gross (int): What is my gross salary per month?
        x.number_of_month_working_nl (int): How many months in the year do I work?
        x.output (bool): _description_
        what_to_return (str): "delta" or "row"

    Returns:
        int : the amount of money that I gain or loose in a year given the arguments
            or
        list[int] = list with the various values  
                    [x.number_of_month_working_nl,x.number_of_months_in_asia,x.number_of_months_in_nl,
                    x.salary_gross_year,transition_payment,extras,  total_income_gross ,pensioen_bijdrage ,
                    nettoloon ,total_income_netto,  expenses_fix,expenses_nl,expenses_asia,  expenses_total, delta]

    """

    
    
     
    number_of_months_in_asia = 12 - (x.number_of_month_working_nl+x.months_nl_non_working) if x.number_of_month_working_nl+x.months_nl_non_working<12 else 0 
    number_of_months_in_nl = x.number_of_month_working_nl+x.months_nl_non_working if x.number_of_month_working_nl+x.months_nl_non_working <12 else 12
    salary_gross_year = x.number_of_month_working_nl * x.proposed_salary_month
            
    # calculate "vakantiedagen" + "vakantietoeslag" + "transitievergoeding"
    transition_payment = (x.number_of_month_working_nl/12)*(1/3)*x.proposed_salary_month if x.number_of_month_working_nl>0 else 0
    if x.calculate_extras:
        extras = salary_gross_year * 0.1 + salary_gross_year *0.08 + transition_payment
    else:
        extras = 0
    total_income_gross = salary_gross_year + extras

    pensioen_bijdrage = calculate_pensioenbijdrage(x.proposed_salary_month, x.number_of_month_working_nl)
    if x.number_of_month_working_nl == 0:
        total_income_netto = 0
    else:
        total_income_netto = calculate_nettoloon_simpel ((total_income_gross- pensioen_bijdrage)/x.number_of_month_working_nl, x.number_of_month_working_nl)
    belastingen = total_income_gross - pensioen_bijdrage-  total_income_netto 
    #total_income_netto =  nettoloon - pensioen_bijdrage
    expenses_fix = (x.fixed_monthly_costs*12)
    expenses_nl =  (x.monthly_costs_nl*number_of_months_in_nl) + x.various_nl + (x.months_nl_non_working*x.monthly_costs_nl_non_working)
    expenses_asia =((x.monthly_costs_asia+x.insurance_asia)*number_of_months_in_asia)

    inkomsten_belasting = calculate_inkomstenbelasting(total_income_gross- pensioen_bijdrage)
    heffingskorting = calculate_heffingskorting(total_income_gross - pensioen_bijdrage)
    arbeidskorting = calculate_arbeidskorting_niet_alle_maanden_werken(x.proposed_salary_month, x.number_of_month_working_nl)
    if inkomsten_belasting - heffingskorting - arbeidskorting >0 :
        belastingen_nieuw = inkomsten_belasting - heffingskorting - arbeidskorting 
        # er is een (klein) verschil tussen belastingen en belastingen_nieuw, verdwijnt als je de extra's en de pensioenbijdrage
        # op 0 zet
    else:
        belastingen_nieuw = 0

    if total_income_gross == 0:
        belastingdruk = 0
    else:
        belastingdruk = round((belastingen/(total_income_gross- pensioen_bijdrage))*100,1)
    if number_of_months_in_asia>0:
        expenses_asia_extra =x.return_flighttickets+x.flighttickets_visa_run  + x.various_asia + (int(number_of_months_in_asia/3)*x.flight_tickets_asia) + x.visas_asia
        expenses_asia +=expenses_asia_extra
    else:
        expenses_asia_extra = 0
        expenses_asia = 0
    expenses_total = expenses_fix + expenses_nl + expenses_asia
    delta = total_income_netto-expenses_total
    

    row = [x.number_of_month_working_nl,delta, number_of_months_in_asia,number_of_months_in_nl,salary_gross_year,transition_payment,extras,  total_income_gross ,pensioen_bijdrage , total_income_netto,  expenses_fix,expenses_nl,expenses_asia,expenses_asia_extra,  expenses_total,  
           belastingen,belastingen_nieuw, inkomsten_belasting, arbeidskorting, heffingskorting,arbeidskorting+ heffingskorting,  belastingdruk]
    if what_to_return == "delta":
        to_return = delta
    else:
        to_return = row
    return to_return


def calculate_pensioenbijdrage(salary_gross_month, number_of_month_working_nl):
    """Calculates pensioenbijdrage

    Args:
        x.salary_gross_month (int): salary
        x.number_of_month_working_nl (int): how many months do I work

    Returns:
        int : amount of pensioenbijdrage
    """

    pensioen_bijdrage_ = (((salary_gross_month*12*1.08)-15935)*0.1265*number_of_month_working_nl/12)
    pensioen_bijdrage = 0 if  pensioen_bijdrage_ < 0 else pensioen_bijdrage_
    return pensioen_bijdrage

def make_graph_values(x):
    """Makes graph of Various values in relation of number of months working. 

    Args:
        income (int): income
    """    
    list_total = []
    for number_of_month_working_nl_ in range (0,13):
        x.number_of_month_working_nl = number_of_month_working_nl_

        row = calculate_year_delta(x, what_to_return="row")
        list_total.append(row)
    columns = ["number_of_month_working_nl", "DELTA","number_of_months_in_asia","number_of_months_in_nl","salary_gross_year_excl_extras","transition_payment","extras","total_income_gross ","pensioen_bijdrage ","total_income_netto","expenses_fix","expenses_nl","expenses_asia","expenses_asia_extra","expenses_total", "werkelijk_te_betalen_belastingen", "werkelijk_te_betalen_belastingen_new", "inkomsten_belasting", "arbeidskorting", "heffingskorting", "totale_korting", "belastingdruk"]
    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    if x.debug == False:
        total_df = total_df[["number_of_month_working_nl", "DELTA"]]
        columns = ["DELTA"]
   
    fig = px.line(total_df, x="number_of_month_working_nl", y=columns, title = f"Various values in relation of number of months working with a monthly income of {x.proposed_salary_month}, year total")
    fig.add_hline(y=0)
    st.plotly_chart(fig)
    #plotly.offline.plot(fig)
    if x.debug:
        total_df_diff = total_df.diff().reset_index()
        fig2 = px.line(total_df_diff, x="index", y=columns, title = f"Various values in relation of number of months working with a monthly income of {x.proposed_salary_month}, difference")
        fig2.add_hline(y=0)
        st.plotly_chart(fig2)
  
    
    #     plotly.offline.plot(fig2)

def main_solver_how_much_salary(x):
    """ Solves how many months I have to work for which salary to have a delta in between 0 and 100
    """    
    list_total=[]
    row=[]
    salaries = list(range (100,25000,1))
    #for x.number_of_month_working_nl in range (0,13):
    for number_of_month_working_nl in range (0,13-x.months_nl_non_working):    
        for salary_gross_month in salaries:

            x.proposed_salary_month = salary_gross_month
            x.number_of_month_working_nl = number_of_month_working_nl
            delta = calculate_year_delta(x, what_to_return= "delta")
            if delta>=x.min_delta and delta <=x.max_delta: 
                row =[number_of_month_working_nl, salary_gross_month, delta]
                list_total.append(row)
                break 
    columns = ["months_working","grensinkomen", "delta"] 
    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    st.subheader("What do I have to earn when I want to work x months?")
    fig = px.line(total_df, x="months_working", y="grensinkomen", title = "Minimal salary when working a certain amount of months")
    #plotly.offline.plot(fig)
    st.plotly_chart(fig)

def main_solver_how_many_months(x):
    """How many months do I have to work?
    """     
    list_total=[]
    salaries = list(range (100,10000,1))
    for salary_gross_month in salaries:
    
        row=[]
        
        #for x.number_of_month_working_nl_ in range (0,122):
        for number_of_month_working_nl_ in range (0,122-x.months_nl_non_working*10):
            x.proposed_salary_month = salary_gross_month
            x.number_of_month_working_nl = number_of_month_working_nl_/10
            number_of_month_working_nl = number_of_month_working_nl_/10
            delta = calculate_year_delta(x, what_to_return="delta")
            if delta>x.min_delta and delta <=x.max_delta:        
                row =[number_of_month_working_nl, salary_gross_month, delta]
                list_total.append(row)
                break
        
    columns = ["months_working","grensinkomen", "delta"] 

    total_df = pd.DataFrame(list_total, columns=columns)#.set_index("months_working")
    
    st.subheader("How many months do I have to work with a certain income?")
    fig = px.line(total_df, x="grensinkomen", y="months_working", title = "Minimal number of months to work with a certain salary")
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)
    x.number_of_month_working_nl = 0
    x.salary_gross_month = 0
   
    
            

def calculate_delta_main(x):
    """_summary_

    Args:
        x.fixed_x.monthly_costs (_type_): _description_
        x.monthly_costs_nl (_type_): _description_
        x.various_nl (_type_): _description_
        x.monthly_costs_asia (_type_): _description_
        x.insurance_asia (_type_): _description_
        x.various_asia (_type_): _description_
        flight_tickets_asia (_type_): _description_
        x.visas_asia (_type_): _description_
        x.return_flighttickets (_type_): _description_
        x.flighttickets_visa_run (_type_): _description_
    """    
    
    list_total=[]
    for number_of_month_working_nl_ in range (0,130-x.months_nl_non_working*10,10):
        row=[number_of_month_working_nl_]
        salaries = list(range (1000,3000,100))
        for salary_gross_month in salaries:
    
            total_capital = 12250
            for y in range(0,1):  # later we want to know what the total capital is after x years 
                x.proposed_salary_month = salary_gross_month
                x.number_of_month_working_nl = number_of_month_working_nl_/10
                delta = calculate_year_delta(x, what_to_return="delta")
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

class CommonParameters:
    """We use use a class to store the common parameters as instance variables. 
       You can then create an instance of the class and pass it to each function that requires those parameters. 
       This approach can make your code more modular and easier to maintain.
    """

    def __init__(self):
        # self, proposed_salary_month,months_nl_non_working,x.monthly_costs_nl_non_working,x.fixed_x.monthly_costs, 
        #      x.monthly_costs_nl,x.various_nl, x.monthly_costs_asia, x.insurance_asia, x.various_asia, flight_tickets_asia, x.visas_asia,
        #      x.return_flighttickets, x.flighttickets_visa_run,extras, debug):
        
        self.proposed_salary_month = st.sidebar.number_input(
            "Proposed salary per month", 0, 10000, 2150
        )
        self.fixed_monthly_costs = st.sidebar.number_input(
            "Fixed monthly costs ('vaste lasten')", 0, 10000, 80
        )
        self.months_nl_non_working = st.sidebar.number_input(
            "Number of months NL non-working", 0, 12, 1
        )
        self.monthly_costs_nl_non_working = st.sidebar.number_input(
            "Monthly costs NL non-working", 0, 10000, 50
        )

        self.monthly_costs_nl = st.sidebar.number_input(
            "Monthly costs NL working", 0, 10000, 350
        )
        self.various_nl = st.sidebar.number_input("Various NL (Total)", 0, 10000, 200)
        self.monthly_costs_asia = st.sidebar.number_input(
            "Monthly costs Asia", 0, 10000, int(350 + (7000 / 36.5))
        )
        self.insurance_asia = st.sidebar.number_input("Insurance Asia (per month)", 0, 10000, 60)
        self.various_asia = st.sidebar.number_input("Various Asia", 0, 10000, 200)
        self.flight_tickets_asia = st.sidebar.number_input(
            "Flight tickets Asia (per 3 mnd)", 0, 10000, 200
        )  # per 3 mnd  = (int(i/3)) * flight_tickets_asia
        self.visas_asia = st.sidebar.number_input("Visas Asia", 0, 10000, 100)
        self.return_flighttickets = st.sidebar.number_input(
            "Return Flights NL-Asia", 0, 10000, 1100
        )
        self.flighttickets_visa_run = st.sidebar.number_input(
            "Flight tickets Visa Run", 0, 10000, 0
        )  # integrated in montly costs asia 400 / flightickets asia
        self.min_delta = 0  # st.sidebar.number_input("min_delta",None,None,0)
        self.max_delta = 100  # st.sidebar.number_input("max_delta",None,None,100)
        self.calculate_extras = st.sidebar.selectbox("Include extras", [True,False], index=0)
        self.debug = st.sidebar.selectbox("Show all lines/graphs", [True,False], index=1)
        self.show_output = False #st.sidebar.selectbox("Show output", [True,False], index=1)
        
        
        
def main():
    st.header("Various calculations about the total amount of money gained/lost vs. number of months working")
    st.write("Calculation to support the article <a href='https://rcsmit.medium.com/how-to-be-on-part-time-retirement-with-a-minimum-wage-job-355e675322c5'>'How to be on a part time retirement'</a> ", unsafe_allow_html=True)
    st.write("'DELTA'  is the change in the amount of money you have from 1/1 to 31/12")
    st.warning("Tax calculation and extras is based on my personal situation. This calculation is only a rough estimate. Use with care!")
    common_param = CommonParameters()

    make_graph_values(common_param)
    calculate_delta_main(common_param)
    main_solver_how_much_salary(common_param)
    main_solver_how_many_months(common_param)


if __name__ == "__main__":
    main()
    
