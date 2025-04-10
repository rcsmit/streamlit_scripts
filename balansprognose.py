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
        x (oject) : various variables
        what_to_return (str): "delta" or "row"
        output (bool) : Show calculation
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
    if x.calculate_extras:
        transition_payment = (x.number_of_month_working_nl/12)*(1/3)*x.proposed_salary_month if x.number_of_month_working_nl>0 else 0
        extras = salary_gross_year * 0.1 + salary_gross_year *0.08 + transition_payment
    else:
        transition_payment = 0
        extras = 0 
    total_income_gross = salary_gross_year + extras
    pensioen_bijdrage = calculate_pensioenbijdrage(x.proposed_salary_month, x.number_of_month_working_nl)
    if x.number_of_month_working_nl == 0:
        total_income_netto = 0
    else:
        total_income_netto = calculate_nettoloon_simpel ((total_income_gross- pensioen_bijdrage)/x.number_of_month_working_nl, x.number_of_month_working_nl)
    belastingen = total_income_gross - pensioen_bijdrage-  total_income_netto 

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

    elif what_to_return =="row_with_output":
        if x.when_output == x.number_of_month_working_nl:
            st.info(f"Calculation for working {int(x.number_of_month_working_nl)} months ")
            
                    
            st.code(f"""Number of months working :      {f'{x.number_of_month_working_nl:>10,.0f}'.replace(',', ' ')}
Number of months in Asia :      {f'{number_of_months_in_asia:>10,.0f}'.replace(',', ' ')}
Number of months with parents : {f'{x.months_nl_non_working:>10,.0f}'.replace(',', ' ')}

INCOME
Gross Annual Salary*            {f'{int(salary_gross_year):>10,.0f}'.replace(',', ' ')}
+ Extras                        {f'{int(extras):>10,.0f}'.replace(',', ' ')}
  (transition payment/reservations)
Total gross income              {f'{int(total_income_gross):>10,.0f}'.replace(',', ' ')}
- Pension contribution          {f'{int(pensioen_bijdrage):>10,.0f}'.replace(',', ' ')}
- Taxes                         {f'{int(belastingen):>10,.0f}'.replace(',', ' ')}
                                 ---------
Total Net Income                                                {f'{int(total_income_netto):>10,.0f}'.replace(',', ' ')}

EXPENSES
Fixed expenses (all year long)* {f'{int(expenses_fix):>10,.0f}'.replace(',', ' ')}
Monthly expenses Europe         {f'{int(x.monthly_costs_nl*number_of_months_in_nl):>10,.0f}'.replace(',', ' ')}
                   (working)*
Miscellaneous NL Expenses       {f'{int(x.various_nl):>10,.0f}'.replace(',', ' ')}
Non-working Months in NL*       {f'{int(x.monthly_costs_nl_non_working * x.months_nl_non_working):>10,.0f}'.replace(',', ' ')}
                                 ---------
Total expenses Europe                             {f'{int(expenses_nl):>10,.0f}'.replace(',', ' ')}

Monthly expenses Asia*          {f'{int((x.monthly_costs_asia+x.insurance_asia)*number_of_months_in_asia):>10,.0f}'.replace(',', ' ')}
Travel insurance*               {f'{int((x.insurance_asia)*number_of_months_in_asia):>10,.0f}'.replace(',', ' ')}
Return flight tickets           {f'{int(x.return_flighttickets):>10,.0f}'.replace(',', ' ')}
Flight tickets(visa runs)       {f'{int(x.flighttickets_visa_run):>10,.0f}'.replace(',', ' ')}
Various Asia                    {f'{int(x.various_asia):>10,.0f}'.replace(',', ' ')}
Flights within Asia*            {f'{int((int(number_of_months_in_asia/3)*x.flight_tickets_asia)):>10,.0f}'.replace(',', ' ')}
Visas in Asia                   {f'{int(x.visas_asia):>10,.0f}'.replace(',', ' ')}
                                   -------
Total Expenses Asia                              {f'{int(expenses_asia):>10,.0f}'.replace(',', ' ')}

Total Expenses                                                  {f'{int(expenses_total):>10,.0f}'.replace(',', ' ')}
                                                                ==========
Change of assets in the year                                    {f'{int(delta):>10,.0f}'.replace(',', ' ')}

=== Breakdown of Monthly Costs ===
Gross Annual Salary               {f'{x.proposed_salary_month:>10,.0f}'.replace(',', ' ')} * {f'{int(x.number_of_month_working_nl):>2}'} = {f'{int(x.proposed_salary_month * x.number_of_month_working_nl):>10,.0f}'.replace(',', ' ')}
Fixed expenses (all year long)    {f'{x.fixed_monthly_costs:>10,.0f}'.replace(',', ' ')} * {f'{12:>2}'} = {f'{x.fixed_monthly_costs*12:>10,.0f}'.replace(',', ' ')}
Monthly expenses Europe (working) {f'{int(x.monthly_costs_nl):>10,.0f}'.replace(',', ' ')} * {f'{int(number_of_months_in_nl):>2}'} = {f'{int(x.monthly_costs_nl*number_of_months_in_nl):>10,.0f}'.replace(',', ' ')}
Non-working Months in NL          {f'{int(x.monthly_costs_nl_non_working):>10,.0f}'.replace(',', ' ')} * {f'{x.months_nl_non_working:>2}'} = {f'{int(x.monthly_costs_nl_non_working * x.months_nl_non_working):>10,.0f}'.replace(',', ' ')}
Monthly expenses Asia             {f'{int(x.monthly_costs_asia):>10,.0f}'.replace(',', ' ')} * {f'{int(number_of_months_in_asia):>2}'} = {f'{int((x.monthly_costs_asia)*number_of_months_in_asia):>10,.0f}'.replace(',', ' ')}
Travel insurance                  {f'{int(x.insurance_asia):>10,.0f}'.replace(',', ' ')} * {f'{int(number_of_months_in_asia):>2}'} = {f'{int((x.insurance_asia)*number_of_months_in_asia):>10,.0f}'.replace(',', ' ')}
Flights within Asia :             {f'{x.flight_tickets_asia:>10,.0f}'.replace(',', ' ')} * {f'{int((int(number_of_months_in_asia/3))):>2}'} = {f'{int((int(number_of_months_in_asia/3) * x.flight_tickets_asia)):>10,.0f}'.replace(',', ' ')}
""")

        to_return = row
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

def make_graph_values(x):
    """Makes graph of Various values in relation of number of months working. 

    Args:
        x : object with various variables
    """    
    list_total = []
    for number_of_month_working_nl_ in range (0,121):
        x.number_of_month_working_nl = number_of_month_working_nl_ / 10

        row = calculate_year_delta(x, what_to_return="row_with_output")
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
        # plotly.offline.plot(fig2)

def main_solver_how_much_salary(x):
    """ Solves how many months I have to work for which salary to have a delta in between 0 and 100
    """    
    list_total=[]
    row=[]
    salaries = list(range (100,25000,1))
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
    """Calculate the delta with various salaries over various months working

    Args:
        x:
        when_output: the month in which we want output. 0 for never output

    """    
    
    list_total=[]
    for number_of_month_working_nl_ in range (0,130-x.months_nl_non_working*10,5):
        row=[number_of_month_working_nl_/10]
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
    fig = px.line(total_df, x="months_working", y=salaries, title = "Change in total capital vs months working with different monthly wages")
    fig.add_hline(y=0)
    # plotly.offline.plot(fig)
    st.plotly_chart(fig)

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
            "Proposed gross salary per month", 0, 10000, 2300
        )
        self.calculate_extras = st.sidebar.selectbox("Include extras (vak.geld/-dgn, transtieverg.)", [True,False], index=0)
        
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
            "Monthly costs Europe working", 0, 10000, 450
        )
        self.various_nl = st.sidebar.number_input("Various NL (Total)", 0, 10000, 200)
        self.monthly_costs_asia = st.sidebar.number_input(
            "Monthly costs Asia", 0, 10000, 1000)
        
        self.insurance_asia = st.sidebar.number_input("Travel insurance Asia (per month)", 0, 10000, 75)
        self.various_asia = st.sidebar.number_input("Various Asia total", 0, 10000, 200)
        self.flight_tickets_asia = st.sidebar.number_input(
            "Flight tickets Asia (per 3 mnd)", 0, 10000, 0
        )  # per 3 mnd  = (int(i/3)) * flight_tickets_asia
        self.visas_asia = st.sidebar.number_input("Visas Asia", 0, 10000,0)
        self.return_flighttickets = st.sidebar.number_input(
            "Return Flights NL-Asia", 0, 10000, 1200
        )
        self.flighttickets_visa_run = st.sidebar.number_input(
            "Flight tickets Visa Runs (total)", 0, 10000, 0
        )  # integrated in montly costs asia 400 / flightickets asia
        self.min_delta = 0  # st.sidebar.number_input("min_delta",None,None,0)
        self.max_delta = 100  # st.sidebar.number_input("max_delta",None,None,100)
        self.debug = st.sidebar.selectbox("Show all lines/graphs", [True,False], index=1)
        self.show_output = False #st.sidebar.selectbox("Show output", [True,False], index=1)
        self.when_output = st.sidebar.number_input("Show calculation for x months (-1 = never)", -1,12,-1)
             
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
    
