from inkomstenbelasting_helpers import *
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly

# plotly.offline.init_notebook_mode(connected=True)


def calculate_year_delta(common_params):
    
    for key, value in kwargs.items():    
        exec(f"{key} = {repr(value)}", globals())
       
        print (key,value)

    number_of_months_in_asia = (
        12 - (number_of_month_working_nl + months_nl_non_working)
        if number_of_month_working_nl + months_nl_non_working < 12
        else 0
    )
    number_of_months_in_nl = (
        number_of_month_working_nl + months_nl_non_working
        if number_of_month_working_nl + months_nl_non_working < 12
        else 12
    )
    salary_gross_year = number_of_month_working_nl * salary_gross_month

    # calculate "vakantiedagen" + "vakantietoeslag" + "transitievergoeding"
    transition_payment = (
        (number_of_month_working_nl / 12) * (1 / 3) * salary_gross_month
        if number_of_month_working_nl > 0
        else 0
    )
    if extras:
        extras = salary_gross_year * 0.1 + salary_gross_year * 0.08 + transition_payment
    else:
        extras = 0
    total_income_gross = salary_gross_year + extras

    pensioen_bijdrage = 0
    if number_of_month_working_nl == 0:
        total_income_netto = 0
    else:
        total_income_netto = calculate_nettoloon_simpel(
            (total_income_gross - pensioen_bijdrage) / number_of_month_working_nl,
            number_of_month_working_nl,
        )
    belastingen = total_income_gross - pensioen_bijdrage - total_income_netto
    # total_income_netto =  nettoloon - pensioen_bijdrage
    expenses_fix = fixed_monthly_costs * 12
    expenses_nl = (
        (monthly_costs_nl * number_of_months_in_nl)
        + various_nl
        + (months_nl_non_working * monthly_costs_nl_non_working)
    )
    expenses_asia = (monthly_costs_asia + insurance_asia) * number_of_months_in_asia

    inkomsten_belasting = calculate_inkomstenbelasting(
        total_income_gross - pensioen_bijdrage
    )
    heffingskorting = calculate_heffingskorting(total_income_gross - pensioen_bijdrage)
    arbeidskorting = calculate_arbeidskorting_niet_alle_maanden_werken(
        salary_gross_month, number_of_month_working_nl
    )
    if inkomsten_belasting - heffingskorting - arbeidskorting > 0:
        belastingen_nieuw = inkomsten_belasting - heffingskorting - arbeidskorting
        # er is een (klein) verschil tussen belastingen en belastingen_nieuw, verdwijnt als je de extra's en de pensioenbijdrage
        # op 0 zet
    else:
        belastingen_nieuw = 0
    if total_income_gross == 0:
        belastingdruk = 0
    else:
        belastingdruk = round(
            (belastingen / (total_income_gross - pensioen_bijdrage)) * 100, 1
        )
    if number_of_months_in_asia > 0:
        expenses_asia_extra = (
            return_flighttickets
            + flighttickets_visa_run
            + various_asia
            + (int(number_of_months_in_asia / 3) * flight_tickets_asia)
            + visas_asia
        )
        expenses_asia += expenses_asia_extra
    else:
        expenses_asia_extra = 0
        expenses_asia = 0
    expenses_total = expenses_fix + expenses_nl + expenses_asia
    delta = total_income_netto - expenses_total

    row = [
        number_of_month_working_nl,
        number_of_months_in_asia,
        number_of_months_in_nl,
        salary_gross_year,
        transition_payment,
        extras,
        total_income_gross,
        pensioen_bijdrage,
        total_income_netto,
        expenses_fix,
        expenses_nl,
        expenses_asia,
        expenses_asia_extra,
        expenses_total,
        delta,
        belastingen,
        belastingen_nieuw,
        inkomsten_belasting,
        arbeidskorting,
        heffingskorting,
        arbeidskorting + heffingskorting,
        belastingdruk,
    ]
    if what_to_return == "delta":
        to_return = delta
    else:
        to_return = row
    return to_return


def make_graph_values(common_params):
    
    for key, value in kwargs.items():    
        exec(f"{key} = {repr(value)}", globals())
       
        print (key,value)
    income = proposed_salary_month
    list_total=[]

    for number_of_month_working_nl in range(0, 13):
        row = calculate_year_delta(kwargs)
        list_total.append(row)
    columns = [
        "number_of_month_working_nl",
        "number_of_months_in_asia",
        "number_of_months_in_nl",
        "salary_gross_year_excl_extras",
        "transition_payment",
        "extras",
        "  total_income_gross ",
        "pensioen_bijdrage ",
        "total_income_netto",
        "expenses_fix",
        "expenses_nl",
        "expenses_asia",
        "expenses_asia_extra",
        "expenses_total",
        "delta",
        "werkelijk_te_betalen_belastingen",
        "werkelijk_te_betalen_belastingen_new",
        "inkomsten_belasting",
        "arbeidskorting",
        "heffingskorting",
        "totale_korting",
        "belastingdruk",
    ]
    total_df = pd.DataFrame(list_total, columns=columns)  # .set_index("months_working")
    if debug == False:
        total_df = total_df[["number_of_month_working_nl", "delta"]]
        columns = ["delta"]
    fig = px.line(
        total_df,
        x="number_of_month_working_nl",
        y=columns,
        title=f"Various values in relation of number of months working with a monthly income of {income}, year total",
    )
    fig.add_hline(y=0)
    st.plotly_chart(fig)


def calculate_delta_main(common_params):
    
    for number_of_month_working_nl_ in range(0, 130 - months_nl_non_working * 10, 10):
        number_of_month_working_nl = number_of_month_working_nl_ / 10
        row = [number_of_month_working_nl]
        salaries = list(range(1000, 3000, 100))
        for salary_gross_month in salaries:

            total_capital = 12250
            for y in range(
                0, 1
            ):  # later we want to know what the total capital is after x years
                delta = calculate_year_delta(kwargs)
                total_capital = total_capital + delta
                row.append(int(delta))
        list_total.append(row)

    columns = ["months_working"] + salaries

    total_df = pd.DataFrame(list_total, columns=columns)  # .set_index("months_working")
    st.subheader("Yearly change in total capital")

    fig = px.line(
        total_df,
        x="months_working",
        y=salaries,
        title="Change in total capital vs months working with different monthly wages",
    )
    fig.add_hline(y=0)

    st.plotly_chart(fig)



def extra_routine(x):
    st.write(f"Total income2 = {x.proposed_salary_month} * {x.months_nl_non_working} =  {x.proposed_salary_month * x.months_nl_non_working}")


def total_income(x):
    st.write(f"Total income = {x.proposed_salary_month} * {x.months_nl_non_working} =  {x.proposed_salary_month * x.months_nl_non_working}")
    extra_routine(x)
def main():
    st.header(
        "Various calculations about the total amount of money gained/lost vs. number of months working"
    )
    st.write(
        "Calculation to support the article <a href='https://rcsmit.medium.com/how-to-be-on-part-time-retirement-with-a-minimum-wage-job-355e675322c5'>'How to be on a part time retirement'</a> ",
        unsafe_allow_html=True,
    )

    st.warning(
        "Tax calculation and extras is based on my personal situation. This calculation is only a rough estimate. Use with care!"
    )
    
    extras = False
    debug = True
    common_params = CommonParameters()
        # proposed_salary_month = proposed_salary_month,
        # months_nl_non_working = months_nl_non_working,
        # monthly_costs_nl_non_working = monthly_costs_nl_non_working,
        # fixed_monthly_costs = fixed_monthly_costs,
        # monthly_costs_nl = monthly_costs_nl,
        # various_nl = various_nl,
        # monthly_costs_asia = monthly_costs_asia,
        # insurance_asia = insurance_asia,
        # various_asia = various_asia,
        # flight_tickets_asia = flight_tickets_asia,
        # visas_asia = visas_asia,
        # return_flighttickets = return_flighttickets,
        # flighttickets_visa_run = flighttickets_visa_run,
        # extras = extras,
        # debug = debug)
    
    total_income(common_params)
    #make_graph_values(common_params)

def test():

    program = 'a = 5\nb=10\nprint("Sum =", a+b)'
    exec(program)
    my_dict = {"name": "John", "age": 30, "location": "USA"}
   

    for key, value in my_dict.items():
        
        myTemplate = "{} = \"{}\""
        
        statement = myTemplate.format(key, value)
        print (statement)
        exec(statement, globals()) 

        print (key,value)
        #exec(f"{key} = {repr(value)}")
    print('test')
    
    print(location)     # prints "John"
   
if __name__ == "__main__":
    main()
    #test()