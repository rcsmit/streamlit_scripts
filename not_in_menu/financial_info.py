 # """Script to manage a double entry accounting system. Made with the help of Chat-GPT

# In a double-entry accounting system, every financial transaction is recorded in at least 
# two different accounts: one account is credited and the other account is debited. The debited account 
# is the one that receives the payment, and the credited account is the one that makes the payment.

# """

from datetime import datetime, date,  timedelta
import pandas as pd
import traceback
from collections import defaultdict
import calendar
import streamlit as st
import datetime

import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots

debug = False
streamlit = True
def print_x(string):
    if streamlit:
        st.write(string)
    else:
        print(string)
   
def read():
    """Read the data. A dummy file can be found at 
     https://github.com/rcsmit/python_scripts_rcsmit/blob/master/input/masterfinance_2023_dummy.xlsm
     Attention : de "KRUIS"-accounts are not 0 because the numbers are randomized

    Returns:
        df : The dataframe with the data
    """    
    file = r"C:\Users\rcxsm\Documents\xls\masterfinance_2023.xlsx"
    #file = r"C:\Users\rcxsm\Documents\python_scripts\python_scripts_rcsmit\input\masterfinance_2023_dummy.xlsm"
    # to be found on
    try:
        df = pd.read_excel (file,
                            sheet_name= "INVOER",
                            header=0,
                            usecols= "a,b,g,h, k,l,m,n",
                            
                            names=["id","source","datum","bedrag", "description","income_expenses","main_category","category"],)
        
        df.datum=pd.to_datetime(df.datum,errors='coerce', dayfirst=True)
        
    except Exception as e:
        print_x("error met laden")
        print_x(f"{e}")
        print_x(traceback.format_exc())
        st.stop()

    df['jaar']=df['datum'].dt.strftime('%Y')
    df['maand']=df['datum'].dt.strftime('%m')
    #df['invbedrag']= df['bedrag']* -1
    df['maand_']=df['datum'].dt.strftime('%Y-%m')
    #df = df[df['description'] != "starting balance_2023"]
    
    # df = df[(df['jaar'] == "2023")| (df['jaar'] == "2022") | (df['description'] == "starting balance_2022")]
    
    # de grootboeken kloppen niet als je 2021 toevoegt
    df = df[(df['jaar'] == "2024")|(df['jaar'] == "2023")| (df['jaar'] == "2022") | (df['description'] == "starting balance_2022")]
    #df = df[(df['jaar'] == "2023")| (df['jaar'] == "2022") |(df['jaar'] == "2021")| (df['description'] != "STARTING_BALANCE_2022")]
    # Convert all text data in the DataFrame to uppercase
    #df = df.apply(lambda x: x.str.upper() if x.dtype == "object" else x)
    df = df.fillna(0)

    new_table = []
    for index, row in df.iterrows():
        income_expenses_temp = row.income_expenses
        main_cat_temp = row.main_category
        category_temp = row.category
        line_temp = [income_expenses_temp, main_cat_temp, category_temp ]
        if line_temp in new_table:
            pass
        else:
            new_table.append(line_temp)
   
   
    df_cat = pd.DataFrame(new_table, columns= ["income_expenses", "main_cat", "category" ]) 
    return df, df_cat

class Account:
    """Handle the various accounts
    """    
    def __init__(self, name):
        self.name = name
        self.balance = 0
        self.transactions = []

    def debit(self, amount):
        self.balance += amount

        

    def credit(self, amount):
        self.balance -= amount

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def __str__(self):
        return f"{self.name} - Balance: {round(self.balance,2)}"


class Transaction:
    """Handle the transactions
    """    
    def __init__(self, date, debit_account, credit_account, amount, description):
        self.date = date
        self.debit_account = debit_account
        self.credit_account = credit_account
        self.amount = amount
        self.description = description

class Ledger:
    """Handle the ledgers (=grootboeken)
    """    
    def __init__(self):
        self.accounts = []
        self.transactions = []
    
    def add_account(self, account):
        self.accounts.append(account)
    
    def add_transaction(self, date, debit_account_name, credit_account_name, amount, description):
        debit_account = next((a for a in self.accounts if a.name == debit_account_name), None)
        credit_account = next((a for a in self.accounts if a.name == credit_account_name), None)
        if not debit_account or not credit_account:
            raise ValueError(f"Invalid account name {debit_account_name}, {credit_account_name}")
        transaction = Transaction(date, debit_account, credit_account, amount, description)
        debit_account.debit(amount)
        credit_account.credit(amount)
        debit_account.add_transaction(transaction)
        credit_account.add_transaction(transaction)
        self.transactions.append(transaction)
        
    def print_x_trial_balance(self):
        """Make the trial balance over all transactions
        """       
        trial_balance =[] 
        total_debits = 0
        total_credits = 0
        for account in self.accounts:
            total_debits += sum(t.amount for t in account.transactions if t.debit_account == account)
            total_credits += sum(t.amount for t in account.transactions if t.credit_account == account)
            debits = sum(t.amount for t in account.transactions if t.debit_account == account)
            credits = sum(t.amount for t in account.transactions if t.credit_account == account)

            trial_balance.append([account.name,round(debits,2),round(credits,2)])
        df = pd.DataFrame(trial_balance, columns = ["ACCOUNT", "DEBIT", "CREDIT"])
        print_x (df)
        print_x(f"Total Debits: {round(total_debits,2)}")
        print_x(f"Total Credits: {round(total_credits,2)}")
    
    def ask_trial_balance_date(self, asked_account, date_given):
        """Return the value on the trial balance of a certain account on a certain date. 
        Used to make a monthly pivot table

        Args:
            asked_acount (str): the asked account
            date_given (date): date

        Returns:
            float: the amount on the trial balance
        """
        total_debits = 0
        total_credits = 0
        to_return = 0

        for account in self.accounts:
            relevant_transactions = [t for t in account.transactions 
                                    if t.date <= date_given and (t.debit_account == account or t.credit_account == account) and account.name == asked_account]
            d = sum(t.amount for t in relevant_transactions if t.debit_account == account)
            c = sum(t.amount for t in relevant_transactions if t.credit_account == account)
            total_debits += d
            total_credits += c
            if account.name == asked_account:
                to_return = d + c

        return to_return


def get_monthly_totals(ledger, period):
    """function to calculate the monthly totals

    Args:
        ledger (_type_): _description_

    Returns:
        dictionary: dictonary with the monthly totals
    """    
    monthly_totals = defaultdict(lambda: defaultdict(float))
    for transaction in ledger.transactions:
        year, month, day = transaction.date.year, transaction.date.month, transaction.date.day
        account_name = transaction.credit_account.name if transaction.credit_account.name != "None" else transaction.debit_account.name
        if period == "month":
            monthly_totals[account_name][f"{year}-{month:02}"] += transaction.amount
        elif period == "year":
            monthly_totals[account_name][f"{year}"] += transaction.amount
        else:
            print_x ("Error in PERIOD")
    
    return monthly_totals



def get_monthly_pivot_table(ledger, period):
    """make the montly totals

    Args:
        ledger (_type_): _description_

    Returns:
        df: dataframe with the montly totals
        df_t : the transposed dataframe
    """    
    monthly_totals = get_monthly_totals(ledger, period)
    df = pd.DataFrame(monthly_totals)
    df = df.fillna(0)  # replace NaN values with 0
    df = df.sort_index()  # sort rows alphabetically by account name
    # df.columns.name = "Month"
    df = df.apply(lambda x: round(x, 2))  # round values to 2 decimal places
    

    df_t = df.T
    
    df_t.loc['TOTAL'] = df_t.sum()
    df.loc['TOTAL'] = df.sum()


    #df_t_sum = df_t.sum(axis=1)
    #df_t['TOTAL_x'] = df_t_sum
    df_t.reset_index(level=0, inplace=True) 
   
    return df, df_t


def make_pivot_period_trial_balance(ledger, accounts_source, period):
    """Make a pivot table with a montly trial balance

    Args:
        ledger (_type_): _description_
        accounts_source (_type_): list with the sources (cash, bank accounts etc)
    """    
    # Define the start and end year
    start_year = 2023
    end_year = 2024

    # Initialize an empty DataFrame to store the trial balances
    trial_balance_df = pd.DataFrame(columns=['Date', 'Account', 'Balance'])

      
    if period == "monthly":
        date_list = make_monthly_list()
    elif period == "daily":
        date_list = make_daily_list()
    else:
        print_x("ERROR in PERIOD")

    
    for my_date in date_list:
        # Loop through each account and retrieve the trial balance on the first day of the month
        for a in accounts_source:
            balance = round(ledger.ask_trial_balance_date(a, my_date),2)
            trial_balance_df=trial_balance_df.fillna(0)
            # Append the trial balance to the DataFrame
            trial_balance_df = pd.concat([trial_balance_df, pd.DataFrame({'Date': [my_date], 'Account': [a], 'Balance': [balance]})], ignore_index=True)



    # Create the pivot table
    pivot_table = pd.pivot_table(trial_balance_df, values='Balance', index=['Account'], columns=['Date'])
    
    pivot_table.loc['TOTAL'] = pivot_table.sum()
    return pivot_table
def make_daily_list():
    """Make a list with dates from 31/12/2021 until today

    Returns:
        list: the list :)
    """    
    start_date = datetime.datetime(2021, 12, 31).date()
    current_date = datetime.datetime.now().date()

    date_list = []
    current = start_date

    while current <= current_date:
        date_list.append(current)
        current += timedelta(days=1)
    return date_list
        
def make_monthly_list():
    """Make a list with the last days of the months from 31/12/2021

    Returns:
        list: the list :P
    """

    last_days = [(datetime.datetime(2021, 12, 31)).date()]
    for year in range(2021, 2025):
        for month in range(1, 13):
            if month >7 and year == 2024:
                break
            else:
                if month == 12:
                    last_day = datetime.datetime(year, month, 31)
                else:
                    last_day = datetime.datetime(year, month+1, 1) - datetime.timedelta(days=1)
                last_day = last_day.date()
                last_days.append(last_day)
    return last_days

def create_ledger(df, accounts, modus):
    """_summary_

    Args:
        df (df): dataframe with transactions
        accounts (list): list with the accounts to mention in the ledger (ic all accounts)
        modus (string): detail niveau "category", "main_category", "income_expenses"

    Returns:
        _type_: _description_
    """    
    ledger = Ledger()
    
    for a in accounts:
        a_ = Account(a)
        # add accounts to the ledger
        ledger.add_account(a_)

   
    for index, row in df.iterrows():
        ledger.add_transaction(
            row["datum"].date(),
            row["source"],
            row[modus],
            row["bedrag"],
            row["description"]
        )
        
    return ledger

def show_income_expenses(df,year):
    pass
    # df =df[df["income_expenses"] != "STARTING_BALANCE" ]
    # df= df.T
    # st.write(df)

 

    # df['sum_in'] = df.loc[:, (df.loc['income_expenses'] == 'IN')].sum()
    # df['sum_uit'] = df.loc[:, (df.loc['income_expenses'] != 'IN')].sum()
    # st.write(df)
    # aaa = 'income_expenses'
    # bbb = 'main_cat'
    # ccc = 'category'
    # ddd = "TOTAL_x_inv"
   
    # # https://stackoverflow.com/questions/70129355/value-annotations-around-plotly-sunburst-diagram
    
    # # Specify the year you want to filter
   
    # columns_to_keep = [aaa,bbb,ccc]
    # #filtered_columns = [col for col in df.columns if col.startswith(year)] + columns_to_keep
    # #df_t =df.T
    # # st.write(df_t)
    
 

def show_sunburst_annotated(df, year, divide_factor):
    st.subheader(f"Uitgaves in {year}")
    aaa = 'income_expenses'
    bbb = 'main_cat'
    ccc = 'category'
    ddd = "TOTAL_x_inv_divided"
    # df = df[df["income_expenses"] != "IN"]
    # df =df[df["income_expenses"] != "STARTING_BALANCE" ]
    if debug:
        st.write("df 374 xxx")
        st.write(df)
    df = df[(df["index"] != "IN") & (df["index"] != "STARTING_BALANCE")]
    # https://stackoverflow.com/questions/70129355/value-annotations-around-plotly-sunburst-diagram
    
    # Specify the year you want to filter
   
    columns_to_keep = [aaa,bbb,ccc]
    # Filter the columns based on the specified year
    filtered_columns = [col for col in df.columns if col.startswith(year)] + columns_to_keep
    if debug:
        st.write(df)
        st.write (filtered_columns)
    # Create a new dataframe with the filtered columns
    df = df[filtered_columns]
    if debug:
        st.write(df)
        print(df.dtypes)
    df_sum = df.fillna(0).sum(axis=1, numeric_only = True)
    # df_numeric = df.fillna(0)

    # Calculate the sum of each row
    #df_sum = df_numeric.sum(axis=1)

    df['TOTAL_x'] = df_sum
    #df_t.reset_index(level=0, inplace=True) 

    from math import sin,cos,pi
    df["TOTAL_x_inv"] = df["TOTAL_x"]*-1 
    df["TOTAL_x_inv_divided"] = df["TOTAL_x_inv"]/divide_factor


    df=df[[aaa,bbb,ccc,ddd]]


    # Combine the rows for 'bbb' and 'ccc'
    combined_amount = df.loc[df[ccc].isin(['zorgtoeslag', 'zorgverzekering']), ddd].sum()
    
    # Remove the rows for 'bbb' and 'ccc'
    df = df.drop(df[df[ccc].isin(['zorgtoeslag', 'zorgverzekering'])].index)
    df.loc[len(df)] = ["UIT_VL", "VASTELASTEN", "zorgverzekering_netto", combined_amount]
    df= df[df[ddd] >= 0]

    
    fig = px.sunburst(df, path=[aaa,bbb,ccc], values=ddd, width=600, height=600, title=f"Uitgaven",)
    totals_groupby =  df.groupby([aaa, bbb, ccc]).sum()
    totals_groupby["aaa_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=aaa).transform('sum')
    totals_groupby["aaa_bbb_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb]).transform('sum')
    totals_groupby["aaa_bbb_ccc_sum"] = getattr(df.groupby([aaa, bbb, ccc]), ddd).sum().groupby(level=[aaa,bbb,ccc]).transform('sum')
    totals_groupby = totals_groupby.sort_values(by=["aaa_sum","aaa_bbb_sum","aaa_bbb_ccc_sum"], ascending=[0,0,0])
   
   
    ## calculate the angle subtended by each category
    sum_ddd = getattr(df,ddd).sum()
    delta_angles = 360*totals_groupby[ddd] / sum_ddd

    annotations = [format(v,".0f") for v in  getattr(totals_groupby,ddd).values]
    
    ## calculate cumulative sum starting from 0, then take a rolling mean 
    ## to get the angle where the annotations should go
    angles_in_degrees = pd.concat([pd.DataFrame(data=[0]),delta_angles]).cumsum().rolling(window=2).mean().dropna().values

    def get_xy_coordinates(angles_in_degrees, r=1):
        return [r*cos(angle*pi/180) for angle in angles_in_degrees], [r*sin(angle*pi/180) for angle in angles_in_degrees]
        #return [r*cos(angle) for angle in angles_in_degrees], [r*sin(angle) for angle in angles_in_degrees]

    x_coordinates, y_coordinates = get_xy_coordinates(angles_in_degrees, r=1.5)
    fig.add_trace(go.Scatter(
        x=x_coordinates,
        y=y_coordinates,
        mode="text",
        text=annotations,
        hoverinfo="skip",
        textfont=dict(size=14)
    ))

    padding = 0.50
    fig.update_layout( margin=dict(l=20, r=20, t=20, b=20),
        
        xaxis=dict(
            range=[-1 - padding, 1 + padding], 
            showticklabels=False
        ), 
        yaxis=dict(
            range=[-1 - padding, 1 + padding],
            showticklabels=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        autosize=False,
        width=800,
        height=800,
    )
    st.plotly_chart(fig)




def barchart_income_expenses(table, titel, period):
    months = table[period].to_list() 
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=table["IN"],
                    base=0,
                    marker_color='lightslategrey',
                    name='inkomsten'
                    ))
    fig.add_trace(go.Bar(x=months, y=table["UIT_TOT"],
                    base=0,
                    marker_color='crimson',
                    name='uitgaven'))

    fig.update_layout(
        yaxis=dict(
            title=f"(€)",
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            )
        ),
        title=dict(
                text=titel,
                x=0.5,
                y=0.85,
                font=dict(
                    family="Arial",
                    size=14,
                    color='#000000'
                )
            )
    )
    st.plotly_chart(fig)


def make_graph_daily_trial_balance_totals(df_daily_trial_balance):
    df_daily_trial_balance.reset_index(level=0, inplace=True)

    #df_daily_trial_balance_selected = df_daily_trial_balance.loc[:, ['Date', 'TOTAL']]
    
    # Convert 'Date' column to datetime
    df_daily_trial_balance['Date'] = pd.to_datetime(df_daily_trial_balance['Date'])

    # Create line plot using Plotly Express
    fig = px.line(df_daily_trial_balance, x='Date', y='TOTAL')

    # Configure the plot layout
    fig.update_layout(
        title='Total',
        xaxis_title='Date',
        yaxis_title='Total',
    )

    # Render the plot using Streamlit
    st.plotly_chart(fig)

def check_categories_and_maincategories():
    """Checks the file to see if there are categories with different maincategories, which
    gives a "duplicates" error in /make_graph_monthly_expenses/
    
    """   
    file_path = r"C:\Users\rcxsm\Documents\xls\masterfinance_2023.xlsx"  # Replace with your file path
    df = pd.read_excel(file_path, sheet_name='INVOER')
 
    categories = df["category"].unique().tolist()
   
    for c in categories:
        df_=df[df["category"]==c]
       
        main_categories = df_["main_category"].unique().tolist()
        if len(main_categories)>1:
            print (c)
            print (main_categories)

def make_graph_monthly_expenses(period, divide_factor, df_):
    """_summary_

    If it gives an error use /check_categories_and_maincategories/

    Args:
        period (_type_): _description_
        divide_factor (_type_): _description_
        df_ (_type_): _description_
    """    
    if period == "month":
        st.subheader("Maandelijkse uitgaves/inkomsten")
    else:
        st.subheader("Jaarlijkse uitgaves/inkomsten")
    if debug:
        st.write("526")
        st.write(df_)
    df_t_ = df_.T.reset_index()
    df_t_.columns = df_t_.iloc[0]

    # Drop the first row from the DataFrame
    df_t_ = df_t_[1:].reset_index(drop=True)

    df_t_['month'] = df_t_['index']


    df_t_ = df_t_.drop(['index'], axis=1)
    st.write(df_t_)
    # Delete first row (starting balance) and last three rows
    df_t_ = df_t_.iloc[1:-3]

    df_t_['date'] = pd.to_datetime(df_t_['month'])

    column_names = df_t_.columns.tolist()[1:]
    
    fig = px.line(df_t_, x="date", y=column_names, title = f"_")
    fig.add_hline(y=0)
    st.plotly_chart(fig)
    

        
    # Melt the DataFrame to convert multiple columns to a single 'value' column
    df_melted = df_t_.melt(id_vars='date', var_name='variable', value_name='value')

    # Create a bar graph using Plotly Express
    fig2 = px.bar(df_melted, x='date', y='value', color='variable', barmode='group', title='Income/Expenses')
    st.plotly_chart(fig2)
    

    for y in ["2022", "2023", "2024"]:
        show_sunburst_annotated(df_, y, divide_factor)

def give_totals_boodschappen(df):
       
    # Assuming 'df' is your DataFrame
    # First, ensure that the 'Bedrag' column is of numeric data type (e.g., float or int)
    # If it's not already, you can convert it like this:
    # df['Bedrag'] = pd.to_numeric(df['Bedrag'], errors='coerce')

    # Filter rows where 'category' is equal to 'boodschappen'
    st.subheader("Boodschappen")
    boodschappen_df = df[df['category'] == 'BOODSCHAPPEN_SEVENUM']

    # Extract the month and year from the datetime column (assuming it's named 'date')
    boodschappen_df['Month'] = boodschappen_df['datum'].dt.strftime('%Y-%m')
    boodschappen_df['Year'] = boodschappen_df['datum'].dt.strftime('%Y')

    # Group by month and calculate the sum of 'Bedrag'
    monthly_result = boodschappen_df.groupby('Month')['bedrag'].sum().reset_index()
    monthly_result.rename(columns={'bedrag': 'Total_Bedrag_Month'}, inplace=True)

    # Group by year and calculate the sum of 'Bedrag'
    yearly_result = boodschappen_df.groupby('Year')['bedrag'].sum().reset_index()
    yearly_result.rename(columns={'bedrag': 'Total_Bedrag_Year'}, inplace=True)

    # Print the resulting DataFrames
    st.write("Monthly Total:")
    st.write(monthly_result)

    st.write("\nYearly Total:")
    st.write(yearly_result)

    # Define the number of months for each year
    months_per_year = {
        '2021': 6.0,
        '2022': 3.8,
        '2023': 7.5
    }
    yearly_totals = boodschappen_df.groupby('Year')['bedrag'].sum()
    # Calculate the average monthly spending for each year
    average_monthly_spending = yearly_totals / pd.Series(months_per_year)

    # Print the result
    st.write("Average Monthly Spending:")
    st.write(average_monthly_spending)
def main():
    print ("-----------")
    modus = st.sidebar.selectbox("Modus", ["income_expenses","category", "main_category"], index = 1)
    period = st.sidebar.selectbox("Period", ["month","year"], index = 0)
    divide_factor = st.sidebar.number_input("Sunburst delen door",1.0,12.0,1.0)
    
    df, df_cat = read()
    st.write(df)
    give_totals_boodschappen(df)


    accounts = pd.concat([df['source'], df[modus]]).unique().tolist()
    accounts_source = (df['source']).unique().tolist()

    ledger = create_ledger(df, accounts, modus)
    if debug:
        st.write("ledger 581 xxx")
        st.write(ledger)
    df,df_t = get_monthly_pivot_table(ledger, period)
    if debug:
        st.write("585 df xxx")
        st.write(df)

        st.write("573 df_t xxx")
        st.write(df_t)
        st.write("573 df_cat xx")
        st.write(df_cat)


    df_ = pd.merge(df_t, df_cat, left_on = "index", right_on ="category", how="inner")
    if debug:
        st.write("592 df_ xxx")
        st.write(df_)
     
    make_graph_monthly_expenses(period, divide_factor, df_)
   
    show_income_expenses(df_,"2022")

    # print_x trial balance on a certain date
    print_x(" ")
    print_x("MONTHLY TRIAL BALANCES")
    print_x(make_pivot_period_trial_balance(ledger, accounts_source, "monthly"))
    print_x("DAILY TRIAL BALANCES")
    
    df_daily_trial_balance = make_pivot_period_trial_balance(ledger, accounts_source, "daily").T
    make_graph_daily_trial_balance_totals(df_daily_trial_balance)

if __name__ == "__main__":
    main()
    # check_categories_and_maincategories():
      