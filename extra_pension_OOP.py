import statistics
import random
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import datetime

class PensionCalculator:
    def __init__(self):
        # Initialize default values
        self.initial_one_time_contribution = 0
        self.monthly_contribution_original = 200
        self.monthly_contribution_original_how = "with inflation"
        
        self.current_age = 46
        self.retirement_age = 69
        self.max_age= 120
        # self.expected_life_expectancy = 81.4
        # self.expected_life_expectancy_sd = 10
        self.annual_return_rate =2.0
        self.annual_return_rate_sd = 0.0
        self.inflation = 2.62
        self.inflation_sd = 0.0

        self.additional_monthly_need_how = "with inflation"
        self.monthly_pension_without_reduction_original = 1458
        self.years_shortfall = 7
        self.additional_monthly_need = 0
        self.windfall_1_year = 5
        self.windfall_1_amount = 0
        
        self.windfall_2_year = 15
        self.windfall_2_amount = 0
        
        self.windfall_3_year = 25
        self.windfall_3_amount = 0
        
        self.windfalls = [
            {"year": self.windfall_1_year, "amount": self.windfall_1_amount},
            {"year": self.windfall_2_year, "amount": self.windfall_2_amount},
            {"year": self.windfall_3_year, "amount": self.windfall_3_amount}
        ]
        

        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Extract the current year from the datetime object
        self.current_year = current_datetime.year

    def interface(self):
             #  # Get user input for parameters and update the calculator instance
        self.initial_one_time_contribution = st.sidebar.number_input("Initial One-Time Contribution:", value=self.initial_one_time_contribution)
        self.monthly_contribution_original = st.sidebar.number_input("Monthly Contribution (current pricelevel):", value=self.monthly_contribution_original)
        self.monthly_contribution_original_how = st.sidebar.selectbox("Monthly contribution How", ["with inflation", "without inflation"],0 )

        st.sidebar.subheader("--- The person ---")
        self.sexe = st.sidebar.selectbox("sexe", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)
        self.retirement_age = st.sidebar.number_input("Retirement Age:", value=self.retirement_age)
        self.max_age = st.sidebar.number_input("Maximum Age:", value=self.max_age)
        #self.expected_life_expectancy = st.sidebar.number_input("Expected Life Expectancy:", value=self.expected_life_expectancy)
        #self.expected_life_expectancy_sd = st.sidebar.number_input("Expected Life Expectancy SD:", value=self.expected_life_expectancy_sd)
        self.birthyear = self.current_year - self.current_age
        st.sidebar.write(f"Years to go to pension: {self.retirement_age - self.current_age}")
        #st.sidebar.write(f"Number of years to sustain {round(self.expected_life_expectancy - self.retirement_age,1)}")
        #st.sidebar.write(f"Ratio : {round((self.expected_life_expectancy - self.retirement_age) / (self.retirement_age - self.current_age),1)}")
        
        st.sidebar.subheader("--- Rates ---") 
        self.annual_return_rate = st.sidebar.number_input("Annual Interest Rate (%):", value=self.annual_return_rate)
        self.inflation = st.sidebar.number_input("Average Annual Inflation Rate (%):", value=self.inflation)
        self.annual_return_rate_sd = st.sidebar.number_input("Annual Interest Rate SD(%):", value=self.annual_return_rate_sd)
        self.inflation_sd = st.sidebar.number_input("Average Annual Inflation Rate SD(%):", value=self.inflation_sd)
        st.sidebar.subheader("--- Pension data ---")
        self.monthly_pension_without_reduction_original = st.sidebar.number_input("Monthly Pension without Reduction (current price level):", value=self.monthly_pension_without_reduction_original)
        self.years_shortfall = st.sidebar.number_input("Years of Shortfall:", value=self.years_shortfall)
        st.sidebar.write(f"Shortfall per month (current price level): {round(self.years_shortfall * 0.02 * self.monthly_pension_without_reduction_original)}")
        self.additional_monthly_need = st.sidebar.number_input("Additional Monthly Need (current price level):", value=self.additional_monthly_need)
        self.additional_monthly_need_how = st.sidebar.selectbox("Additional monthly needed How", ["with inflation", "without inflation"],0 )

        st.sidebar.subheader("--- Windfalls ---") # "meevallers"
        self.windfall_1_year = st.sidebar.number_input("Windfall 1 (Year):", value=self.windfall_1_year)
        self.windfall_1_amount = st.sidebar.number_input("Windfall 1 (Amount):", value=self.windfall_1_amount)   
        self.windfall_2_year = st.sidebar.number_input("Windfall 2 (Year):", value=self.windfall_2_year)
        self.windfall_2_amount = st.sidebar.number_input("Windfall 2 (Amount):", value=self.windfall_2_amount)
        self.windfall_3_year = st.sidebar.number_input("Windfall 3 (Year):", value=self.windfall_3_year)
        self.windfall_3_amount = st.sidebar.number_input("Windfall 3 (Amount):", value=self.windfall_3_amount)

        st.sidebar.subheader("--- Simulations ---")
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,100)
        self.new_method =  True # st.sidebar.selectbox("Use AG table", [True, False],0)
        self.print_individual =  st.sidebar.selectbox("Print individual runs", [True, False],1)

    def calculate_pension(self, num_simulations=1000):
        """Calculate the pension and balances

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
        """
        deceased_ages, saldo_at_death_values, results, start_year = [],[],[], self.current_year
        # Projections Life Table AG2022
        # https://www.actuarieelgenootschap.nl/kennisbank/ag-l-projections-life-table-ag2022.htm
        if self.sexe== "male":
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2022DefinitiefGevalideerd_male.csv")
        else:
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2022DefinitiefGevalideerd_female.csv")
       
        special_years = [self.windfall_1_year, self.windfall_2_year,self. windfall_3_year]
        special_amounts = [self.windfall_1_amount, self.windfall_2_amount, self.windfall_3_amount]
        s1,s2y = int(time.time()),int(time.time())
        for _ in range(num_simulations):
            # Calculate the percentage completion
            completion_percentage = (_ + 1) / num_simulations * 100

            # Check if the current iteration is a multiple of 10%
            if completion_percentage % 10 == 0:
                s2x = int(time.time())
                print(f"Progress: {int(completion_percentage)}% complete [{_+1}/{num_simulations}] [Round : {s2x-s2y} seconds | Cummulative : {s2x-s1} seconds]")
                s2y = s2x
            # display_progress_bar(_,num_simulations)
            person_alive = True
            # Calculate pension-related values for each simulation
            annual_contribution_values = []
            balance_values = []
            interest_values = []
            annual_shortfall_values = []
            
            annual_contribution_original = self.monthly_contribution_original * 12
            annual_pension = self.monthly_pension_without_reduction_original * 12
            annual_shortfall_original = (self.years_shortfall * 0.02 * annual_pension) + (self.additional_monthly_need * 12)

            balance = self.initial_one_time_contribution
            years_until_retirement = self.retirement_age - self.current_age
            annual_return_rate = np.maximum(np.random.normal(self.annual_return_rate, self.annual_return_rate_sd),0)  # SD = 2
            # self.expected_life_expectancy_run = int( np.maximum(np.random.normal(self.expected_life_expectancy, self.expected_life_expectancy_sd),0))
           
            inflation = np.maximum(np.random.normal(self.inflation, self.inflation_sd), 0)

            # until the retirement
            for i in range(0, years_until_retirement + 1):
                current_year = i + start_year
                # Generate random values for annual_return_rate and inflation
                if  balance>0:
                    interest = round(balance * (annual_return_rate / 100), 2)
                    balance += interest 
                else:
                    interest = 0
                balance +=  annual_contribution_original

                if i in special_years:
                    # adding the windfalls ("meevallers")
                    special_index = special_years.index(i)
                    special_addition = special_amounts[special_index]
                    balance += special_addition

                annual_contribution_values.append(annual_contribution_original)
                balance_values.append(int(balance))
                interest_values.append(interest)
                annual_shortfall_values.append(0)
                if self.monthly_contribution_original_how == "with inflation":
                    annual_contribution_original = round(annual_contribution_original * (100 + inflation) / 100, 2)
                else:
                    annual_contribution_original = round(annual_contribution_original * (100 + 0) / 100, 2)
             
                if person_alive:
                    age = self.current_age +i 
                   
                    # use of the AG2022 table. 
                    probability_to_die = df_prob_die[str(current_year)].to_numpy()[df_prob_die['age'].to_numpy() == age].item()
                    
                    #probability_to_die = df_prob_die.loc[df_prob_die['age'] == age, str(current_year)].values[0]
                    if random.random() <= probability_to_die:
                                deceased_ages.append(age)
                                saldo_at_death_values.append(balance)
                                person_alive = False
                    
            if self.additional_monthly_need_how == "with inflation":
                annual_shortfall_corrected = annual_shortfall_original * ((100 + inflation) / 100) ** years_until_retirement
            else:
                annual_shortfall_corrected = annual_shortfall_original * ((100 + 0) / 100) ** years_until_retirement
            
            # after the retirement
            for j in range(years_until_retirement + 1, self.max_age - self.current_age + 1):
                if balance >0 :
                    interest = balance * (annual_return_rate / 100)
                    balance += interest
                else:
                    interest = 0
                balance -= annual_shortfall_corrected
                if self.additional_monthly_need_how == "with inflation":
                    annual_shortfall_corrected = annual_shortfall_corrected * ((100 + inflation) / 100)
                else:
                    annual_shortfall_corrected = annual_shortfall_corrected * ((100 + 0) / 100)

                annual_contribution_values.append(0)
                balance_values.append(int(balance))
                interest_values.append(interest)
                annual_shortfall_values.append(annual_shortfall_corrected)

                if person_alive:
                    age = self.current_age +j
                    
                    # use of the AG2022 table
                    if age>100:
                        # values not given in the table. Rounded to 0.6
                        probability_to_die = 0.6
                    else:
                        probability_to_die = df_prob_die[str(current_year)].to_numpy()[df_prob_die['age'].to_numpy() == age].item()
                    #probability_to_die = df_prob_die.loc[df_prob_die['age'] == age, str(current_year)].values[0]
                    if random.random() <= probability_to_die:
                                deceased_ages.append(age)
                                saldo_at_death_values.append(balance)
                                person_alive = False

                
            results.append({
                'annual_contribution_values': annual_contribution_values,
                'balance_values': balance_values,
                'interest_values': interest_values,
                'annual_shortfall_values': annual_shortfall_values,
                
            })
        
        # Store all results in the instance variable
        self.results = results
        self.deceased_ages = deceased_ages
        self.median_age_at_death = round(statistics.median(deceased_ages),1)
  
        sorted_ages = np.sort(deceased_ages)

        self.percentile_2_5 = np.percentile(sorted_ages, 2.5)
        self.percentile_95 = np.percentile(sorted_ages, 95)
        self.percentile_25 = np.percentile(sorted_ages, 25)
        self.percentile_75 = np.percentile(sorted_ages, 75)
        self.percentile_97_5 = np.percentile(sorted_ages, 97.5)

        st.write(f"Average saldo at the death of  {num_simulations} persons ({self.sexe}) : {round(sum(saldo_at_death_values)/len(saldo_at_death_values))} - SD {round(np.std(saldo_at_death_values),1)}")
        if sum(saldo_at_death_values) > 0:
            st.write(f"Profit for pension funds : {round(sum(saldo_at_death_values))}")
        else:
            st.write(f"Loss for pension funds : {round(sum(saldo_at_death_values))}")

        s2 = int(time.time())
        st.write(f"Time needed: {s2-s1} seconds")
        #self.show_ages_at_death(num_simulations, self.deceased_ages)

    def show_ages_at_death(self, num_simulations,):
        """Show a graph of the age of death of people in the simulations

        Args:
            num_simulations (int): sumber of simulations 
        """        

        df_deceased = pd.DataFrame({'ages': self.deceased_ages})
        all_ages = pd.DataFrame({'ages':  range(self.current_age, self.max_age+ 1)})

        # Count the frequency of each age in the 'deceased_ages' list
        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='left').fillna(0)

        # trace = go.Histogram(
        #     x=self.deceased_ages,
        #     xbins=dict(
        #         start=min(self.deceased_ages),
        #         end=max(self.deceased_ages),
        #         size=1  # Adjust the bin size as needed
        #     ),
        #     opacity=0.7  # Set the opacity of bars
        # )

        # # Create the layout for the histogram
        # layout = go.Layout(
        #     title='Histogram Deceased ages',
        #     xaxis=dict(title='Value'),
        #     yaxis=dict(title='Frequency'),
        # )

        # # Create the figure and plot it
        # fig3 = go.Figure(data=[trace], layout=layout)
        #fig3= go.Bar(x=end_table["ages"], y=end_table["frequency"])
        # fig3.add_vline(x=statistics.median(self.deceased_ages), line_dash="dash", line_color="grey", annotation_text="mediaan", annotation_position="top right")
        # fig3.add_vline(x=self.percentile_2_5, line_dash="dash", line_color="grey", annotation_text="2.5%", annotation_position="top right")
        # fig3.add_vline(x=self.percentile_25, line_dash="dash", line_color="grey", annotation_text="25%", annotation_position="top right")
        # fig3.add_vline(x=self.percentile_75, line_dash="dash", line_color="grey", annotation_text="75%", annotation_position="top right")
        
        # fig3.add_vline(x=self.percentile_97_5, line_dash="dash", line_color="grey", annotation_text="97.5%", annotation_position="top right")
        
        vlines = [statistics.median(self.deceased_ages), self.percentile_2_5, self.percentile_25, self.percentile_75, self.percentile_97_5]
        vtxt = ["median", "2,5%", "25%", "75%", "97,5%"]
        # Create a bar graph
        fig3 = go.Figure(data=[go.Bar(x=end_table["ages"], y=end_table["frequency"])])
        for i,txt in zip(vlines, vtxt) :
            # Add vertical lines at x=40
            fig3.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=i,
                    x1=i,
                    y0=0,
                    name=txt,
                    y1=max(end_table["frequency"]),  # Adjust the y1 value as needed
                    line=dict(color="grey", width=1)
                )
            )
            fig3.add_annotation(
                go.layout.Annotation(
                    text=txt,
                    x=i,
                    y=max(end_table["frequency"]),  # Adjust the y position as needed
                    showarrow=True,
                    arrowhead=2,
        
                    arrowwidth=2,
                   
                )
            )

        # Update the layout to adjust the appearance of the graph
        fig3.update_layout(
            title="Age Frequency Bar Graph with Vertical Line at 40",
            xaxis_title="Ages",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig3)
   
        st.write(f"Average age at death of {num_simulations} individuals ({self.sexe}): {round(sum(self.deceased_ages)/len(self.deceased_ages),1)} - SD {round(np.std(self.deceased_ages),1)}")
        st.write(f"Median age at death: {round(statistics.median(self.deceased_ages),1)}")
        st.write (f"2.5% Percentile: {self.percentile_2_5:.2f} / 95% Percentile: {self.percentile_95:.2f} / 97.5% Percentile: {self.percentile_97_5:.2f}")
        st.write(f"Sum of persons {end_table['frequency'].sum()}")
    def plot_values_with_confidence_intervals(self, what):
        """Plot a graph with the  values with the CI's

        Args:
            what (str): which column to plot
        """        
        st.subheader(what)
        # Extract balance values from results
        values = np.array([result[what] for result in self.results])
        
        # Transpose the balance_values array
        values = values.T  # Each column represents a run, and each row represents a year

        # Create the time axis (years)
        # max_years = self.max_age - self.current_age +1 # Limit to a maximum of 100 - current_age years
        # years = list(range(max_years))

        # Calculate mean balance and standard deviation for each year
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        confidence_interval = std*1.96
        # Create the time axis (years)
        # years = list(range(len(mean_balance)))
        years = [self.current_age + i for i in range(len(mean))]
        max_years = self.max_age - self.current_age+1
        # Create a Plotly figure
        fig = go.Figure()

        # Add the mean balance line
        fig.add_trace(go.Scatter(x=years, y=mean[:max_years], mode='lines', name='Mean', line=dict(color='blue')))
        #fig.add_trace(go.Scatter(x=years, y=std[:max_years], mode='lines', name='Std', line=dict(color='red')))

        fig.add_trace( go.Scatter(x=years , y=mean[:max_years] + confidence_interval[:max_years], name='Upper CI',
                                    line=dict(color='dimgrey', width=.5),))# fill down to xaxis
        fig.add_trace(go.Scatter(x=years , y=mean[:max_years] - confidence_interval[:max_years],name='Lower CI', fill='tonexty',fillcolor='rgba(0, 128, 0, 0.2)',
                                    line=dict(color='dimgrey',width=.5),)) # fill to trace0 y

        # Update the layout
        fig.update_layout(
            xaxis_title='Years',
            yaxis_title=what,
            title=f'{what} Over Time with 95% Confidence Intervals',
            legend=dict(x=0, y=1),
            showlegend=True
        )
        fig.update_traces(mode='lines')
        fig.add_hline(y=0,  line_color="black")
        
        fig.add_vline(x=self.retirement_age, line_dash="dash", line_color="green", annotation_text="Retirement Age", annotation_position="top left")
        #fig.add_vline(x=self.expected_life_expectancy, line_dash="dash", line_color="black", annotation_text="Expected Life Expectancy", annotation_position="bottom left")
        #fig.add_vline(x=self.expected_life_expectancy+10, line_dash="dash", line_color="grey", annotation_text="ELE +10", annotation_position="top right")
        fig.add_vline(x=self.percentile_97_5, line_dash="dash", line_color="grey", annotation_text="97.5%", annotation_position="top right")
        fig.add_vline(x=self.percentile_2_5, line_dash="dash", line_color="grey", annotation_text="2.5%", annotation_position="top right")
        fig.add_vline(x=self.median_age_at_death , line_dash="dash", line_color="grey", annotation_text="Median", annotation_position="top right")
        # Set Y-axis range
        fig.update_yaxes(range=[-30000, mean.max()*1.1])

        # Show the Plotly figure
        st.plotly_chart(fig)

        # print_individual = True
        if self.print_individual:
            # Create a list to store traces (one trace for each run)
            traces = []

            # Get the number of runs (number of columns)
            num_runs = values.shape[1]

            # Create a separate trace for each column (run)
            for run in range(num_runs):
                trace = go.Scatter(x=np.arange(self.current_age, self.current_age + len(values)), y=values[:, run], mode='lines', name=f'Run {run+1}')
                traces.append(trace)

            # Create the layout for the plot
            layout = go.Layout(
                title=f'{what} - Line Plot for Each Column',
                xaxis=dict(title='Year'),
                yaxis=dict(title=what),
                showlegend=True
            )

            # Create the figure and plot it
            fig2 = go.Figure(data=traces, layout=layout)

            st.plotly_chart(fig2)
    def show_total_balance(self):
        """Show the total balance in time (eg. the profit or loss of the life insurance company)
        """        
        # Create a DataFrame from deceased_ages
        df_deceased = pd.DataFrame({'ages': self.deceased_ages})
        all_ages = pd.DataFrame({'ages':  range(self.current_age, self.max_age + 1)})

        # Count the frequency of each age in the 'deceased_ages' list
        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='left').fillna(0)
        end_table["year"] = self.birthyear + end_table["ages"]
        values = np.array([result['balance_values'] for result in self.results])
            
        # Transpose the balance_values array
        values = values.T  # Each column represents a run, and each row represents a year
        
        # Create the time axis (years)
        max_years = 100 - self.current_age +1 # Limit to a maximum of 100 - current_age years

        # Calculate mean balance and standard deviation for each year
        mean = np.mean(values, axis=1)
        end_table['mean'] = mean
        end_table['per_year'] = end_table['mean'] * end_table['frequency']
        end_table['per_year_cumm'] = end_table['per_year'].cumsum()
        st.subheader("Profit/loss for the insurance company through the time (excl. costs)")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=end_table['year'], y=end_table['per_year_cumm'], mode='lines', name='Cumm Summ Balance', line=dict(color='blue')))
        fig5.add_hline(y=0,  line_color="black")
        st.plotly_chart(fig5)   

def main():

    calculator = PensionCalculator()

    calculator.interface()

    calculator.calculate_pension(num_simulations=calculator.num_simulations)
    calculator.plot_values_with_confidence_intervals("balance_values")
    calculator.show_ages_at_death(calculator.num_simulations)
    calculator.show_total_balance()

if __name__ == "__main__":
    main()
