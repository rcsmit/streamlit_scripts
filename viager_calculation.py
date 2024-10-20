import statistics
import random
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
    
class PensionCalculator:
    def __init__(self, current_age=90,monthly_compensation_original=2000 ):
        # Initialize default values
        self.house_value_at_start = 200_000
        self.pay_out_at_start = 0
        self.monthly_compensation_original = monthly_compensation_original
        self.monthly_contribution_original_how = "with inflation"
        
        self.current_age = current_age
        self.max_age= 120
        self.sexe = "male"
        self.annual_return_rate =2.0
        self.annual_return_rate_sd = 0.0
        self.inflation = 2.62
        self.inflation_sd = 0.0
        self.cost_house = 1.0
        self.additional_monthly_need_how = "with inflation"
        self.num_simulations=10
        self.new_method =  True
        self.print_individual =  False
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Extract the current year from the datetime object
        self.current_year = current_datetime.year
        self.load_life_table()
    def interface(self):
             #  # Get user input for parameters and update the calculator instance
        self.house_value_at_start = st.sidebar.number_input("house_value_at_start:", value=self.house_value_at_start)
        self.pay_out_at_start = st.sidebar.number_input("Lump sum / bouquet at start:", value=self.pay_out_at_start)
        
        self.monthly_compensation_original = st.sidebar.number_input("Monthly compensation / Rente (current pricelevel):", value=self.monthly_compensation_original)
        self.monthly_compensation_original_how = st.sidebar.selectbox("Monthly compensation  How", ["with inflation", "without inflation"],0 )

        st.sidebar.subheader("--- The person ---")
        self.sexe = st.sidebar.selectbox("sexe", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)
        self.max_age = st.sidebar.number_input("Maximum Age:", value=self.max_age)
        self.birthyear = self.current_year - self.current_age
        
        st.sidebar.subheader("--- Rates ---") 
        self.annual_return_rate = st.sidebar.number_input("Annual increase value house (%):", value=self.annual_return_rate)
        self.inflation = st.sidebar.number_input("Average Annual Inflation Rate (%):", value=self.inflation)
        self.annual_return_rate_sd = st.sidebar.number_input("Annual increase value house SD(%):", value=self.annual_return_rate_sd)
        self.inflation_sd = st.sidebar.number_input("Average Annual Inflation Rate SD(%):", value=self.inflation_sd)
        self.cost_house = st.sidebar.number_input("Maintenance costs house (%):", value=self.cost_house)
        
        st.sidebar.subheader("--- Simulations ---")
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,self.num_simulations) #00)
        self.new_method =  True # st.sidebar.selectbox("Use AG table", [True, False],0)
        self.print_individual =  False # st.sidebar.selectbox("Print individual runs", [True, False],1)

    def load_life_table(self):
        base_url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/"
        file_name = f"AG2024DefinitiefGevalideerd_{self.sexe}.csv"
        self.df_prob_die = pd.read_csv(base_url + file_name)
    def check_death(self,age,year_simulation):
        # use of the AG2022 table
        if age>100:
            # values not given in the table. Rounded to 0.6
            probability_to_die = 0.6
        else:
            probability_to_die = self.df_prob_die[str(year_simulation)].to_numpy()[self.df_prob_die['age'].to_numpy() == age] #.item()
        
        if random.random() <= probability_to_die:
            return True
        else:
            return False

        
    def calculate_pension(self, mode):
        """Calculate the pension and balances

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
        """
     
        results, deceased_ages, saldo_at_death_values   = [],[],[]
        paid_out_values, value_house_values, costs_house_values = [],[],[]
        # Projections Life Table AG2022
        # https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2
       
        s1,s2y = int(time.time()),int(time.time())
        for _ in range(self.num_simulations):
            # Calculate the percentage completion
            completion_percentage = (_ + 1) / self.num_simulations * 100

            # Check if the current iteration is a multiple of 10%
            if completion_percentage % 10 == 0:
                s2x = int(time.time())
                #print(f"Progress: {int(completion_percentage)}% complete [{_+1}/{self.num_simulations}] [Round : {s2x-s2y} seconds | Cummulative : {s2x-s1} seconds]")
                s2y = s2x
            
            sim_result = self.run_single_simulation()
            

            results.append(sim_result['annual_data'])
            deceased_ages.append(sim_result['age_at_death'])
            saldo_at_death_values.append(sim_result['saldo_at_death'])
            paid_out_values.append(sim_result['total_paid_out'])
            value_house_values.append(sim_result['final_house_value'])
            costs_house_values.append(sim_result['total_house_costs'])
        s2 = int(time.time())
        #print(f"Time needed: {s2-s1} seconds")
        if mode == "simulate":
            return round(sum(saldo_at_death_values)/len(saldo_at_death_values))
        else:
            return {
                'results': results,
                'deceased_ages': deceased_ages,
                'saldo_at_death_values': saldo_at_death_values,
                'paid_out_values': paid_out_values,
                'value_house_values': value_house_values,
                'costs_house_values': costs_house_values
            }
        
        # #self.show_output(num_simulations, deceased_ages, saldo_at_death_values, paid_out_values, value_house_values, costs_house_values)
        
        # 
    def run_single_simulation(self):
        
        person_alive = True
        annual_data = {'compensation': [], 'balance': [], 'interest': []}
        balance = 0 
        balance -= self.pay_out_at_start
        annual_return_rate = np.maximum(np.random.normal(self.annual_return_rate, self.annual_return_rate_sd),0)  # SD = 2
        inflation = np.maximum(np.random.normal(self.inflation, self.inflation_sd), 0)

        monthtly_compensation = self.monthly_compensation_original
        year_simulation = self.current_year
        age=self.current_age
        paid_out = 0
        costs_house=0
        
        value_house = self.house_value_at_start
        for j in range(1, int(self.max_age - self.current_age + 1)):
            
            if person_alive:
                value_house = value_house * ((100+ annual_return_rate) / 100)
                cost_house_amount = round(value_house * (self.cost_house / 100), 2)
                balance -= cost_house_amount 
                monthtly_compensation = monthtly_compensation * ((100+inflation)/100)
                paid_out +=monthtly_compensation*12
                balance -= 12 * monthtly_compensation 
                costs_house +=cost_house_amount
                
                age +=1

                annual_data['compensation'].append(int(12 * monthtly_compensation))
                annual_data['balance'].append(int(balance))
                annual_data['interest'].append(int((100 + annual_return_rate) / 100))

                if self.check_death(age,year_simulation):
                    balance += value_house
                    
                    return {
                    'annual_data': annual_data,
                    'age_at_death': age,
                    'saldo_at_death': int(balance),
                    'total_paid_out': int(paid_out),
                    'final_house_value': int(value_house),
                    'total_house_costs': int(costs_house)
                }
            

        #st.write(f"{year_simuation} : {age=} {annual_compensation_values=} - {balance_values=} - {interest_values=}")
        
              

    def show_output(self,calc_results):
        #print (calc_results)
        if sum(calc_results["saldo_at_death_values"]) > 0:
            st.info(f"Average saldo at the death of  {self.num_simulations} persons ({self.sexe}) : {round(sum(calc_results["saldo_at_death_values"])/len(calc_results["saldo_at_death_values"]))}. Total Profit for viager buyer : {round(sum(calc_results["saldo_at_death_values"]))}")# - SD {round(np.std(saldo_at_death_values),1)}")
        else:
            st.info(f"Average saldo at the death of  {self.num_simulations} persons ({self.sexe}) : {round(sum(calc_results["saldo_at_death_values"])/len(calc_results["saldo_at_death_values"]))}. Total Loss for viager buyer : {round(sum(calc_results["saldo_at_death_values"]))}")
        col1,col2= st.columns([4,6])
        with col1:
            self.show_ages_at_death(self.num_simulations, self.sexe, calc_results["deceased_ages"],self.current_age)
        with col2:
            # with st.expander("Extra info"):
            # Create a DataFrame
            df = pd.DataFrame({
                'Deceased Ages': calc_results['deceased_ages'],
                'Paid out': calc_results['paid_out_values'],
                'Value house': calc_results['value_house_values'],
                'Costs house': calc_results['costs_house_values'],
                'Saldo at Death': calc_results['saldo_at_death_values']
            })
           
           

            # Group by 'Deceased Ages' and 'Saldo at Death' to count occurrences
            summary_df = df.groupby(['Deceased Ages', 'Saldo at Death','Value house','Costs house','Paid out']).size().reset_index(name='Count')

            # Display the DataFrame
            st.table(summary_df)
        st.info("Inspired by : A 90-year-old woman signed a deal with a 47-year-old lawyer to give him her apartment upon her death in exchange for monthly payments. She outlived him, and his widow continued the payments. She received more than double the apartment's value \n https://www.threads.net/@unbfacts/post/DBTbibYuzYy")

     
    def display_results(self,calc_results):
        st.info(f"Average saldo at death: {np.mean(calc_results['saldo_at_death_values']):.2f}")
        st.info(f"{'Profit' if sum(calc_results['saldo_at_death_values']) > 0 else 'Loss'} for viager buyer: {sum(calc_results['saldo_at_death_values']):.2f}")
        
        df = pd.DataFrame({
            'Deceased Ages': calc_results['deceased_ages'],
            'Paid out': calc_results['paid_out_values'],
            'Value house': calc_results['value_house_values'],
            'Costs house': calc_results['costs_house_values'],
            'Saldo at Death': calc_results['saldo_at_death_values']
        })

        summary_df = df.groupby(['Deceased Ages', 'Saldo at Death', 'Value house', 'Costs house', 'Paid out']).size().reset_index(name='Count')
        st.table(summary_df)
   
    def show_ages_at_death(self, num_simulations,sexe, deceased_ages, current_age):
        """Show a graph of the age of death of people in the simulations

        Args:
            num_simulations (int): sumber of simulations 
        """        

        df_deceased = pd.DataFrame({'ages': deceased_ages})
        all_ages = pd.DataFrame({'ages':  range(self.current_age, max(deceased_ages)+ 1)})

        # Count the frequency of each age in the 'deceased_ages' list
        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='left').fillna(0)

        sorted_ages = np.sort(deceased_ages)

        percentile_2_5 = np.percentile(sorted_ages, 2.5)
        percentile_95 = np.percentile(sorted_ages, 95)
        percentile_25 = np.percentile(sorted_ages, 25)
        percentile_75 = np.percentile(sorted_ages, 75)
        percentile_97_5 = np.percentile(sorted_ages, 97.5)
        vlines = [statistics.median(deceased_ages), percentile_2_5, percentile_25, percentile_75, percentile_97_5]
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
            title="Age Frequency Bar Graph",
            xaxis_title="Ages",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig3)

        # CDF
                
        # Sort by ages
        end_table = end_table.sort_values(by='ages')

        # Calculate cumulative sum of frequencies
        end_table['cumulative_frequency'] = end_table['frequency'].cumsum()

        # Normalize cumulative sum to get proportions
        end_table['cdf'] = end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max()*100
        # Normalize cumulative sum to get proportions
        end_table['cdf_1'] = 100-(end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max())*100

        for c in ["cdf", "cdf_1"]:
            if c =="cdf":
                l = [50,75,95,99]
                name  =f'Cumulative Distribution Function (CDF) of Ages ({sexe} - {current_age})'
                name2 = "CDF"
                verb = "to be deceased"
            else:
                #  Complementary Cumulative Distribution Function (CCDF),
                l = [50,25,5,1]
                name = f"Survival function ({sexe} - {current_age})"
                name2 = "CCDF"
                verb = "to be still alive"
            # Create CDF plot using Plotly
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=end_table['ages'], y=end_table[c], mode='lines', name=name2))
            fig.update_layout(title=name,
                            xaxis_title='Age',
                            yaxis_title=name2)
            
            for prob in l: 
                # Find the age where cumulative probability is closest to 0.5
                age_at_prob = end_table.loc[(end_table[c] - prob).abs().idxmin()]['ages']
                        
                # Find the exact probability at that age
                exact_probability = round((end_table.loc[end_table['ages'] == age_at_prob, c].values[0]),1)

                # Add vertical line at age where cumulative probability is 0.5
                fig.add_vline(x=age_at_prob, line_dash="dash", line_color="red", annotation_text=f"{exact_probability}")
                st.write(f"{exact_probability}% probability {verb} at {age_at_prob} years")
            st.plotly_chart(fig)

        st.write(f"Average age at death of {num_simulations} individuals ({self.sexe}): {round(sum(deceased_ages)/len(deceased_ages),1)} - SD {round(np.std(deceased_ages),1)}")
        st.write(f"Median age at death: {round(statistics.median(deceased_ages),1)}")
        st.write (f"2.5% Percentile: {percentile_2_5:.2f} / 95% Percentile: {percentile_95:.2f} / 97.5% Percentile: {percentile_97_5:.2f}")
      
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

        # Calculate mean balance and standard deviation for each year
        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        confidence_interval = std*1.96
        years = [self.current_age + i for i in range(len(mean))]
        max_years = self.max_age - self.current_age+1
        # Create a Plotly figure
        fig = go.Figure()

        # Add the mean balance line
        fig.add_trace(go.Scatter(x=years, y=mean[:max_years], mode='lines', name='Mean', line=dict(color='blue')))
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

def complete_graph():
    """ Make a grap with the compensation on the x axis and the saldo at the Yaxis. Every age has her own line
        This to make the break even point visible
    """
    # Define parameters
    start_compensation = 500
    max_compensation = 5000
    compensation_step = 100

    # Create a list of ages and compensations
    ages = list(range(20, 91,10))
    compensations = list(range(start_compensation, max_compensation + 1, compensation_step))

    # Create a DataFrame to store results
    pension_results = pd.DataFrame(index=compensations, columns=ages)

    # Loop through each combination of age and compensation
    for age in ages:
        print (age)
        for compensation in compensations:
            calculator = PensionCalculator(age, compensation)
            result = calculator.calculate_pension("simulate")
            pension_results.loc[compensation, age] = result  # Fill in the result


    # Make sure to reset the index to get the compensation as a column
    pension_results.reset_index(inplace=True)
    pension_results.rename(columns={'index': 'Compensation'}, inplace=True)

    # Melt the DataFrame to long format for Plotly
    pension_results_melted = pension_results.melt(id_vars='Compensation', var_name='Age', value_name='Result')

    # Plot results with different lines for each age
    fig = px.line(pension_results_melted, 
                x='Compensation', 
                y='Result', 
                color='Age',  # Different lines for each age
                labels={'Compensation': 'Monthly Compensation', 'Result': 'Pension Result'},
                title='Pension Results by Compensation for Different Ages')
    # Add a thick horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=pension_results['Compensation'].min(),  # Start of the line on x-axis
        y0=0,  # Start of the line on y-axis
        x1=pension_results['Compensation'].max(),  # End of the line on x-axis
        y1=0,  # End of the line on y-axis
        line=dict(color='Red', width=2)  # Color and thickness of the line
    )
    # Show the plot in Streamlit
    st.plotly_chart(fig)

def optimizer():
    """Find break even point (when the endsaldo is 0) given a certain age
    """    
    # Define parameters
    tolerance = 100000  # Tolerance for near-zero results
    start_compensation = 0
    max_compensation = 5000
    compensation_step = 20
    age_list, compensation_list, result_list = [],[],[]
    # Create a list of ages and compensations
    ages = list(range(20, 101,5))
    min_compensation = 0
    # Create a DataFrame to store results
   
    current_compensation =0
    # Loop through each combination of age and compensation
    for age in ages:
        temp_saldo = 1000000000
        if age %10 ==0:
            print (age)
        for compensation in range(min_compensation, max_compensation,compensation_step):
            calculator = PensionCalculator(age, compensation)
            result = calculator.calculate_pension("simulate")
                       
            if result <0:
                if abs(result) < abs(temp_saldo):
                    temp_saldo = result
                else:
                    age_list.append(age)
                    compensation_list.append(compensation)
                    result_list.append(result)
                    # min_compensation = compensation
                    print(f"Found combination: Age={age}, Compensation={compensation}, Result={result}")
                    break
                    
            else:
                if abs(result) < abs(temp_saldo):
                    temp_saldo = abs(result)
                else:
                    age_list.append(age)
                    compensation_list.append(compensation)
                    result_list.append(result)
                    min_compensation = compensation
                    print(f"Found combination: Age={age}, Compensation={compensation}, Result={result}")
                    
                    break

    
    print(age_list)
    print(compensation_list)
    print(result_list)


    results_df = pd.DataFrame({
        'Age': age_list,
        'Compensation': compensation_list,
        'Result': result_list
    })
    # Plot results in a scatter plot
    # Plot results in a scatter plot with mouseover tooltips
    fig = px.scatter(results_df, 
                    x='Age', 
                    y='Compensation', 
                    labels={'Age': 'Age', 'Compensation': 'Monthly Compensation'},
                    title=f'Combinations of Age and Compensation with Pension Result',
                    hover_data=['Result'])  # Adding
    st.plotly_chart(fig)
    complete_graph()
    
def main():
    modus = st.sidebar.selectbox("Modus",["calculator", "optimizer", "complete graph"])
    if modus =="calculator":
        calculator = PensionCalculator()
        calculator.interface()
      
        #if st.button("GO"):
            
        results = calculator.calculate_pension("calculate")
        calculator.show_output(results)
    elif modus == "optimizer": 
        optimizer()
    else:
        complete_graph()

    
    #calculator.plot_values_with_confidence_intervals("balance_values")
    #calculator.show_ages_at_death(calculator.num_simulations, calculator.sexe, calculator.current_age)
    #calculator.show_total_balance()
    st.info("Not a serious financial advice. Use with care.\nBased on mortality chances in the Netherlands, 2024 (https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2)")
    st.info("https://www.bbc.com/news/magazine-33326787")

if __name__ == "__main__":
    #os.system('cls')
    print(f"--------------{datetime.datetime.now()}-------------------------")
    main()
    
    #scipy()
    