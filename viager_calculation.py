import statistics
import random
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import time    

class Person:
    def __init__(self, age, sex):
        self.age = age
        self.sex = sex
        
        self.is_alive = True

    def age_up(self):
        if self.is_alive:
            self.age += 1

    def check_death(self, probability_to_die):
        if random.random() <= probability_to_die:
            self.is_alive = False
            return True
        return False
    
class PensionCalculator:
    def __init__(self, current_age=90,monthly_compensation_original=2000 ):
        # Initialize default values
        self.person = None

        self.house_value_at_start = 200_000
        self.pay_out_at_start = 0
        self.monthly_compensation_original = monthly_compensation_original
        self.monthly_contribution_original_how = "with inflation"
        
        self.current_age = current_age
       
        self.sex = "male"
        self.annual_return_rate =2.0
        self.annual_return_rate_sd = 0.0
        self.inflation = 2.62
        self.inflation_sd = 0.0
        self.cost_house = 1.0
        self.additional_monthly_need_how = "with inflation"
        self.num_simulations=10
        self.new_method =  True
        self.print_individual =  False
        self.start_compensation = 0 
        self.max_compensation = 10_000 
        self.compensation_step = 100 

        self.start_age = 30 
        self.max_age = 120 
        self.age_step = 10
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
        self.sex = st.sidebar.selectbox("sex", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)
      
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

        self.start_compensation = st.sidebar.number_input("Start compensation",-100000,100000,self.start_compensation)
        self.max_compensation = st.sidebar.number_input("Max compensation",0,100000,self.max_compensation)
        self.compensation_step = st.sidebar.number_input("Compensation step",0,10000,self.compensation_step)

        self.start_age = st.sidebar.number_input("Start age",0,10000,self.start_age)
        self.max_age_ = st.sidebar.number_input("Max age",0,10000,self.max_age)
        if self.current_age >= self.max_age :
            st.error(f"Max age has to be at least {self.current_age}.")
            st.stop()
        self.age_step = st.sidebar.number_input("age step",0,10000,self.age_step)
        self.max_age = self.max_age_ +self.age_step #include the last values in the graphs
    def load_life_table(self):
        base_url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/"
        file_name = f"AG2024DefinitiefGevalideerd_{self.sex}.csv"
        self.df_prob_die = pd.read_csv(base_url + file_name)
    # def check_death(self,age,year_simulation):
    #     # use of the AG2022 table
    #     if age>100:
    #         # values not given in the table. Rounded to 0.6
    #         probability_to_die = 0.6
    #     else:
    #         probability_to_die = self.df_prob_die[str(year_simulation)].to_numpy()[self.df_prob_die['age'].to_numpy() == age] #.item()
        
    #     if random.random() <= probability_to_die:
    #         return True
    #     else:
    #         return False

        
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
            
            single_sim_result = self.run_single_simulation()
            #print (single_sim_result)
            #if single_sim_result is not None:
            if 1==1:
                results.append(single_sim_result['annual_data'])
                deceased_ages.append(single_sim_result['age_at_death'])
                saldo_at_death_values.append(single_sim_result['saldo_at_death'])
                paid_out_values.append(single_sim_result['total_paid_out'])
                value_house_values.append(single_sim_result['final_house_value'])
                costs_house_values.append(single_sim_result['total_house_costs'])
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
        person = Person(self.current_age, self.sex)
        
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
        #for j in range(1, int(self.max_age - self.current_age + 1)):
        while person.is_alive and person.age <= self.max_age:   
            #if person_alive:
            value_house = value_house * ((100+ annual_return_rate) / 100)
            cost_house_amount = round(value_house * (self.cost_house / 100), 2)
            balance -= cost_house_amount 
            monthtly_compensation = monthtly_compensation * ((100+inflation)/100)
            paid_out +=monthtly_compensation*12
            balance -= 12 * monthtly_compensation 
            costs_house +=cost_house_amount
            
            annual_data['compensation'].append(int(12 * monthtly_compensation))
            annual_data['balance'].append(int(balance))
            annual_data['interest'].append(int((100 + annual_return_rate) / 100))

            #if self.check_death(age,year_simulation):
            probability_to_die = self.get_probability_to_die(person.age, year_simulation)
            
            if person.check_death(probability_to_die):
                balance += value_house
                
                return {
                'annual_data': annual_data,
                'age_at_death': person.age,
                'saldo_at_death': int(balance),
                'total_paid_out': int(paid_out),
                'final_house_value': int(value_house),
                'total_house_costs': int(costs_house)
            }

            person.age_up()
            year_simulation += 1
        st.error(f"Person dead or too old  {person.age=}  {self.max_age=} ")
        st.stop()
            

    def get_probability_to_die(self, age, year_simulation):
        # Implement this method to get the probability of dying from your life table
        # You might need to adjust your life table loading and accessing method
        if age > 100:
            return 0.6
        return self.df_prob_die[str(year_simulation)].to_numpy()[self.df_prob_die['age'].to_numpy() == age].item()
    
       
    def show_output(self,calc_results):
        #print (calc_results)
        if sum(calc_results["saldo_at_death_values"]) > 0:
            st.info(f'Average saldo at the death of  {self.num_simulations} persons ({self.sex}) : {round(sum(calc_results["saldo_at_death_values"])/len(calc_results["saldo_at_death_values"]))}. Total Profit for viager buyer : {round(sum(calc_results["saldo_at_death_values"]))}')# - SD {round(np.std(saldo_at_death_values),1)}")
        else:
            st.info(f'Average saldo at the death of  {self.num_simulations} persons ({self.sex}) : {round(sum(calc_results["saldo_at_death_values"])/len(calc_results["saldo_at_death_values"]))}. Total Loss for viager buyer : {round(sum(calc_results["saldo_at_death_values"]))}')
        col1,col2= st.columns([4,6])
        with col1:
            self.show_ages_at_death(self.num_simulations, self.sex, calc_results["deceased_ages"],self.current_age)
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
   
    def show_ages_at_death(self, num_simulations,sex, deceased_ages, current_age):
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
                name  =f'Cumulative Distribution Function (CDF) of Ages ({sex} - {current_age})'
                name2 = "CDF"
                verb = "to be deceased"
            else:
                #  Complementary Cumulative Distribution Function (CCDF),
                l = [50,25,5,1]
                name = f"Survival function ({sex} - {current_age})"
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

        st.write(f"Average age at death of {num_simulations} individuals ({self.sex}): {round(sum(deceased_ages)/len(deceased_ages),1)} - SD {round(np.std(deceased_ages),1)}")
        st.write(f"Median age at death: {round(statistics.median(deceased_ages),1)}")
        st.write (f"2.5% Percentile: {percentile_2_5:.2f} / 95% Percentile: {percentile_95:.2f} / 97.5% Percentile: {percentile_97_5:.2f}")
      
        st.write(f"Sum of persons {end_table['frequency'].sum()}")

def complete_graph(self):
    """ Make a grap with the compensation on the x axis and the saldo at the Yaxis. Every age has her own line
        This to make the break even point visible
    """
    # Define parameters
    ages = list(range(self.start_age, self.max_age,self.age_step))
    compensations = list(range(self.start_compensation, self.max_compensation + 1, self.compensation_step))


    # Create a DataFrame to store results
    pension_results = pd.DataFrame(index=compensations, columns=ages)

    # Loop through each combination of age and compensation
    for age in ages:
        print (age)
        for compensation in compensations:
            self.current_age = age
            self.monthly_compensation_original = compensation
          
            #calculator = PensionCalculator(age, compensation)
            result = self.calculate_pension("simulate")
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

def optimizer(self):
    """Find break even point (when the endsaldo is 0) given a certain age
    """    
    # Define parameters

    # Create a list of ages and compensations
    ages = list(range(self.start_age, self.max_age,self.age_step))
    compensations = list(range(self.start_compensation, self.max_compensation + 1, self.compensation_step))

    age_list, compensation_list, result_list = [],[],[]
   
   
    # Loop through each combination of age and compensation
    for age in ages:
        temp_saldo = 1000000000
        if age %10 ==0:
            print (age)
        for compensation in compensations:
            self.current_age = age
            self.monthly_compensation_original = compensation
          
            
            result = self.calculate_pension("simulate")
                       
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
    

def main():
    s1 = int(time.time())
    st.header("Viager calculation")
    modus = st.sidebar.selectbox("Modus",["calculator", "optimizer", "complete graph"])
    calculator = PensionCalculator()
    calculator.interface()
    if modus =="calculator":

        results = calculator.calculate_pension("calculate")
        calculator.show_output(results)
    elif modus == "optimizer": 
        if st.button("GO"):
            optimizer(calculator)
    else:
        if st.button("GO"):
            complete_graph(calculator)

    st.info("Inspired by : A 90-year-old woman signed a deal with a 47-year-old lawyer to give him her apartment upon her death in exchange for monthly payments. She outlived him, and his widow continued the payments. She received more than double the apartment's value \n https://www.threads.net/@unbfacts/post/DBTbibYuzYy")
    st.info("Not a serious financial advice. Use with care.\nBased on mortality chances in the Netherlands, 2024 (https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2)")
    st.info("https://www.bbc.com/news/magazine-33326787")
    s2 = int(time.time())
    st.write(f"Used time : {s2-s1} sec.")
if __name__ == "__main__":
    #os.system('cls')
    print(f"--------------{datetime.datetime.now()}-------------------------")
    main()
    
    #scipy()
    