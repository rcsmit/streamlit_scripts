import statistics
import random
import time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import datetime

# THIS USES AG DATA


class LifeExpectancyCalculator:
    def __init__(self):
        # Initialize default values
        self.current_age = 48
        self.max_age= 110
        # Get the current date and time
        current_datetime = datetime.datetime.now()
        # Extract the current year from the datetime object
        self.current_year = current_datetime.year

    def interface(self):
        st.sidebar.subheader("--- The person ---")
        self.sexe = st.sidebar.selectbox("sexe", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)

        st.sidebar.subheader("--- Simulations ---")
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,10_000)
        self.new_method =  True # st.sidebar.selectbox("Use AG table", [True, False],0)
        self.print_individual =  st.sidebar.selectbox("Print individual runs", [True, False],1)
        self.ag_jaar =  st.sidebar.selectbox("Year AG table", ["2022","2024"],1)
        self.startjaar =  st.sidebar.number_input("startjaar",2022,2100,2026)
    def calculate_life_expectancy(self):
        """Calculate life expectancy

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
            ag_jaar (str): year as string
        """
        

        self.berekening_laning()
        
    
        self.monte_carlo_simulation()

    def berekening_laning(self):

        if self.sexe== "male":
            df_prob_die = pd.read_csv(f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG{self.ag_jaar}DefinitiefGevalideerd_male.csv", index_col=0)
        else:
            df_prob_die = pd.read_csv(f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG{self.ag_jaar}DefinitiefGevalideerd_female.csv", index_col=0)

        df_prob_die.columns = df_prob_die.columns.astype(int)
        # Cumulatieve overlevingskansen berekenen vanaf de startleeftijd (bijvoorbeeld 45)
        start_age = self.current_age
      
        
        # Cumulatieve overlevingskansen berekenen vanaf de startleeftijd (bijvoorbeeld 45)
        
        survival_prob = pd.DataFrame(index=df_prob_die.index, columns=df_prob_die.columns)
        survival_prob.loc[start_age] = 1  # Start bij 100% overleving op de startleeftijd
        startkans =  df_prob_die.at[self.current_age, self.startjaar]
        for year in df_prob_die.columns:
            for age in df_prob_die.index[df_prob_die.index > start_age]:
                survival_prob.at[age, year] = survival_prob.at[age - 1, year] * (1 - df_prob_die.at[age-1, year])
        survival_prob.columns = survival_prob.columns.astype(int)
       
        # Bereken de periodelevensverwachting voor een specifiek jaar vanaf leeftijd 45
        period_life_expectancy = round( survival_prob[self.startjaar][start_age:].sum() - 0.5,2)

        # Bereken de cohortlevensverwachting vanaf leeftijd 45
        cohort_life_expectancy = 0
        cohort_survival = 1  # Start bij 100% overleving vanaf leeftijd 45
        for i, year in enumerate(range(self.startjaar, self.startjaar + len(df_prob_die.index))):
            current_age = start_age + i
            # print (current_age)
            if current_age in df_prob_die.index and year in df_prob_die.columns:
                cohort_survival *= (1 - df_prob_die.at[current_age, year])
                # print (cohort_survival)
                cohort_life_expectancy += cohort_survival
                # print (cohort_life_expectancy)

        cohort_life_expectancy -= 0.5  # Halvering van het laatste jaar
        cohort_life_expectancy=round(cohort_life_expectancy,2)
        
        # Print resultaten
        st.success(f"Periodelevensverwachting vanaf leeftijd {start_age} voor {self.startjaar}: {period_life_expectancy} -> eindleeftijd: {start_age+period_life_expectancy}")
        st.success(f"Cohortlevensverwachting vanaf leeftijd {start_age} vanaf {self.startjaar}: {cohort_life_expectancy} -> eindleeftijd: {start_age+cohort_life_expectancy}") # 34,93 
       
    def monte_carlo_simulation(self):
        st.subheader("Monte carlo simulation")
        deceased_ages= []
       
        if self.sexe== "male":
            df_prob_die = pd.read_csv(f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG{self.ag_jaar}DefinitiefGevalideerd_male.csv")
        else:
            df_prob_die = pd.read_csv(f"https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG{self.ag_jaar}DefinitiefGevalideerd_female.csv")
        df_prob_die.set_index('age', inplace=True)
        
        df_prob_die.columns = df_prob_die.columns.astype(int)
        # Stel de eerste kolom in als index
        placeholder=st.empty()
        for a in range(self.num_simulations):
            if a % 10==0:
                placeholder.info(f"{a+1}/{self.num_simulations}")
            
            year = self.startjaar
            for age in df_prob_die.index[df_prob_die.index >=  self.current_age]:

                    probability_to_die =  df_prob_die.at[age, year]
                                        
                    if random.random() <= probability_to_die:
                        deceased_ages.append(age)
                       
                        break
                    year+=1
           
  
        # Store all results in the instance variable
        placeholder.empty()
        self.deceased_ages = deceased_ages
        self.median_age_at_death = round(statistics.median(deceased_ages),1)
  
        sorted_ages = np.sort(deceased_ages)
      
        self.percentile_2_5 = np.percentile(sorted_ages, 2.5)
        self.percentile_95 = np.percentile(sorted_ages, 95)
        self.percentile_25 = np.percentile(sorted_ages, 25)
        self.percentile_75 = np.percentile(sorted_ages, 75)
        self.percentile_97_5 = np.percentile(sorted_ages, 97.5)

        end_table = self.calculate_age_distribution()
         
        self.show_end_info(end_table)
        self.plot_probability_over_time(df_prob_die)
        
        self.plot_age_freq_bar_graph(end_table)

                
    
    
        self.plot_cdf_survival_function(end_table)
        
        return end_table




    def calculate_age_distribution(self):
        """_summary_

        calcualte age distrbution
        caclulate cdf
        calculate survival function

        Args:
            deceased_ages (_type_): _description_

        Returns:
            _type_: _description_
        """        
        df_deceased = pd.DataFrame({'ages': self.deceased_ages})

        # Generate the sequence of ages with steps of 0.1
        ages_with_steps = np.arange(self.current_age, self.max_age + 0.1, 0.1)

        # Create a DataFrame
        all_ages = pd.DataFrame({'ages': ages_with_steps})
        #all_ages = pd.DataFrame({'ages':  range(self.current_age, self.max_age+ 1)})

        # Count the frequency of each age in the 'deceased_ages' list
        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='right')#.fillna(0)

        end_table = end_table.sort_values(by='ages')

        # Calculate cumulative sum of frequencies
        end_table['cumulative_frequency'] = end_table['frequency'].cumsum()

        # Normalize cumulative sum to get proportions
        end_table['cdf'] = end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max()*100
        # Normalize cumulative sum to get proportions
        end_table['cdf_1'] = 100-(end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max())*100
        return end_table
    
    def plot_probability_over_time(self, df_prob_die):        
        # Filter data for age x
        age_x_data = df_prob_die[df_prob_die.index == self.current_age]
        trace = go.Scatter(x=age_x_data.columns[1:],
                        y=age_x_data.iloc[0, 1:],
                        mode='lines',
                        name=f'Age {age_x_data.iloc[0, 0]}')

        # Create layout
        layout = go.Layout(title=f'Probability to Die Over Time for Age {self.current_age}',
                        xaxis=dict(title='Year'),
                        yaxis=dict(title='Probability'),
                        hovermode='closest',
                        showlegend=True)

        # Create figure
        fig = go.Figure(data=[trace], layout=layout)

        # Show plot
        st.plotly_chart(fig)  
    def show_end_info(self, end_table):
        st.write(f"Average age at death of {self.num_simulations} individuals ({self.sexe}): {round(sum(self.deceased_ages)/len(self.deceased_ages),2)} [{round(sum(self.deceased_ages)/len(self.deceased_ages)-self.current_age,2)}] - SD {round(np.std(self.deceased_ages),2)}")
        st.write(f"Median age at death: {round(statistics.median(self.deceased_ages),2)} [{round(statistics.median(self.deceased_ages)-self.current_age,2)}]")
        st.write (f"2.5% Percentile: {self.percentile_2_5:.2f} / 95% Percentile: {self.percentile_95:.2f} / 97.5% Percentile: {self.percentile_97_5:.2f}")
        st.write(f"Sum of persons {end_table['frequency'].sum()}")
        st.info(f"Projections Life Table AG{self.ag_jaar} https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag{self.ag_jaar}-2")

    def plot_cdf_survival_function(self, end_table):
        for c in ["cdf", "cdf_1"]:
            st.write("______________________________")
            if c =="cdf":
                l = [1,2.5,5,10,25,50,75,95,99]
                name  =f'Cumulative Distribution Function (CDF) of Ages ({self.sexe} - {self.current_age})'
                name2 = "CDF"
                verb = "to be deceased"
            else:
                #  Complementary Cumulative Distribution Function (CCDF),
                l = [99,95,90,75,50,25,5,1]
                name = f"Survival function ({self.sexe} - {self.current_age})"
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
                st.write(f"{exact_probability}% probability {verb} at {age_at_prob} years (in {round(age_at_prob-self.current_age,2)} years)")
            st.plotly_chart(fig)

    def plot_age_freq_bar_graph(self, end_table):
        vlines = [statistics.median(self.deceased_ages), self.percentile_2_5, self.percentile_25, self.percentile_75, self.percentile_97_5]
        vtxt = ["median", "2,5%", "25%", "75%", "97,5%"]
      
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

    
      
def main():
    st.title("Dutch life expectancy and mortality/survival chances")
    st.info("This script is a Streamlit application that simulates life expectancy and mortality using data from Dutch actuarial tables (AG2024). It allows users to input parameters like age and gender and run simulations to predict the age of death for individuals.")
    calculator = LifeExpectancyCalculator()

    calculator.interface()
    calculator.calculate_life_expectancy()
    # calculator.show_ages_at_death(calculator.num_simulations, calculator.sexe, calculator.current_age)

    #st.info("Projections Life Table AG2022 } https://www.actuarieelgenootschap.nl/kennisbank/ag-l-projections-life-table-ag2022.htm")
    
    st.info("Statistics Netherlands (CBS) uses different techniques for life expectancy https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf")
   
if __name__ == "__main__":
    main()
