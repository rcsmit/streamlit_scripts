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
        self.current_age = 47
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
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,100)
        self.new_method =  True # st.sidebar.selectbox("Use AG table", [True, False],0)
        self.print_individual =  st.sidebar.selectbox("Print individual runs", [True, False],1)

    def calculate_pension(self, num_simulations):
        """Calculate the pension and balances

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
        """
        deceased_ages= []
        if self.sexe== "male":
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2024DefinitiefGevalideerd_male.csv")
        else:
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2024DefinitiefGevalideerd_female.csv")

    
       
        for _ in range(num_simulations):
            # Calculate the percentage completion
            completion_percentage = (_ + 1) / num_simulations * 100

            # Check if the current iteration is a multiple of 10%
                
            # display_progress_bar(_,num_simulations)
            person_alive = True
            for j in range(0, (10*self.max_age - self.current_age) + 1):
                if person_alive:
                    age_ = (10*self.current_age) +j
                    # use of the AG2022 table. 
                    if age_>1000:
                        # values not given in the table. Rounded to 0.6
                        probability_to_die = 0.6
                    else:
                        probability_to_die = (df_prob_die[str(self.current_year)].to_numpy()[df_prob_die['age'].to_numpy() == int(age_/10)].item())/10
                    age= age_/10
                    if random.random() <= probability_to_die:
                        deceased_ages.append(age)
                        # print (f"Person {_} died at {age} years")
                        person_alive = False
                    # use of the AG2022 table
            # if person_alive:
            #     print (f" person {_} is still alive")
  
        # Store all results in the instance variable
        
        self.deceased_ages = deceased_ages
        self.median_age_at_death = round(statistics.median(deceased_ages),1)
  
        sorted_ages = np.sort(deceased_ages)
      
        self.percentile_2_5 = np.percentile(sorted_ages, 2.5)
        self.percentile_95 = np.percentile(sorted_ages, 95)
        self.percentile_25 = np.percentile(sorted_ages, 25)
        self.percentile_75 = np.percentile(sorted_ages, 75)
        self.percentile_97_5 = np.percentile(sorted_ages, 97.5)

                  
            
        # Filter data for age 54
        age_54_data = df_prob_die[df_prob_die['age'] == self.current_age]

        # Create trace for age 54
        trace = go.Scatter(x=age_54_data.columns[1:],
                        y=age_54_data.iloc[0, 1:],
                        mode='lines',
                        name=f'Age {age_54_data.iloc[0, 0]}')

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
            st.write("______________________________")
            if c =="cdf":
                l = [10,25,50,75,95,99]
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
                st.write(f"{exact_probability}% probability {verb} at {age_at_prob} years (in {round(age_at_prob-.current_age,2)} years)")
            st.plotly_chart(fig)

        st.write(f"Average age at death of {num_simulations} individuals ({self.sexe}): {round(sum(self.deceased_ages)/len(self.deceased_ages),1)} - SD {round(np.std(self.deceased_ages),1)}")
        st.write(f"Median age at death: {round(statistics.median(self.deceased_ages),1)}")
        st.write (f"2.5% Percentile: {self.percentile_2_5:.2f} / 95% Percentile: {self.percentile_95:.2f} / 97.5% Percentile: {self.percentile_97_5:.2f}")
        st.write(f"Sum of persons {end_table['frequency'].sum()}")
    
    
      
def main():
    st.title("Dutch life expectancy and mortality/survival chances")
    calculator = PensionCalculator()

    calculator.interface()
    calculator.calculate_pension(calculator.num_simulations)
    # calculator.show_ages_at_death(calculator.num_simulations, calculator.sexe, calculator.current_age)

    #st.info("Projections Life Table AG2022 } https://www.actuarieelgenootschap.nl/kennisbank/ag-l-projections-life-table-ag2022.htm")
    st.info("Projections Life Table AG2024 https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2")

   
if __name__ == "__main__":
    main()
