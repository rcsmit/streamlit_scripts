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
        self.sexe = st.sidebar.selectbox("Sexe", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)

        st.sidebar.subheader("--- Simulations ---")
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,10_000)
        self.new_method =  True # st.sidebar.selectbox("Use AG table", [True, False],0)
        self.print_individual = False #  st.sidebar.selectbox("Print individual runs", [True, False],1)
        self.ag_jaar = 2024#  st.sidebar.selectbox("Year AG table", ["2022","2024"],1)
        self.startjaar =  st.sidebar.number_input("Start Yaar",2022,2100,self.current_year)
    def calculate_life_expectancy(self):
        """Calculate life expectancy

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
            ag_jaar (str): year as string
        """
        

        
    
        self.monte_carlo_simulation()
        self.berekening_laning()
        
    def berekening_laning(self):
        st.subheader("Berekening Laning")
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
        
        st.success(
            f"**If everything stays as it is in {self.startjaar}**\n\n"
            f"From age **{start_age}**, you are expected to live another "
            f"**{period_life_expectancy} years** on average.\n"
            f"This corresponds to an expected age of "
            f"**{start_age + period_life_expectancy:.1f} years** "
            f"(period life expectancy)."
        )

        st.success(
            f"**If your life course is followed over time**\n\n"
            f"From age **{start_age}**, you are expected to live another "
            f"**{cohort_life_expectancy} years** on average.\n"
            f"This corresponds to an expected age of "
            f"**{start_age + cohort_life_expectancy:.1f} years** "
            f"(cohort life expectancy)."
        )

        # # Print resultaten
        # st.success(f"Periodelevensverwachting  vanaf leeftijd {start_age} voor {self.startjaar}: {period_life_expectancy} -> eindleeftijd: {start_age+period_life_expectancy}")
        # st.success(f"Cohortlevensverwachting vanaf leeftijd {start_age} vanaf {self.startjaar}: {cohort_life_expectancy} -> eindleeftijd: {start_age+cohort_life_expectancy}") # 34,93 
       
    def monte_carlo_simulation(self):
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
        expected_ages = np.arange(self.current_age, self.max_age -5 , 1)
        present_ages = np.sort(end_table["ages"].unique())

        missing_ages = np.setdiff1d(expected_ages, present_ages)

        if len(missing_ages) > 0:
            st.error(
                f"The results do not cover all ages between "
                f"{self.current_age} and {self.max_age-5}. "
                f"Increase the number of simulations for a more reliable result."
            )
            #st.stop()
        
        col1,col2=st.columns([1,3])
        with col1:
            st.metric(
                label="Most likely age at death",
                value=f"{self.median_age_at_death} years",
                delta=f"{round(self.median_age_at_death - self.current_age,1)} years from now"
            )
        with col2:
            st.info(
                f"You have a **50% chance** to live beyond **{self.median_age_at_death}**. "
                f"About **90% of people** with similar characteristics die between "
                f"**{round(self.percentile_2_5,1)}** and **{round(self.percentile_97_5,1)}**."
            )
        result_table = end_table.copy()

        result_table["year"] = 2026 + (result_table["ages"] - self.current_age).astype(int)
        
        result_table["perc_died"] =  result_table["frequency"] / self.num_simulations *100
        result_table["perc_alive_cumm"] =  result_table["cdf_1"] 
        result_table["perc_death_cumm"] =  result_table["cdf"] 
        result_table = result_table[["ages", "year", "perc_died", "perc_alive_cumm", "perc_death_cumm"]]
        
        self.show_end_info(end_table)
        # self.plot_probability_over_time(df_prob_die)        
        self.plot_age_freq_bar_graph(end_table)
        self.plot_cdf_survival_function(end_table)
        st.write(result_table)

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
        age_counts["ages"] = age_counts["ages"].astype(float) # prevent You are merging on int and float columns where the float values are not equal to their int representation.

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
        # st.write(f"Average age at death of {self.num_simulations} individuals ({self.sexe}): {round(sum(self.deceased_ages)/len(self.deceased_ages),2)} 
        # [in {round(sum(self.deceased_ages)/len(self.deceased_ages)-self.current_age,2)} years] -
        #  SD {round(np.std(self.deceased_ages),2)}")
        # st.write(f"Median age at death: {round(statistics.median(self.deceased_ages),2)} [in {round(statistics.median(self.deceased_ages)-self.current_age,2)} years]")
        # st.write (f"2.5% Percentile: {self.percentile_2_5:.2f} / 95% Percentile: {self.percentile_95:.2f} / 97.5% Percentile: {self.percentile_97_5:.2f}")
        # st.write(f"Sum of persons {end_table['frequency'].sum()}")
        # st.sidebar.info(f"Projections Life Table AG{self.ag_jaar} https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag{self.ag_jaar}-2")

        mean_age = round(sum(self.deceased_ages) / len(self.deceased_ages), 1)
        mean_years_left = round(mean_age - self.current_age, 1)
        sd_age = round(np.std(self.deceased_ages), 1)

        median_age = round(statistics.median(self.deceased_ages), 1)
        median_years_left = round(median_age - self.current_age, 1)
        st.subheader("Results")
        st.write(f"{int(end_table['frequency'].sum()):,} simulations")
        st.write(
            f"**Most likely outcome**\n\n"
            f"The median age at death is **{median_age} years**. "
            f"Half of people die before this age, half live longer "
            f"(about **{median_years_left} years from now**)."
        )
        st.write(
            f"**Average outcome**\n\n"
            f"The average age at death is **{mean_age} years**, which is about **{mean_years_left} years from now**.\n\n"
            f"The spread (SD) around this average is **{sd_age} years**."
        )

        st.write(
            f"**Uncertainty range**\n\n"
            f"• 2.5% die before **{self.percentile_2_5:.1f} years**\n"
            f"• 95% die before **{self.percentile_95:.1f} years**\n"
            f"• 97.5% die before **{self.percentile_97_5:.1f} years**"
        )

       
    def plot_cdf_survival_function(self, end_table):
        col_cdf, col_survival = st.columns(2)

        with col_cdf:
            st.subheader("Chance deceased")

        with col_survival:
            st.subheader("Chance alive")

        for c, col in zip(["cdf", "cdf_1"], [col_cdf, col_survival]):
            with col:
                
                st.write("______________________________")
                if c =="cdf":
                    l = [1,2.5,5,10,25,50,75,95,99]
                    # name  =f'Cumulative Distribution Function (CDF) of Ages ({self.sexe} - {self.current_age})'
                    # name2 = "CDF"
                    name = f"Chance you have died ({self.sexe}, age {self.current_age})"
                    name2 = "Chance deceased (%)"
                    verb = "to be deceased"
                else:
                    #  Complementary Cumulative Distribution Function (CCDF),
                    l = [99,97.5,95,90,75,50,25,5,1]
                    # name = f"Survival function ({self.sexe} - {self.current_age})"
                    # name2 = "CCDF"
                    name = f"Chance you are still alive ({self.sexe}, age {self.current_age})"
                    name2 = "Chance alive (%)"
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
                    # exact_probability = round((end_table.loc[end_table['ages'] == age_at_prob, c].values[0]),1)

                    # Interpolate age at given probability level
                
                    if c =="cdf":
                        x = end_table[c].values
                        y = end_table["ages"].values
                        age_at_prob = round(np.interp(prob, x, y),1)
                    else:
                        x = end_table["cdf_1"].values   # chance still alive (%)
                        y = end_table["ages"].values
                        age_at_prob = round(np.interp(prob, x[::-1], y[::-1]),1)
                    # Interpolated probability (will be ~prob by definition)
                    exact_probability = round(prob, 1)
                    # Add vertical line at age where cumulative probability is 0.5
                    fig.add_vline(x=age_at_prob, line_dash="dash", line_color="red", annotation_text=f"{exact_probability}")
                    # st.write(f"{exact_probability}% probability {verb} at {age_at_prob} years (in {round(age_at_prob-self.current_age,2)} years)")
                    st.write(
                                f"At age **{age_at_prob}**, the chance is **{exact_probability}%** that you are {verb}."
                            )
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
            title=(
                f"Age at death distribution "
                f"({self.sexe}, age {self.current_age}, "
                f"{self.num_simulations:,} simulations)"
            ),
            xaxis_title="Ages",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig3)

    
      
def main():
    tab1,tab2= st.tabs(["Life Expectancy NL", "About this app"])
    with tab2:
        show_info()
    with tab1:
        st.title("Your life expectancy at a glance")    
        # st.info(
        #     "This tool gives an estimate of how old you will become, based on official Dutch life expectancy data.\n\n"
        #     "Enter your age and gender, and the app shows how long people like you usually live."
        # )
        calculator = LifeExpectancyCalculator()

        calculator.interface()
        calculator.calculate_life_expectancy()
    
def show_info():
    st.header("Info")

    st.info(
        "**Method used in this app**\n\n"
        "The results are based on a Monte Carlo simulation. This means the app "
        "simulates many individual life paths year by year. For each simulated year, "
        "it uses the probability of death that matches both your age and the calendar year.\n\n"
        "By repeating this process many times, the app builds a realistic distribution "
        "of possible ages at death, including averages, medians, and uncertainty ranges."
    )

    st.info(
        "**Note on interpretation**\n\n"
        "Life expectancy is a statistical expectation, not a prediction for any single person.\n\n"
        "Individual outcomes can vary widely due to lifestyle, health, chance, and other factors.\n\n"
        "The estimates are based on Dutch population statistics and actuarial life tables.")

    st.info(
        "**Period vs cohort life expectancy**\n\n"
        "**Period life expectancy** answers the question:\n"
        "“How long would I live if death rates stayed exactly the same as in one specific year?”\n\n"
        "It is a snapshot based on a single calendar year and does not account for future improvements "
        "in health care or living conditions.\n\n"
        "**Cohort life expectancy** answers a different question:\n"
        "“How long am I likely to live, given that death rates may change as time passes?”\n\n"
        "This approach follows a person through future years and reflects expected improvements over time. "
        "For individuals, cohort life expectancy is usually more realistic."
    )


   
    st.info(
        f"**Data source**\n\n"
        f"This tool uses the official Dutch Actuarial Life Tables "
        f"(AG2024), published by the Royal Dutch Actuarial Association. "
        f"These tables contain age- and year-specific probabilities of death.\n\n"
        f"Source: https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2"
    )
    st.info(
        "**Related methods**\n\n"
        "Statistics Netherlands (CBS) uses related but different statistical techniques "
        "for national life expectancy estimates.\n\n"
        "Background reading: "
        "https://pure.rug.nl/ws/portalfiles/portal/13869387/stoeldraijer_et_al_2013_DR.pdf"
    )

if __name__ == "__main__":
    main()
