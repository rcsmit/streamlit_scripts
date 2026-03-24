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
        self.max_age= 110
        self.annual_return_rate =2.0
        self.annual_return_rate_sd = 0.0
        self.inflation = 2.62
        self.inflation_sd = 0.0
        self.taxes = 1.0
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

        current_datetime = datetime.datetime.now()
        self.current_year = current_datetime.year

    def interface(self):
        self.initial_one_time_contribution = st.sidebar.number_input("Initial One-Time Contribution:", value=self.initial_one_time_contribution)
        self.monthly_contribution_original = st.sidebar.number_input("Monthly Contribution (current pricelevel):", value=self.monthly_contribution_original)
        self.monthly_contribution_original_how = st.sidebar.selectbox("Monthly contribution How", ["with inflation", "without inflation"],0 )

        st.sidebar.subheader("--- The person ---")
        self.sexe = st.sidebar.selectbox("sexe", ["male", "female"],0)
        self.current_age = st.sidebar.number_input("Current Age:", value=self.current_age)
        self.retirement_age = st.sidebar.number_input("Retirement Age:", value=self.retirement_age)
        if self.current_age > self.retirement_age:
            st.error("You cannot be older than your retirement age")
            st.stop()
        self.max_age = st.sidebar.number_input("Maximum Age:", value=self.max_age)
        self.birthyear = self.current_year - self.current_age
        st.sidebar.write(f"Years to go to pension: {self.retirement_age - self.current_age}")
        
        st.sidebar.subheader("--- Rates ---") 
        self.annual_return_rate = st.sidebar.number_input("Annual Interest Rate (%):", value=self.annual_return_rate)
        self.inflation = st.sidebar.number_input("Average Annual Inflation Rate (%):", value=self.inflation)
        self.annual_return_rate_sd = st.sidebar.number_input("Annual Interest Rate SD(%):", value=self.annual_return_rate_sd)
        self.inflation_sd = st.sidebar.number_input("Average Annual Inflation Rate SD(%):", value=self.inflation_sd)
        self.taxes = st.sidebar.number_input("Taxes (%):", value=self.taxes)
        
        st.sidebar.subheader("--- Pension data ---")
        self.monthly_pension_without_reduction_original = st.sidebar.number_input("Monthly Pension without Reduction (current price level):", value=self.monthly_pension_without_reduction_original)
        self.years_shortfall = st.sidebar.number_input("Years of Shortfall:", value=self.years_shortfall)
        st.sidebar.write(f"Shortfall per month (current price level): {round(self.years_shortfall * 0.02 * self.monthly_pension_without_reduction_original)}")
        self.additional_monthly_need = st.sidebar.number_input("Additional Monthly Need (current price level):", value=self.additional_monthly_need)
        self.additional_monthly_need_how = st.sidebar.selectbox("Additional monthly needed How", ["with inflation", "without inflation"],0 )

        st.sidebar.subheader("--- Windfalls ---")
        self.windfall_1_year = st.sidebar.number_input("Windfall 1 (Year):", value=self.windfall_1_year)
        self.windfall_1_amount = st.sidebar.number_input("Windfall 1 (Amount):", value=self.windfall_1_amount)   
        self.windfall_2_year = st.sidebar.number_input("Windfall 2 (Year):", value=self.windfall_2_year)
        self.windfall_2_amount = st.sidebar.number_input("Windfall 2 (Amount):", value=self.windfall_2_amount)
        self.windfall_3_year = st.sidebar.number_input("Windfall 3 (Year):", value=self.windfall_3_year)
        self.windfall_3_amount = st.sidebar.number_input("Windfall 3 (Amount):", value=self.windfall_3_amount)

        st.sidebar.subheader("--- Simulations ---")
        self.num_simulations = st.sidebar.number_input("Number of simulations",1,10_000_000,100)
        self.new_method = True
        self.print_individual = st.sidebar.selectbox("Print individual runs", [True, False],1)

    def calculate_pension(self, num_simulations=1000):
        """Calculate the pension and balances

        Args:
            num_simulations (int, optional): num simulations. Defaults to 1000.
        """
        deceased_ages, saldo_at_death_values, results, start_year = [], [], [], self.current_year

        if self.sexe == "male":
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2024DefinitiefGevalideerd_male.csv")
        else:
            df_prob_die = pd.read_csv("https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/AG2024DefinitiefGevalideerd_female.csv")

        # Build windfall lookup dict to handle duplicate years correctly (FIX #2)
        windfall_lookup: dict[int, float] = {}
        for yr, amt in [
            (int(self.windfall_1_year), self.windfall_1_amount),
            (int(self.windfall_2_year), self.windfall_2_amount),
            (int(self.windfall_3_year), self.windfall_3_amount),
        ]:
            windfall_lookup[yr] = windfall_lookup.get(yr, 0) + amt

        s1, s2y = int(time.time()), int(time.time())
        for _ in range(num_simulations):
            completion_percentage = (_ + 1) / num_simulations * 100
            if completion_percentage % 10 == 0:
                s2x = int(time.time())
                print(f"Progress: {int(completion_percentage)}% complete [{_+1}/{num_simulations}] [Round : {s2x-s2y} seconds | Cummulative : {s2x-s1} seconds]")
                s2y = s2x

            person_alive = True
            annual_contribution_values = []
            balance_values = []
            interest_values = []
            annual_shortfall_values = []
            
            annual_contribution_original = self.monthly_contribution_original * 12
            annual_pension = self.monthly_pension_without_reduction_original * 12
            annual_shortfall_original = (self.years_shortfall * 0.02 * annual_pension) + (self.additional_monthly_need * 12)

            balance = self.initial_one_time_contribution
            years_until_retirement = self.retirement_age - self.current_age
            annual_return_rate = np.maximum(np.random.normal(self.annual_return_rate, self.annual_return_rate_sd), 0)
            inflation = np.maximum(np.random.normal(self.inflation, self.inflation_sd), 0)

            # --- Pre-retirement loop ---
            for i in range(0, years_until_retirement + 1):
                current_year = i + start_year  # correctly tracks calendar year

                if balance > 0:
                    interest = round(balance * (annual_return_rate / 100), 2)
                    taxes_amount = round(balance * (self.taxes / 100), 2)
                    balance += interest - taxes_amount
                else:
                    interest = 0

                balance += annual_contribution_original

                # FIX #2: use dict lookup so duplicate windfall years are summed correctly
                if i in windfall_lookup:
                    balance += windfall_lookup[i]

                annual_contribution_values.append(annual_contribution_original)
                balance_values.append(int(balance))
                interest_values.append(interest)
                annual_shortfall_values.append(0)

                if self.monthly_contribution_original_how == "with inflation":
                    annual_contribution_original = round(annual_contribution_original * (100 + inflation) / 100, 2)
                # FIX #3: removed dead inflation calc for "without inflation" branch (no-op was misleading)

                if person_alive:
                    age = self.current_age + i
                    probability_to_die = df_prob_die[str(current_year)].to_numpy()[df_prob_die['age'].to_numpy() == age].item()
                    if random.random() <= probability_to_die:
                        deceased_ages.append(age)
                        saldo_at_death_values.append(balance)
                        person_alive = False

            if self.additional_monthly_need_how == "with inflation":
                annual_shortfall_corrected = annual_shortfall_original * ((100 + inflation) / 100) ** years_until_retirement
            else:
                annual_shortfall_corrected = annual_shortfall_original

            # --- Post-retirement loop ---
            for j in range(years_until_retirement + 1, self.max_age - self.current_age + 1):
                current_year = j + start_year  # FIX #1: update calendar year each post-retirement year

                if balance > 0:
                    interest = balance * (annual_return_rate / 100)
                    taxes_amount = round(balance * (self.taxes / 100), 2)
                    balance += interest - taxes_amount
                else:
                    interest = 0

                balance -= annual_shortfall_corrected

                if self.additional_monthly_need_how == "with inflation":
                    annual_shortfall_corrected = annual_shortfall_corrected * ((100 + inflation) / 100)

                annual_contribution_values.append(0)
                balance_values.append(int(balance))
                interest_values.append(interest)
                annual_shortfall_values.append(annual_shortfall_corrected)

                if person_alive:
                    age = self.current_age + j
                    if age > 100:
                        probability_to_die = 0.6
                    else:
                        probability_to_die = df_prob_die[str(current_year)].to_numpy()[df_prob_die['age'].to_numpy() == age].item()
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

        self.results = results
        self.deceased_ages = deceased_ages
        self.median_age_at_death = round(statistics.median(deceased_ages), 1)

        sorted_ages = np.sort(deceased_ages)
        self.percentile_2_5 = np.percentile(sorted_ages, 2.5)
        self.percentile_95 = np.percentile(sorted_ages, 95)
        self.percentile_25 = np.percentile(sorted_ages, 25)
        self.percentile_75 = np.percentile(sorted_ages, 75)
        self.percentile_97_5 = np.percentile(sorted_ages, 97.5)

        st.write(f"Average saldo at the death of {num_simulations} persons ({self.sexe}) : {round(sum(saldo_at_death_values)/len(saldo_at_death_values))} - SD {round(np.std(saldo_at_death_values),1)}")
        if sum(saldo_at_death_values) > 0:
            st.write(f"Profit for pension funds : {round(sum(saldo_at_death_values))}")
        else:
            st.write(f"Loss for pension funds : {round(sum(saldo_at_death_values))}")

        s2 = int(time.time())
        st.write(f"Time needed: {s2-s1} seconds")

    def show_ages_at_death(self, num_simulations, sexe, current_age):
        """Show a graph of the age of death of people in the simulations"""
        df_deceased = pd.DataFrame({'ages': self.deceased_ages})
        all_ages = pd.DataFrame({'ages': range(self.current_age, self.max_age + 1)})

        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='left').fillna(0)

        vlines = [statistics.median(self.deceased_ages), self.percentile_2_5, self.percentile_25, self.percentile_75, self.percentile_97_5]
        vtxt = ["median", "2,5%", "25%", "75%", "97,5%"]

        fig3 = go.Figure(data=[go.Bar(x=end_table["ages"], y=end_table["frequency"])])
        for i, txt in zip(vlines, vtxt):
            fig3.add_shape(
                go.layout.Shape(
                    type="line",
                    x0=i, x1=i,
                    y0=0, y1=max(end_table["frequency"]),
                    line=dict(color="grey", width=1)
                )
            )
            fig3.add_annotation(
                go.layout.Annotation(
                    text=txt, x=i, y=max(end_table["frequency"]),
                    showarrow=True, arrowhead=2, arrowwidth=2,
                )
            )

        fig3.update_layout(
            title="Age Frequency Bar Graph",
            xaxis_title="Ages",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig3)

        end_table = end_table.sort_values(by='ages')
        end_table['cumulative_frequency'] = end_table['frequency'].cumsum()
        end_table['cdf'] = end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max() * 100
        end_table['cdf_1'] = 100 - (end_table['cumulative_frequency'] / end_table['cumulative_frequency'].max()) * 100

        for c in ["cdf", "cdf_1"]:
            if c == "cdf":
                l = [50, 75, 95, 99]
                name = f'Cumulative Distribution Function (CDF) of Ages ({sexe} - {current_age})'
                name2 = "CDF"
                verb = "to be deceased"
            else:
                l = [50, 25, 5, 1]
                name = f"Survival function ({sexe} - {current_age})"
                name2 = "CCDF"
                verb = "to be still alive"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=end_table['ages'], y=end_table[c], mode='lines', name=name2))
            fig.update_layout(title=name, xaxis_title='Age', yaxis_title=name2)

            for prob in l:
                age_at_prob = end_table.loc[(end_table[c] - prob).abs().idxmin()]['ages']
                exact_probability = round((end_table.loc[end_table['ages'] == age_at_prob, c].values[0]), 1)
                fig.add_vline(x=age_at_prob, line_dash="dash", line_color="red", annotation_text=f"{exact_probability}")
                st.write(f"{exact_probability}% probability {verb} at {age_at_prob} years")
            st.plotly_chart(fig)

        st.write(f"Average age at death of {num_simulations} individuals ({self.sexe}): {round(sum(self.deceased_ages)/len(self.deceased_ages),1)} - SD {round(np.std(self.deceased_ages),1)}")
        st.write(f"Median age at death: {round(statistics.median(self.deceased_ages),1)}")
        st.write(f"2.5% Percentile: {self.percentile_2_5:.2f} / 95% Percentile: {self.percentile_95:.2f} / 97.5% Percentile: {self.percentile_97_5:.2f}")
        st.write(f"Sum of persons {end_table['frequency'].sum()}")

    def plot_values_with_confidence_intervals(self, what):
        """Plot a graph with the values with the CI's"""
        st.subheader(what)
        values = np.array([result[what] for result in self.results])
        values = values.T

        mean = np.mean(values, axis=1)
        std = np.std(values, axis=1)
        confidence_interval = std * 1.96
        years = [self.current_age + i for i in range(len(mean))]
        max_years = self.max_age - self.current_age + 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years[:max_years], y=mean[:max_years], mode='lines', name='Mean', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=years[:max_years], y=(mean + confidence_interval)[:max_years], name='Upper CI',
                                 line=dict(color='dimgrey', width=.5)))
        fig.add_trace(go.Scatter(x=years[:max_years], y=(mean - confidence_interval)[:max_years], name='Lower CI',
                                 fill='tonexty', fillcolor='rgba(0, 128, 0, 0.2)',
                                 line=dict(color='dimgrey', width=.5)))

        fig.update_layout(
            xaxis_title='Years',
            yaxis_title=what,
            title=f'{what} Over Time with 95% Confidence Intervals',
            legend=dict(x=0, y=1),
            showlegend=True
        )
        fig.update_traces(mode='lines')
        fig.add_hline(y=0, line_color="black")
        fig.add_vline(x=self.retirement_age, line_dash="dash", line_color="green",
                      annotation_text="Retirement Age", annotation_position="top left")
        fig.add_vline(x=self.percentile_97_5, line_dash="dash", line_color="grey",
                      annotation_text="97.5%", annotation_position="top right")
        fig.add_vline(x=self.percentile_2_5, line_dash="dash", line_color="grey",
                      annotation_text="2.5%", annotation_position="top right")
        fig.add_vline(x=self.median_age_at_death, line_dash="dash", line_color="grey",
                      annotation_text="Median", annotation_position="top right")

        # FIX #5: y-axis range uses mean[:max_years] consistently
        fig.update_yaxes(range=[-30000, mean[:max_years].max() * 1.1])

        st.plotly_chart(fig)

        if self.print_individual:
            traces = []
            num_runs = values.shape[1]
            for run in range(num_runs):
                trace = go.Scatter(
                    x=np.arange(self.current_age, self.current_age + len(values[:max_years])),
                    y=values[:max_years, run],
                    mode='lines',
                    name=f'Run {run+1}'
                )
                traces.append(trace)

            layout = go.Layout(
                title=f'{what} - Line Plot for Each Column',
                xaxis=dict(title='Year'),
                yaxis=dict(title=what),
                showlegend=True
            )
            fig2 = go.Figure(data=traces, layout=layout)
            st.plotly_chart(fig2)

    def show_total_balance(self):
        """Show the total balance in time (profit or loss of the life insurance company)"""
        df_deceased = pd.DataFrame({'ages': self.deceased_ages})
        all_ages = pd.DataFrame({'ages': range(self.current_age, self.max_age + 1)})

        age_counts = df_deceased['ages'].value_counts().reset_index()
        age_counts.columns = ['ages', 'frequency']
        end_table = all_ages.merge(age_counts, on='ages', how='left').fillna(0)
        end_table["year"] = self.birthyear + end_table["ages"]

        values = np.array([result['balance_values'] for result in self.results])
        values = values.T

        # FIX #4: use self.max_age instead of hardcoded 100
        max_years = self.max_age - self.current_age + 1

        mean = np.mean(values, axis=1)
        end_table['mean'] = mean[:len(end_table)]  # guard against length mismatch
        end_table['per_year'] = end_table['mean'] * end_table['frequency']
        end_table['per_year_cumm'] = end_table['per_year'].cumsum()

        st.subheader("Profit/loss for the insurance company through the time (excl. costs)")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=end_table['year'], y=end_table['per_year_cumm'],
                                  mode='lines', name='Cumm Summ Balance', line=dict(color='blue')))
        fig5.add_hline(y=0, line_color="black")
        st.plotly_chart(fig5)


def main():
    calculator = PensionCalculator()
    calculator.interface()
    calculator.calculate_pension(num_simulations=calculator.num_simulations)
    calculator.plot_values_with_confidence_intervals("balance_values")
    calculator.show_ages_at_death(calculator.num_simulations, calculator.sexe, calculator.current_age)
    calculator.show_total_balance()

if __name__ == "__main__":
    main()