import streamlit as st
import pandas as pd

# Streamlit Sidebar Inputs
st.sidebar.header("Lease Parameters")
lease_cost_yearly = st.sidebar.number_input("Annual Lease Cost (Baht)", value=1_000_000, step=100_000)
lease_increase_rate = st.sidebar.slider("Lease Increase Rate (%)", min_value=0.0, max_value=10.0, value=5.0) / 100
lease_period_years = st.sidebar.number_input("Lease Period (Years)", value=30, step=1)

st.sidebar.header("Tax Parameters")
tax_rate = st.sidebar.slider("Tax Rate (%)", min_value=0.0, max_value=10.0, value=1.1) / 100

st.sidebar.header("Income Parameters")
houses_income_monthly = st.sidebar.number_input("Monthly Income from Houses (Baht)", value=100_000, step=10_000)
houses_income_increase_rate = st.sidebar.slider("Houses Income Increase Rate (%)", min_value=0.0, max_value=10.0, value=3.0) / 100

villas_cost = st.sidebar.number_input("Construction Cost per Villa (Baht)", value=2_200_000, step=100_000) * 2
villas_income_monthly = st.sidebar.number_input("Monthly Income from Villas (Baht)", value=200_000, step=10_000)
villas_income_increase_rate = st.sidebar.slider("Villas Income Increase Rate (%)", min_value=0.0, max_value=10.0, value=3.0) / 100
st.sidebar.header("Rental costs")
rental_cost_percentage = st.sidebar.slider("Rental Costs (% of Total Income)", min_value=0.0, max_value=100.0, value=50.0) / 100

# Initialize variables for tracking
data = []

# Lease cost and payments
total_lease_cost = 0

for year in range(1, lease_period_years + 1):
    lease_payment = 0
    if (year - 1) % 3 == 0:  # Payment every 3 years
        lease_payment = lease_cost_yearly * 3 * (1 + lease_increase_rate) ** ((year - 1) // 3)
        total_lease_cost += lease_payment

    # Houses income
    annual_houses_income = houses_income_monthly * 12 * (1 + houses_income_increase_rate) ** (year - 1)

    # Villas income (starts after 2 years)
    if year > 2:
        annual_villas_income = villas_income_monthly * 12 * (1 + villas_income_increase_rate) ** (year - 3)
    else:
        annual_villas_income = 0

    # Total annual income
    total_annual_income = annual_houses_income + annual_villas_income

    rental_costs = total_annual_income * rental_cost_percentage
    net_income = total_annual_income - rental_costs
    # Append data for the year
    data.append({
        "Year": year,
        "Lease Payment": lease_payment,
        "Houses Income": annual_houses_income,
        "Villas Income": annual_villas_income,
        "Total Income": total_annual_income,
        "Rental Costs": rental_costs,
        "Net Income": net_income,
    })


# Tax on total lease value
total_tax = lease_cost_yearly * lease_period_years * tax_rate

# Construction costs for villas (added in year 1)
data[0]["Construction Costs"] = villas_cost
for i in range(1, len(data)):
    data[i]["Construction Costs"] = 0

# Create a dataframe
df = pd.DataFrame(data)

# Add cumulative columns
df["Cumulative Lease Cost"] = df["Lease Payment"].cumsum()
df["Cumulative Construction Costs"] = df["Construction Costs"].cumsum()
df["Cumulative Total Costs"] = df["Cumulative Lease Cost"] + df["Cumulative Construction Costs"] + total_tax
df["Cumulative Income"] = df["Total Income"].cumsum()

df["Cumulative Net Income"] = df["Net Income"].cumsum()
df["Cumulative profit"] = df["Cumulative Net Income"] - df["Cumulative Total Costs"]
# Calculate compound ROI
#df["Compound ROI (%)"] = ((df["Cumulative Net Income"] / df["Cumulative Total Costs"]) ** (1 / df["Year"]) - 1) * 100
df["Compound ROI (%)"] = (((df["Cumulative profit"]-villas_cost) / villas_cost) ** (1 / df["Year"]) - 1) * 100



# Calculate total costs and final ROI
total_income = df["Cumulative Income"].iloc[-1]
total_net_income = df["Cumulative Net Income"].iloc[-1]
total_profit = df["Cumulative profit"].iloc[-1]

total_costs = df["Cumulative Total Costs"].iloc[-1]
roi = (total_profit - villas_cost) / villas_cost * 100
roi_compound = (((total_profit-villas_cost) / villas_cost) ** (1 / 30) - 1) * 100

# Streamlit Output
st.title("30-Year Lease ROI Calculation")

st.subheader("Results")
st.write(f"**Total Lease Cost:** {total_lease_cost:,.2f} Baht")
st.write(f"**Total Tax:** {total_tax:,.2f} Baht")
st.write(f"**Total Income:** {total_income:,.2f} Baht")
st.write(f"**Total Net Income:** {total_net_income:,.2f} Baht")
st.write(f"**Total Costs:** {total_costs:,.2f} Baht")
st.write(f"**Investment:** {villas_cost:.0f} Baht")

st.write(f"**Final ROI:** {roi:.2f}%")
st.write(f"**Final compound ROI:** {roi_compound:.2f}%")


st.subheader("Yearly Breakdown")
#Convert all columns except the last one to integers
columns_to_int = df.columns[:-1]  # Exclude the last column
df[columns_to_int] = df[columns_to_int].astype(int)

# Ensure the last column remains a float
df["Compound ROI (%)"] = df["Compound ROI (%)"].round(2)
st.dataframe(df)
