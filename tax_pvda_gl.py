import plotly.graph_objs as go
#import plotly.io as pio
import streamlit as st
import pandas as pd

def calculate_taxes():
    def calculate_tax_plan_pvda_gl(income):
        tax_brackets_plan1 = {
            25000: 0.3,
            35000: 0.35,
            45000: 0.40,
            65000: 0.45,
            150000:0.4950,
            float('inf'): 0.6
        }

        tax = 0
        prev_bracket = 0

        for bracket, rate in tax_brackets_plan1.items():
            if income <= bracket:
                tax += (income - prev_bracket) * rate
                break
            else:
                tax += (bracket - prev_bracket) * rate
                prev_bracket = bracket

        return tax

    def calculate_tax_plan_current(income):
        tax_brackets_plan2 = {
            76000: 0.369,
            float('inf'): 0.4950
        }

        tax = 0
        prev_bracket = 0

        for bracket, rate in tax_brackets_plan2.items():
            if income <= bracket:
                tax += (income - prev_bracket) * rate
                break
            else:
                tax += (bracket - prev_bracket) * rate
                prev_bracket = bracket

        return tax
    def plot_graph(df):

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Income"], y=df["Huidig"], mode='lines', name='Huidig'))
        fig.add_trace(go.Scatter(x=df["Income"], y=df["PvdA GL"], mode='lines', name='PvdA GL'))

        fig.update_layout(title='Tax Comparison between Plans',
                        xaxis_title='Income',
                        yaxis_title='Tax Amount')
        st.plotly_chart(fig)

    # Generate comparison graph
    incomes = list(range(1000, 100000, 1000))

    incomes.append(150000)
    taxes_plan1 = [calculate_tax_plan_current(income) for income in incomes]
    taxes_plan2 = [calculate_tax_plan_pvda_gl(income) for income in incomes]

    data = {
        'Income': incomes,
        'Huidig': taxes_plan1,
        'PvdA GL': taxes_plan2
    }

    df = pd.DataFrame(data)
    plot_graph(df)
    st.write(df)
    result = []
    for i in range(0, len(df), 2):
        if i+1 < len(df):
            total_huidig = df.iloc[i]['Huidig'] + df.iloc[i+1]['Huidig'] 
            total_pvda_gl =  df.iloc[i]['PvdA GL'] + df.iloc[i+1]['PvdA GL']
            result.append([df.iloc[i+1]['Income'], total_huidig, total_pvda_gl])

    # Creating a new DataFrame with summed values
    summed_df = pd.DataFrame(result, columns=['Income', 'Total_Huidig', 'Total_PvdAGL'])
  

    return df

def load_inkomen():
    url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\cbs_gestandaardiseerd_inkomen_2021.csv"
    df = pd.read_csv(url)
                     
    # Splitting the pattern column into two new columns
    df[['Min_Value', 'Max_Value']] = df['gestandaardiseerd inkomen (x 1 000 euro)'].str.extract(r"tussen (\d+) en (\d+)")
    df['Min_Value'] = pd.to_numeric(df['Min_Value'])*1000
    df['Income'] = pd.to_numeric(df['Max_Value'])*1000

    # Assuming 'Min_Value' and 'Max_Value' are the columns you want at the start
    df = df[['Min_Value', 'Max_Value'] + [col for col in df.columns if col not in ['Min_Value', 'Max_Value']]]

    df= df[["Income","Alle huishoudens"]]
    df.loc[len(df)] = [150000, 88]
    df = df.dropna(subset=['Income'])
 
    
    
    print (df)
    return df

def merge_dfs(df_inkomen, df_taxes):
    df = df_inkomen. merge(df_taxes, on="Income")
    df["opbrengensten_huidig"] = df["Alle huishoudens"] * df["Huidig"] * 1000
    df["opbrengensten_pvda_gl"] = df["Alle huishoudens"] * df["PvdA GL"] * 1000
    df["Verschil"] =  df["opbrengensten_pvda_gl"]  -df["opbrengensten_huidig"] 
    
    st.write(df)
    som = df["Verschil"].sum()
    st.info(f"Cummulatief verschil = {format(som, ',.0f')}")
    st.write("Alle huishoudens = Aantal huishoudens x 1000")
    st.write("Huidig / PvdA GL = belasting opbrengst per huishouden")
    st.write("opbrengensten_huidig / opbrengensten_pvda_gl = totaal aantal opbrengsten per inkomensgroep (factor 1000 is meegerekend)")
    st.write("Aangenomen is dat de huishoudens boven de 100.000 euro gemiddeld 150.000 aan inkomsten hebben")

def main():
    st.subheader("Verschil huidig en plan PvdA GL")

    st.write("Reproductie en bestudering https://twitter.com/mdradvies/status/172444654377468769 ")
    df_taxes = calculate_taxes()
    df_inkomen = load_inkomen()
    df_result = merge_dfs(df_inkomen, df_taxes)

    st.warning("Er wordt geen rekening gehouden met heffingskortingen. Inkomens boven 100.000 worden genegeerd")

if __name__ == "__main__":
    main()