import plotly.graph_objs as go
#import plotly.io as pio
import streamlit as st
import pandas as pd


def calculate_taxes(gemiddeld_inkomen_toptarief):
    """_summary_

    Args:
        gemiddeld_inkomen_toptarief (_type_): _description_

    Returns:
        _type_: _description_
    """    
    tax_brackets_plan_pvda_gl = {
        25000: 30.0,
        35000: 35.0,
        45000: 40.0,
        65000: 45.0,
        150000:49.50,
        float('inf'): 60.0
    }

            
    # Convert dictionary to DataFrame
    df = pd.DataFrame(tax_brackets_plan_pvda_gl.items(), columns=['Income', 'Tax_Rate'])

    if 'df' not in st.session_state:
    
        st.session_state.df = df
        st.session_state.key = 0

    df = st.session_state.df
    def reset():
        st.session_state.key += 1

    edited_df =st.data_editor(df, key=f'editor_{st.session_state.key}')
    st.write("Eerste bedrag is bovenkant van de schijf. Laatste regel is toptarief")                     
    st.button('Reset', on_click=reset)
     
    # Convert DataFrame back to dictionary
    tax_brackets_plan1 = edited_df.set_index('Income')['Tax_Rate'].to_dict()


    def calculate_tax_plan_pvda_gl(income):
        
        tax = 0
        prev_bracket = 0

        for bracket, rate in tax_brackets_plan1.items():
            if income <= bracket:
                tax += (income - prev_bracket) * rate/100
                break
            else:
                tax += (bracket - prev_bracket) * rate/100
                prev_bracket = bracket

        return tax

    def calculate_tax_plan_current(income):
        """_summary_

        Args:
            income (_type_): _description_

        Returns:
            _type_: _description_
        """        
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

    incomes.append(gemiddeld_inkomen_toptarief)
    taxes_plan1 = [calculate_tax_plan_current(income) for income in incomes]
    taxes_plan2 = [calculate_tax_plan_pvda_gl(income) for income in incomes]

    data = {
        'Income': incomes,
        'Huidig': taxes_plan1,
        'PvdA GL': taxes_plan2
    }

    df = pd.DataFrame(data)
    plot_graph(df)

    result = []
    for i in range(0, len(df), 2):
        if i+1 < len(df):
            total_huidig = df.iloc[i]['Huidig'] + df.iloc[i+1]['Huidig'] 
            total_pvda_gl =  df.iloc[i]['PvdA GL'] + df.iloc[i+1]['PvdA GL']
            result.append([df.iloc[i+1]['Income'], total_huidig, total_pvda_gl])

    # Creating a new DataFrame with summed values
    summed_df = pd.DataFrame(result, columns=['Income', 'Total_Huidig', 'Total_PvdAGL'])
  

    return df

def load_inkomen(gemiddeld_inkomen_toptarief):
    # https://www.cbs.nl/nl-nl/visualisaties/inkomensverdeling
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\cbs_gestandaardiseerd_inkomen_2021.csv"
    url="https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/cbs_gestandaardiseerd_inkomen_2021.csv"
    url="https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/cbs_besteedbaar_inkomen_2021.csv"
    
    df = pd.read_csv(url)
                     
    # Splitting the pattern column into two new columns
    #df[['Min_Value', 'Max_Value']] = df['gestandaardiseerd inkomen (x 1 000 euro)'].str.extract(r"tussen (\d+) en (\d+)")
    df[['Min_Value', 'Max_Value']] = df['besteedbaar inkomen (x 1 000 euro)'].str.extract(r"tussen (\d+) en (\d+)")
    df['Min_Value'] = pd.to_numeric(df['Min_Value'])*1000
    df['Income'] = pd.to_numeric(df['Max_Value'])*1000

    # Assuming 'Min_Value' and 'Max_Value' are the columns you want at the start
    df = df[['Min_Value', 'Max_Value'] + [col for col in df.columns if col not in ['Min_Value', 'Max_Value']]]

    df= df[["Income","Alle huishoudens"]]
    df.loc[len(df)] = [gemiddeld_inkomen_toptarief, 88]
    df = df.dropna(subset=['Income'])

    return df

def merge_dfs(df_inkomen, df_taxes):
    df = df_inkomen. merge(df_taxes, on="Income")
    df["opbrengensten_huidig"] = df["Alle huishoudens"] * df["Huidig"] * 1000
    df["opbrengensten_pvda_gl"] = df["Alle huishoudens"] * df["PvdA GL"] * 1000
    df["Verschil"] =  df["opbrengensten_pvda_gl"]  -df["opbrengensten_huidig"] 
    
    df["totaal_inkomen"] =df["Alle huishoudens"] * 1000* df["Income"]
    
    som_verschil = df["Verschil"].sum()
    som_inkomen = df["totaal_inkomen"].sum()    # zou 464mld moeten zijn 
                                                # https://longreads.cbs.nl/materiele-welvaart-in-nederland-2022/inkomen-van-huishoudens/
    st.info(f"Cummulatief verschil = {format(som_verschil, ',.0f')}")
    st.info(f"Cummulatief inkomen = {format(som_inkomen, ',.0f')}")
    st.write("x")
    st.write("Alle huishoudens = Aantal huishoudens x 1000")
    st.write("Huidig / PvdA GL = belasting opbrengst per huishouden")
    st.write("opbrengensten_huidig / opbrengensten_pvda_gl = totaal aantal opbrengsten per inkomensgroep (factor 1000 is meegerekend)")
    st.write("Aangenomen is dat de huishoudens boven de 100.000 euro gemiddeld 150.000 aan inkomsten hebben")
    st.write(df)
def main():
    st.subheader("Verschil huidig en plan PvdA GL")

    st.write("Reproductie en bestudering https://twitter.com/mdradvies/status/172444654377468769 ")
    gemiddeld_inkomen_toptarief = st.sidebar.number_input("Gemiddeld inkomen toptarief", 1,1_000_000,150_000)
    df_taxes = calculate_taxes(gemiddeld_inkomen_toptarief)
    df_inkomen = load_inkomen(gemiddeld_inkomen_toptarief)
    df_result = merge_dfs(df_inkomen, df_taxes)

    st.warning("Er wordt geen rekening gehouden met heffingskortingen. Inkomens boven 100.000 worden genegeerd")

if __name__ == "__main__":
    main()