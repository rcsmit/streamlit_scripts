# https://opendata.cbs.nl/?dl=9CB37#/CBS/nl/dataset/85663NED/table


import pandas as pd
import streamlit as st
import re
import numpy as np
import plotly.express as px
import cbsodata

try:
    st.set_page_config(
        page_title="Lonen vs cpi",
        layout="wide"
    )
except:
    pass
def manipuleer_laatste_jaar(df):
    """ Verwijdert maand en maandnummer van de maand juli in het laatste jaar. 
        Als juli nog niet bereikt is, dan de laatste maand.

        Dit zodat de data voor het laatste jaar als jaargemiddelde wordt weergegeven.
        Args:
            df (pd.DataFrame): De dataframe met de gegevens.
        Returns:
            pd.DataFrame: De aangepaste dataframe met de maand en maandnummer van de laatste maand in het laatste jaar op NaN gezet.

    """ 
    
    jaar_num = pd.to_numeric(df["jaar"], errors="coerce")
    maandnr_num = pd.to_numeric(df["maandnr"], errors="coerce")

    
    
    jaar_num = pd.to_numeric(df["jaar"], errors="coerce")
    maandnr_num = pd.to_numeric(df["maandnr"], errors="coerce")

    # laatste jaar bepalen
    laatste_jaar = int(np.nanmax(jaar_num))

    # maanden in laatste jaar
    m_last = maandnr_num[jaar_num == laatste_jaar].dropna().astype(int)
    target_last = 7 if (m_last >= 7).any() else (int(m_last.max()) if not m_last.empty else None)
    
    mask = jaar_num.isin([laatste_jaar]) & (maandnr_num == target_last)
    df.loc[mask, ["maandnr", "maand"]] = [pd.NA, pd.NA]
   
    return df
def verleg_basisjaar(df, basisjaar, kolommen, groep):
    """ Verlegt het basisjaar van de opgegeven kolommen in de dataframe naar het opgegeven basisjaar.
        De waarden in de opgegeven kolommen worden omgerekend naar indexcijfers met het basisjaar als 100.
        
        Args:
            df (pd.DataFrame): De dataframe met de gegevens.
            basisjaar (int): Het jaar dat als basisjaar moet worden gebruikt.
            kolommen (list): Lijst van kolomnamen die moeten worden aangepast.
            groep (str): De kolomnaam die de groepen aangeeft waarop de basiswaarden moeten worden bepaald.
        
        Returns:
            pd.DataFrame: De aangepaste dataframe met de kolommen omgerekend naar indexcijfers.
    """
    
  
    basis = (
        df.loc[df["jaar"]==basisjaar, [groep]+kolommen]
          .groupby(groep, as_index=False).mean()
          .add_suffix("_basis")
    )
    basis = basis.rename(columns={f"{groep}_basis": groep})
    out = df.merge(basis, on=groep, how="left")
    # vectorized: deel alles in één keer
    out[kolommen] = out[kolommen].values / out[[c+"_basis" for c in kolommen]].values * 100.0
    out.drop(columns=[c+"_basis" for c in kolommen], inplace=True)

    return out

# functie om waarden te ontleden
def parse_period(p):
    """ Parse een periode string en retourneer jaar, kwartaal, maand en maandnummer.
        Args:
            p (str): De periode string in het formaat "YYYY", "YYYY kwartaal X" of "YYYY maand".
        
        Returns:
            pd.Series: Een pandas Series met vier elementen: jaar (int), kwartaal (int of None), maand (str of None), maandnummer (int of None).
    """ 

    
    # maanden mapping
    maanden = {
        "januari": 1, "februari": 2, "maart": 3,
        "april": 4, "mei": 5, "juni": 6,
        "juli": 7, "augustus": 8, "september": 9,
        "oktober": 10, "november": 11, "december": 12
    }

    parts = p.split()
    jaar = int(parts[0])
    maand = None
    maandnr = None
    kwartaal = None
    
    if len(parts) == 1:
        # enkel jaar
        kwartaal = None
    elif "kwartaal" in p:
        # extract kwartalnummer met regex
        match = re.search(r"(\d)e kwartaal", p.lower())
        if match:
            kwartaal = int(match.group(1))
    else:
        maand = parts[1]
        maandnr = maanden.get(maand.lower())
    
    return pd.Series([jaar, kwartaal, maand, maandnr])


@st.cache_data(ttl=86400)
def get_data_caolonen_cached(tabel: str) -> pd.DataFrame:
    print (f"getting data {tabel}")
    return pd.DataFrame(cbsodata.get_data(tabel))

@st.cache_data()
def get_data_caolonen(tabel):
    
    df = get_data_caolonen_cached(tabel)
    df[["jaar", "kwartaal", "maand", "maandnr"]] = df["Perioden"].apply(parse_period)
    df = manipuleer_laatste_jaar(df)
    df = df[(df["CaoSectoren"].str.strip().str.lower() == "totaal cao-sectoren")&
            (df["Versie"]=="Huidige cijfers")&
            (df["BedrijfstakkenBranchesSBI2008"]=="A-U Alle economische activiteiten") & 
            (df["kwartaal"].isnull()) &
            (df["maand"].isnull())]
    return df

# Downloaden van selectie van data
@st.cache_data()
def get_cao_lonen(basisjaar):
    """ Haalt de cao lonen data op en verlegt het basisjaar naar het opgegeven basisjaar.
        Args:   
            basisjaar (int): Het jaar dat als basisjaar moet worden gebruikt.
        Returns:    
            pd.DataFrame: De aangepaste dataframe met de kolommen omgerekend naar indexcijfers.
    """       
    kolommen = [
        "CaoLonenPerMaandExclBijzBeloningen_1",
        "CaoLonenPerMaandInclBijzBeloningen_2",
        "CaoLonenPerUurExclBijzBeloningen_3",
        "CaoLonenPerUurInclBijzBeloningen_4",
        "ContractueleLoonkostenPerMaand_5",
        "ContractueleLoonkostenPerUur_6",
        "ContractueleArbeidsduur_7",
        "CaoLonenPerMaandExclBijzBeloningen_8",
        "CaoLonenPerMaandInclBijzBeloningen_9",
        "CaoLonenPerUurExclBijzBeloningen_10",
        "CaoLonenPerUurInclBijzBeloningen_11",
        "ContractueleLoonkostenPerMaand_12",
        "ContractueleLoonkostenPerUur_13",
        "ContractueleArbeidsduur_14",
        "PercentageAfgeslotenCaoS_15"
    ]
    groep = "BedrijfstakkenBranchesSBI2008"
    df_1 = get_data_caolonen("82838NED") # 2010 = 100 
    df_1 = verleg_basisjaar(df_1, 2020, kolommen, groep)
    df_1=df_1[df_1["jaar"]<=2020]

    df_2 = get_data_caolonen("85663NED") # 2020 = 100 
    df_2=df_2[df_2["jaar"]> 2020]
    
    df_cao_lonen = pd.concat([df_1, df_2], ignore_index=True)
    
    df_cao_lonen= verleg_basisjaar(df_cao_lonen, basisjaar,kolommen, groep)
   
    return df_cao_lonen

@st.cache_data()
def get_cpi_from_cbs():
    print ("get_cpi_from_cbs")
    df = pd.DataFrame(cbsodata.get_data("83131NED"))
    return df

@st.cache_data()
def get_cpi(basisjaar):
    df = get_cpi_from_cbs()
    kolommen = ["CPI_1","CPIAfgeleid_2","MaandmutatieCPI_3", "MaandmutatieCPIAfgeleid_4","JaarmutatieCPI_5","JaarmutatieCPIAfgeleid_6"] 
    df[["jaar", "kwartaal", "maand", "maandnr"]] = df["Perioden"].apply(parse_period)
    df = manipuleer_laatste_jaar(df)
    df = df[(df["kwartaal"].isnull()) & (df["maand"].isnull())]
    df = df[df["Bestedingscategorieen"] == "000000 Alle bestedingen"]
    
    df = verleg_basisjaar(df, basisjaar, kolommen, "Bestedingscategorieen")
  
    return df
@st.cache_data()
def get_minimumloon(basisjaar):
    kolommen = ["loon_40",	"loon_38",	"loon_36"]
    sheet_id_minimumloon = "11bCLM4-lLZ56-XJjBjvXyXJ11P3PiNjV6Yl96x-tEnM"
    sheet_name_minimumloon = "data"

    url_minimumloon = f"https://docs.google.com/spreadsheets/d/{sheet_id_minimumloon}/gviz/tq?tqx=out:csv&sheet={sheet_name_minimumloon}"
    #url_minimumloon = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/minimumloon_dummy.csv"
    df_minimumloon = pd.read_csv(url_minimumloon, delimiter=",",  decimal="," )
    # omzetten naar datetime
    df_minimumloon["datum"] = pd.to_datetime(df_minimumloon["datum"], errors="coerce")

    # nieuwe kolommen
    df_minimumloon["jaar"] = df_minimumloon["datum"].dt.year
    df_minimumloon["maand"] = df_minimumloon["datum"].dt.month
    df_minimumloon["dag"] = df_minimumloon["datum"].dt.day

    # filter: alleen maand 7 (juli)
    df_minimumloon = df_minimumloon[df_minimumloon["maand"] == 7]
   
    for col in kolommen:
        df_minimumloon[col] = pd.to_numeric(df_minimumloon[col])

    df_minimumloon = verleg_basisjaar(df_minimumloon, basisjaar, kolommen, "dummy")

    return df_minimumloon

def make_plot(df_totaal,teller,noemer):
    """ Maakt een plot van de opgegeven dataframe met de opgegeven kolommen.
        Args:
            df_totaal (pd.DataFrame): De dataframe met de gegevens.
    """
    # lijst met kolommen die je wilt plotten
    kolommen = [
        "CaoLonenPerMaandExclBijzBeloningen_1",
        "CPI_1",
        "loon_40",
        "loon_38",
        "loon_36"
    ]

    # def make_plot(df_totaal):
    
    
    col1,col2=st.columns(2)
    with col1:
        dfp = df_totaal.sort_values("jaar")
        fig = px.line(dfp, x="jaar", y=kolommen, markers=True, template="plotly_white",
                    title="CAO-lonen, CPI en minimumloon")
        
        # legenda onder
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(b=100),
            xaxis_title="Jaar",
            yaxis_title="Index"
        )

        # horizontale lijn op 100
        fig.add_hline(y=100, line_dash="dot")
        fig.add_annotation(xref="paper", x=1, y=100, text="100", showarrow=False, xanchor="left", yanchor="bottom")

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.line(dfp, x="jaar", y="breuk", markers=True, template="plotly_white",
                    title=f"Verhouding {teller} / {noemer} * 100")
        
        
        # legenda onder
        fig.update_layout(
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(b=100),
            xaxis_title="Jaar",
            yaxis_title="Ratio"
        )

        # horizontale lijn op 100
        fig.add_hline(y=100, line_dash="dot")
        fig.add_annotation(xref="paper", x=1, y=100, text="100", showarrow=False, xanchor="left", yanchor="bottom")

        st.plotly_chart(fig, use_container_width=True)

  

def info():

    st.info("Cao-lonen, contractuele loonkosten en arbeidsduur; (2010=100), 1972-2023. Gewijzigd op: 2 november 2023 - https://opendata.cbs.nl/#/CBS/nl/dataset/82838NED/table?ts=1759502982850")
    st.info("Cao-lonen, contractuele loonkosten en arbeidsduur; indexcijfers (2020=100) Gewijzigd op: 2 oktober 2025 - https://opendata.cbs.nl/?dl=9CB37#/CBS/nl/dataset/85663NED/table")

    st.info("Consumentenprijzen; prijsindex 2015=100. Gewijzigd op: 1 oktober 2025 - https://opendata.cbs.nl/statline/?dl=3F0E#/CBS/nl/dataset/83131NED/table")
def main_():
    kol = ["CaoLonenPerMaandExclBijzBeloningen_1","CPI_1","loon_40","loon_38","loon_36"]
    col1,col2,col3,col4=st.columns(4)
    with col1:
        teller = st.selectbox("Teller", kol, index=0)
    with col2:
        noemer = st.selectbox("Noemer", kol, index=1)
    with col3:
        basisjaar=st.number_input("Basisjaar voor indexcijfers", min_value=2005, max_value=2025, value=2015, step=1)
    
    df_cao_lonen = get_cao_lonen(basisjaar)
    df_cpi = get_cpi(basisjaar)
    df_minimumloon = get_minimumloon(basisjaar)

    
    # merge stap voor stap
    df_merge = df_cao_lonen.merge(df_cpi, on="jaar", how="inner")
    df_totaal = df_merge.merge(df_minimumloon, on="jaar", how="outer")
    
    with col4:
        min,max = st.slider("Selecteer jaartal bereik", int(df_totaal["jaar"].min()), int(df_totaal["jaar"].max()), (2010, 2025), 1)
    df_totaal = df_totaal[(df_totaal["jaar"]>=min ) & (df_totaal["jaar"]<=max)]
    
    
    df_totaal["breuk"] = df_totaal[teller] / df_totaal[noemer] * 100
    make_plot(df_totaal,teller,noemer)
def main():
    tab1,tab2=st.tabs(["Plot","Informatie"])
    with tab1:
        main_()
    with tab2:
        info()

if __name__ == "__main__":
    main()



