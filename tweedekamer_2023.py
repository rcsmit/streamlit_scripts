import json
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from scipy.stats import chi2_contingency
import datetime

import plotly.graph_objects as go

try:
    st.set_page_config(layout="wide", page_title="Stemafwijkingen 2023")
except:
    pass

@st.cache_data(show_spinner=False)
def load_votes(jaar):
    if jaar ==2023:
        return load_votes_2023()
    elif jaar ==2025:
        return load_votes_2025()
    else:
        st.error("Fout in jaar")
        st.stop()

@st.cache_data(show_spinner=False)
def load_votes_2025():
    # C:\Users\rcxsm\Documents\python_scripts\python_scripts_rcsmit\fetch_combine_anp_tk2025.py

    # url_results=r"C:\Users\rcxsm\Documents\python_scripts\alle_resultaten_per_gemeente.csv"
    # url_partynames = r"C:\Users\rcxsm\Downloads\party_names.csv"

    url_results= "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/alle_resultaten_per_gemeente.csv"
    url_partynames =  "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/partij_keys.csv"

    df_results = pd.read_csv(url_results, dtype={"cbs_code":str})
    df_partynames = pd.read_csv(url_partynames)
    df_partynames =df_partynames[["party_key","LijstNaam"]]

    df_results_new=df_results.merge(df_partynames, on="party_key", how="left")
    df_results_new=df_results_new.fillna("UNKNOWN_X")
  
    df_results_new=df_results_new[["Regio","Waarde", "LijstNaam"]]
    df_results_new=df_results_new[df_results_new["Regio"] !="Venray"]  # Venray moet nog worden geteld
    print (df_results_new)
    return df_results_new



@st.cache_data(show_spinner=False)
def load_votes_2023():
    """Load the votes of 2023"""

    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/TK2023_uitslag_gemeente.csv"
    df = pd.read_csv(url, sep=";")
    df = df[df["LijstNaam"].notna()]  # ipv != None
    df = df[df["RegioCode"].str.startswith("G", na=False)]
    df = df[df["VeldType"] == "LijstAantalStemmen"]
    return df


@st.cache_data(show_spinner=False)
def load_geojson():
    """Load the file with the shapees of the municipalities

    Returns:
        _type_: json file
    """
    url = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gemeente_2023.geojson"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def show_info():
    """Show information about the app"""
    st.title(
        "Welke gemeenten stemmen het meest afwijkend van Nederland? Tweede Kamer 2023"
    )
    st.info(
        "Reproductie van https://www.rtl.nl/nieuws/politiek/artikel/5535930/apeldoorn-nederland-het-klein-zo-gemiddeld-stemt-jouw-gemeente"
    )
    st.info(
        "Data  https://data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2023?utm_source=chatgpt.com#panel-description"
    )
    st.info("Gemeentegrenzen  https://cartomap.github.io/nl/")

    st.info(
        """FORMULES
            
chi2_prop = SUM { (p_gemeente - p_landelijk")**2 / p_landelijk}

chi2_rtl = SUM {   abs ( p_gemeente- p_landelijk) }
            
Gebruik **chi2_prop** als je ook **p-waardes** wilt en een maat die rekening houdt met de **verwachte verdeling**. Gebruik **chi2_rtl** als je een **eenvoudige afstand** wilt voor een **stabiele ranglijst** zonder statistische aannames.

**Waarom**

* **chi2_prop**

  * Weegt verschillen met **1 / p_landelijk**
  * Accentueert afwijkingen bij **kleine partijen**
  * Koppelt direct aan **χ²-toets** voor **p-waardes**
  * Kan groot worden als **p_landelijk ~ 0**. Groepeer mini-partijen of zet een ondergrens.

* **chi2_rtl = Σ|p_gemeente − p_landelijk|**

  * Eenvoudig en **schaalvrij**
  * **Robuust** tegen extreem kleine p’s
  * Geeft **geen p-waarde**
  * Is 2 × **total variation distance**, dus goed als pure afstand.

**Aanbeveling**

* Voor jouw “wijkt af van Nederland” met significatie: **chi2_prop**
* Voor een rustige kaart en ranglijst zonder test: **chi2_rtl**

**TL;DR**
Wil je **significantie** en nadruk op **relatieve afwijkingen**: **chi2_prop**.
Wil je een **neutrale afstand** zonder p-waardes: **chi2_rtl**.
    """
    )

    st.info(
        "Het bestand bevat gemeente NBSB;G9010;K12;P28;, wat niet voorkomt in de RTL data en ook niet op het internet"
    )

    # st.markdown(f"""
    # **Chi-kwadraattoets**

    # - χ² = {chi2:.3f} / {chi2_prop:.3f} (proporties)
    # - Vrijheidsgraden = {dof}
    # - p-waarde = {p:.4f}
    # """)

    # if p < 0.05:
    #     st.write(f"➡️ {uitgelichte_gemeente} stemt significant anders dan Nederland.")
    # else:
    #     st.write (f"✅ {uitgelichte_gemeente} stemt ongeveer zoals Nederland.")

    # De chi-kwadraattoets vergelijkt absolute aantallen tussen twee verdelingen.
    # Omdat Nederland véél meer stemmen heeft dan een gemeente, worden de verschillen enorm uitvergroot.
    # Zelfs kleine afwijkingen worden dan “statistisch significant”, waardoor p ≈ 0.

    # Kort gezegd:

    # De toets denkt dat elk verschil belangrijk is, omdat de steekproef extreem groot is.

    # 2. Wat we eigenlijk willen meten

    # Je wilt niet weten of uitgelicht ≠ Nederland op basis van miljoenen stemmen,
    # maar of de verdeling van stemmen per partij in uitgelicht proportioneel afwijkt van de landelijke verdeling.

    # Dat vraagt om een vergelijking van percentages (proporties), niet absolute aantallen.



def calculate_results_gemeente(df,jaar, kol_regio,kol_partij,kol_stemmen):
    """Bereken de resultaten van een bepaalde gemeente

    Args:
        df (_type_): _description_
    """    

    gemeentes = sorted(df[kol_regio].unique().tolist())
    index_leiden = gemeentes.index("Apeldoorn")  # geeft positie van 'Leiden'
    uitgelichte_gemeente = st.selectbox("Gemeente", gemeentes, index=index_leiden)

    u_df = df[df[kol_regio] == uitgelichte_gemeente]
    st.write(
        f"Totaal aantal geldige stemmen in {uitgelichte_gemeente} = {u_df['Waarde'].sum()}"
    )

    # Som stemmen per partij per regio
    agg = df.groupby([kol_regio, kol_partij], as_index=False)[kol_stemmen].sum()

    # Selecteer uitgelicht
    apel = agg.query(f"{kol_regio} == '{uitgelichte_gemeente}'")[
        [kol_partij, kol_stemmen]
    ].rename(columns={kol_stemmen: uitgelichte_gemeente})

    # Landelijk totaal
    landelijk = (
        agg.groupby(kol_partij, as_index=False)[kol_stemmen]
        .sum()
        .rename(columns={kol_stemmen: "Nederland"})
    )

    # Merge
    m = pd.merge(landelijk, apel, on=kol_partij, how="inner").set_index(kol_partij)

    # Bereken percentages
    m["% Nederland"] = 100 * m["Nederland"] / m["Nederland"].sum()
    m[f"% {uitgelichte_gemeente}"] = (
        100 * m[uitgelichte_gemeente] / m[uitgelichte_gemeente].sum()
    )
    m["Verschil (pp)"] = m[f"% {uitgelichte_gemeente}"] - m["% Nederland"]
    m["Verschil (%))"] = (
        (m[f"% {uitgelichte_gemeente}"] - m["% Nederland"]) / m["% Nederland"] * 100
    )

    # Chi-kwadraattoets
    # obs = m[[uitgelichte_gemeente, "Nederland"]].T.values
    # chi2, p, dof, expected = chi2_contingency(obs)

    # Chi2 op basis van proporties
    chi2_prop = (
        (m[f"% {uitgelichte_gemeente}"] - m["% Nederland"]) ** 2 / m["% Nederland"]
    ).sum()
    chi2_rtl = (abs(m[f"% {uitgelichte_gemeente}"] - m["% Nederland"])).sum()

    # Resultaat tonen
    st.subheader(f"{uitgelichte_gemeente} vs Nederland – Tweede Kamer {jaar}")
    st.dataframe(m.sort_values("% Nederland", ascending=False).round(2))

    st.markdown(
        f"""
    **Chi-kwadraattoets**

    - χ² = {chi2_prop:.3f} (proporties)
    - RTL = {chi2_rtl:.3f} (RTL-methode)
    
    """
    )

@st.cache_data(show_spinner=False)
def calculate_results_landelijk(jaar, df, kol_regio,kol_partij,kol_stemmen):
    """Bereken de landelijke afwijkingen van het gemiddelde stemgedrag.
    Bereken tevens de chi2-waardes (verschillende methodes), p-waarde, ranking en percentiel

    Args:
        df (_type_): _description_

    Returns:
        df: df met de afwijkingen, chi2-waardes (verschillende methodes), p-waarde, ranking en percentiel
    """    

    agg = df.groupby([kol_regio, kol_partij], as_index=False)[kol_stemmen].sum()
    landelijk = agg.groupby(kol_partij, as_index=False)[kol_stemmen].sum()
    landelijk.columns = [kol_partij, "Nederland"]

    # Landelijke verdeling in fracties
    landelijk["p_landelijk"] = (
        100 * landelijk["Nederland"] / landelijk["Nederland"].sum()
    )

    resultaten = []
    gemeenten = agg[kol_regio].unique()

    for g in gemeenten:
        lokaal = agg.query(f"{kol_regio} == @g")[[kol_partij, kol_stemmen]].rename(
            columns={kol_stemmen: g}
        )
        m = pd.merge(landelijk, lokaal, on=kol_partij, how="inner").fillna(0)
        m["p_gemeente"] = 100 * m[g] / m[g].sum()

        # Chi-kwadraattoets
        obs = m[[g, "Nederland"]].T.values
        try:
            chi2_test, p, dof, expected = chi2_contingency(obs)
        except:
            chi2_test, p, dof, expected = None,None,None,None
        # Chi2 op basis van proporties
        chi2_prop = ((m["p_gemeente"] - m["p_landelijk"]) ** 2 / m["p_landelijk"]).sum()
        chi2_rtl = (abs(m["p_gemeente"] - m["p_landelijk"])).sum()

        # resultaten.append({"Gemeente": g, "Chi2": chi2, "Chi2_prop": chi2_prop})
        resultaten.append(
            {
                "Gemeente": g,
                f"Chi2_rtl_{jaar}": chi2_rtl,
                f"Chi2_prop_{jaar}": chi2_prop,
                f"Chi2_test_{jaar}": chi2_test,
                f"p_{jaar}": p,
            }
        )

    df_res = pd.DataFrame(resultaten).sort_values(f"Chi2_rtl_{jaar}", ascending=True)
    for fieldname in [f"Chi2_rtl_{jaar}", f"Chi2_prop_{jaar}", f"Chi2_test_{jaar}"]:
        # Rangorde op basis van Chi2_prop, hoogste eerst
        try:
            df_res[f"Rank_{fieldname}"] = (
                df_res[fieldname].rank(method="dense", ascending=True).astype(int)
            )
            df_res[f"Percentiel_{fieldname}"] = (
                df_res[fieldname].rank(pct=True, ascending=True) * 100
            )
        except:
            df_res[f"Rank_{fieldname}"] = None
            
            df_res[f"Percentiel_{fieldname}"] = None
            
    return df_res


def make_plot(df_res, jaar, metric):
    """_summary_

    Args:
        df_res (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 1) Laad GeoJSON
    gjson = load_geojson()
    if jaar==2025:
        fix = {"Hengelo (O)":"Hengelo",
            "Den Bosch": "'s-Hertogenbosch",
            "Bergen (L)": "Bergen (L.)",  # voorbeeld
            "Bergen (NH)": "Bergen (NH.)",  # voorbeeld
        }
    elif jaar==2023:
        fix = {"Hengelo (O)":"Hengelo",
         "Bergen (L)": "Bergen (L.)",  # voorbeeld
            "Bergen (NH)": "Bergen (NH.)",  # voorbeeld
        }
    else:
        st.error("Fout in jaar")
        st.stop()
   
    df_res["Gemeente_fix"] = df_res["Gemeente"].replace(fix)

    

    # 3) Range en colormap
    vmin = float(df_res[metric].min())
    vmax = float(df_res[metric].max())
    cmap = cm.LinearColormap(
        #colors=["#e8f3ec", "#a9d0c3", "#6fb0a1", "#3e7e80", "#214b5a"],
        colors=["#ffff00", "#ff0000"],
        vmin=vmin,
        vmax=vmax,
    ).to_step(7)
    cmap.caption = (
        "gemiddeld            zeer afwijkend"
        if metric == "Chi2_prop"
        else "laag             hoog"
    )

    # 4) Map DataFrame-data naar GeoJSON
    # Voeg jouw df_res-data toe aan de GeoJSON via 'statnaam'
    data_dict = df_res.set_index("Gemeente_fix").to_dict(orient="index")

    for feature in gjson["features"]:
        name = feature["properties"].get("statnaam")
        if name in data_dict:
            feature["properties"].update(data_dict[name])

    # 5) Kaart en stijl
    def style_function(feature):
        val = feature["properties"].get(metric)
        color = "#cccccc" if val is None else cmap(val)
        return {
            "fillColor": color,
            "color": "#ffffff",
            "weight": 0.6,
            "fillOpacity": 0.85,
        }

    # 6) Tooltip met ALLE velden uit df_res + statnaam
    tooltip_fields = ["statnaam"] + [c for c in df_res.columns if c != "Gemeente_fix"]
    tooltip_aliases = tooltip_fields

    m = folium.Map(
        location=[52.2, 5.3], zoom_start=7, tiles="cartodbpositron"
    )

    gj = folium.GeoJson(
        gjson,
        name="Gemeenten",
        style_function=style_function,
        highlight_function=lambda f: {"weight": 2, "color": "#222222"},
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            localize=True,
            sticky=True,
            labels=True,
        ),
    )
    gj.add_to(m)
    cmap.add_to(m)

    st.markdown(f"## Kaart: {metric}")
    st_folium(m, returned_objects=[])
    
def plot_scatter(df_res_all,xaxis,yaxis):
   
    # Voorbeeld dataframe
    # df_res_all bevat o.a. kolommen: Gemeente, Rank_Chi2_rtl_2023, Rank_Chi2_rtl_2025
    # df_res_all = pd.DataFrame(...)
    show_text = st.checkbox("Toon tekstlabels", value=True)
    if show_text:
        mode_="markers+text"
    else:
        mode_="markers"

    fig = go.Figure(
        data=[
            go.Scatter(
                x=df_res_all[xaxis],
                y=df_res_all[yaxis],
                mode=mode_,
                #mode="markers",
                text=df_res_all["Gemeente"],           # mouseover
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Rank 2023: %{x}<br>"
                    "Rank 2025: %{y}<extra></extra>"
                ),
                marker=dict(size=8, color="teal", opacity=0.7, line=dict(width=0.5, color="white"))
            )
        ]
    )

        
    # Referentielijn x=y
    min_val = min(df_res_all["Rank_Chi2_rtl_2023"].min(), df_res_all["Rank_Chi2_rtl_2025"].min())
    max_val = max(df_res_all["Rank_Chi2_rtl_2023"].max(), df_res_all["Rank_Chi2_rtl_2025"].max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="gray", dash="dash"),
            name="x = y",
            hoverinfo="skip",
        )
    )

        
    # Tekstlabels
    fig.add_annotation(
        text="Is meer afwijkend dan vorig jaar",
        xref="paper", yref="paper",
        x=0.02, y=0.98, showarrow=False,
        font=dict(size=24, color="gray")
    )
    fig.add_annotation(
        text="Is minder afwijkend dan vorig jaar",
        xref="paper", yref="paper",
        x=0.98, y=0.10, showarrow=False,
        font=dict(size=24, color="gray"),
        xanchor="right", yanchor="bottom"
    )
    fig.update_layout(
        title="Vergelijking Rank_Chi2_rtl 2023 vs 2025",
        xaxis_title="Rank_Chi2_rtl_2023",
        yaxis_title="Rank_Chi2_rtl_2025",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, width=True)
    st.write(df_res_all)

def main():
    """Main functie
    """  

    # Pas deze kolomnamen aan als ze anders zijn
    kol_regio = "Regio"
    kol_partij = "LijstNaam"
    kol_stemmen = "Waarde"  
    tab1, tab2, tab3,tab4 = st.tabs(["Resultaten", "Enkele gemeente","Partij", "Info"])

    with tab4:
        show_info()
    with tab1:
        df_res_all = pd.DataFrame()
        jaren = [2023, 2025]
        columns_metrics = [
            f"Chi2_prop",
            f"Chi2_rtl_",
            f"Rank_Chi2_rtl",
            f"Percentiel_Chi2_rtl",
            f"Rank_Chi2_prop",
            f"Percentiel_Chi2_prop",
        ]

        # 2) Keuze metriek
        metric = st.radio("Kleur op", columns_metrics, 2, horizontal=True)
        for jaar in  jaren:
            df = load_votes(jaar)
            df_res = calculate_results_landelijk(jaar, df, kol_regio,kol_partij,kol_stemmen)
            if jaar ==jaren[-1]:
                metric_=f"{metric}_{jaar}"
                make_plot(df_res, jaar, metric_)
                st.write(f"Aantal gemeentes in {jaar} : {len(df_res)}")
                st.dataframe(df_res.style.format({"Chi2": "{:.4f}"}))
            if df_res_all.empty:
                df_res_all = df_res
            else:
                df_res_all = df_res_all.merge(df_res, on="Gemeente", how="inner")
        st.markdown("## Vergelijking 2023 vs 2025")
        #st.dataframe(df_res_all.style.format({"Chi2": "{:.4f}"}))
        plot_scatter(df_res_all,xaxis=f"Rank_Chi2_rtl_2023", yaxis=f"Rank_Chi2_rtl_2025")

        #plot_scatter(df_res_all,xaxis=f"Chi2_rtl_2023", yaxis=f"Chi2_rtl_2025")
    with tab2:
        jaar = st.radio("Jaar", jaren, index=1, horizontal=True, key="gemeente_jaar")
        df_j = load_votes(jaar)
        calculate_results_gemeente(df_j, jaar, kol_regio,kol_partij,kol_stemmen)
    with tab3:
        jaar = st.radio("Jaar", jaren, index=1, horizontal=True, key="partij_jaar")
        df_j = load_votes(jaar)
        
     
        den = df_j.groupby(kol_regio)[kol_stemmen].transform("sum")
        df_j["percentage"] = (100 * df_j[kol_stemmen] / den).fillna(0).round(2)
    
        #check = df.groupby(kol_regio)["percentage"].sum().round(2)
        df_j["Gemeente"]=df_j["Regio"]
        
        partij = st.selectbox("Partij", sorted(df_j[kol_partij].unique().tolist()), index=0)
        df_p=df_j[df_j["LijstNaam"] == partij]
       
        if len(df_p)>0:
    
            df_p=df_p[["Gemeente","Waarde","percentage"]].sort_values("percentage", ascending=False)
            df_p["Zetels"] = round(df_p["percentage"]/0.66667,1)
            #df_p[f"Percentage_{partij}"] = df_p["percentage"]
            
            df_p = df_p.rename(columns={"percentage": f"Percentage_{partij}"})
            
          
            try:
                make_plot(df_p, jaar, f"Percentage_{partij}")
            except:
                #error VRIJVER 2025
                st.error("Fout bij het maken van de kaart")
            st.write(df_p)
        else:
            st.error("Partij heeft geen stemmen")
            st.stop()
if __name__ == "__main__":
    main()
