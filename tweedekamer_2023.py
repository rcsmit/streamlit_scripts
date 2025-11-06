import json
import requests
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import branca.colormap as cm
from scipy.stats import chi2_contingency
import datetime
import numpy as np
import random
import plotly.graph_objects as go
import statsmodels.api as sm


from statsmodels.formula.api import glm
#from statsmodels.genmod.families import Beta

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

    url_results= "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/alle_resultaten_per_gemeente_2025.csv"
    url_partynames =  "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/partij_keys.csv"

    df_results = pd.read_csv(url_results, dtype={"cbs_code":str})
    df_partynames = pd.read_csv(url_partynames)
    df_partynames =df_partynames[["party_key","LijstNaam"]]

    df_results_new=df_results.merge(df_partynames, on="party_key", how="left")
    df_results_new=df_results_new.fillna("UNKNOWN_X")
    
    df_results_new=df_results_new[["Regio","Waarde", "LijstNaam","voters_current"]]
    #df_results_new=df_results_new[df_results_new["Regio"] !="Venray"]  # Venray moet nog worden geteld
    den = df_results_new.groupby("Regio")["Waarde"].transform("sum")
    df_results_new["percentage_votes"] = (100 * df_results_new["Waarde"] / den).fillna(0).round(2)
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
    den = df.groupby("Regio")["Waarde"].transform("sum")
    df["percentage_votes"] = (100 * df["Waarde"] / den).fillna(0).round(2)
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
    st.info("Gemeentegrenzen : https://cartomap.github.io/nl/")
    st.info("Obesitas:  https://www.vzinfo.nl/overgewicht/regionaal/obesitas#obesitas-volwassenen")
    st.info("Inkomen (2020) : https://www.cbs.nl/nl-nl/maatwerk/2023/35/inkomen-per-gemeente-en-wijk-2020")
    st.info("Opleiding [%>= HBO/WO] (2024) : https://www.clo.nl/indicatoren/nl210016-hbo-en-wo-gediplomeerden-2024")
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



def calculate_results_gemeente(df,jaar):
    """Bereken de resultaten van een bepaalde gemeente

    Args:
        df (_type_): _description_
    """    
    gemeentes = sorted(df["Regio"].unique().tolist())
    index_leiden = gemeentes.index("Apeldoorn")  # geeft positie van 'Leiden'
    uitgelichte_gemeente = st.selectbox("Gemeente", gemeentes, index=index_leiden)

    u_df = df[df["Regio"] == uitgelichte_gemeente]
    st.write(
        f"Totaal aantal geldige stemmen in {uitgelichte_gemeente} = {u_df['Waarde'].sum()}"
    )

    # Som stemmen per partij per regio
    agg = df.groupby(["Regio", "LijstNaam"], as_index=False)["Waarde"].sum()
   
    # Selecteer uitgelicht
    apel = agg.query("Regio == @uitgelichte_gemeente")[
            ["LijstNaam", "Waarde"]
        ].rename(columns={"Waarde": uitgelichte_gemeente})
    # Landelijk totaal
    landelijk = (
        agg.groupby("LijstNaam", as_index=False)["Waarde"]
        .sum()
        .rename(columns={"Waarde": "Nederland"})
    )

    # Merge
    m = pd.merge(landelijk, apel, on="LijstNaam", how="inner").set_index("LijstNaam")

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
def calculate_results_landelijk(jaar, df):
    """Bereken de landelijke afwijkingen van het gemiddelde stemgedrag.
    Bereken tevens de chi2-waardes (verschillende methodes), p-waarde, ranking en percentiel

    Args:
        df (_type_): _description_

    Returns:
        df: df met de afwijkingen, chi2-waardes (verschillende methodes), p-waarde, ranking en percentiel
    """    

    agg = df.groupby(["Regio", "LijstNaam"], as_index=False)["Waarde"].sum()
    landelijk = agg.groupby("LijstNaam", as_index=False)["Waarde"].sum()
    landelijk.columns = ["LijstNaam", "Nederland"]

    # Landelijke verdeling in fracties
    landelijk["p_landelijk"] = (
        100 * landelijk["Nederland"] / landelijk["Nederland"].sum()
    )

    resultaten = []
    gemeenten = agg["Regio"].unique()

    for g in gemeenten:
        lokaal = agg.query("Regio == @g")[["LijstNaam", "Waarde"]].rename(columns={"Waarde": g})

        m = pd.merge(landelijk, lokaal, on="LijstNaam", how="inner").fillna(0)
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

def make_map(df_res, jaar, metric):
    gjson = load_geojson()

    # naamfix per jaar
    if jaar == 2025:
        fix = {"Hengelo (O)": "Hengelo", "Den Bosch": "'s-Hertogenbosch",
            "Bergen (L)": "Bergen (L.)", "Bergen (NH)": "Bergen (NH.)"}
    elif jaar == 2023:
        fix = {"Hengelo (O)": "Hengelo", "Bergen (L)": "Bergen (L.)",
            "Bergen (NH)": "Bergen (NH.)"}
    else:
        st.error("Fout in jaar"); st.stop()

    df_res["Gemeente_fix"] = df_res["Gemeente"].replace(fix)

    # Data → GeoJSON properties
    data_dict = df_res.set_index("Gemeente_fix").to_dict(orient="index")
    for feature in gjson["features"]:
        name = feature["properties"].get("statnaam")
        if name in data_dict:
            feature["properties"].update(data_dict[name])

    # -------- kleurconfig --------
    use_categorical = (metric == "populairste_coalitie")

    if use_categorical:
        # 1 = rood, 2 = paars, 3 = rood (zoals gevraagd)
        CAT_COLORS = {1: "#ff0000", 2: "#800080", 3: "#0000ff", 0: "#cccccc", None: "#eeeeee"}
        def style_function(feature):
            val = feature["properties"].get("populairste_coalitie")
            color = CAT_COLORS.get(val, "#cccccc")
            return {"fillColor": color, "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.85}
    else:
        # glijdende schaal voor continue metrics
        vmin = float(df_res[metric].min()); vmax = float(df_res[metric].max())
        cmap = cm.LinearColormap(colors=["#ffff00", "#ff0000"], vmin=vmin, vmax=vmax).to_step(7)
        cmap.caption = "gemiddeld            zeer afwijkend" if metric == "Chi2_prop" else "laag             hoog"
        def style_function(feature):
            val = feature["properties"].get(metric)
            color = "#cccccc" if val is None else cmap(val)
            return {"fillColor": color, "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.85}

    # -------- map + tooltip --------
    tooltip_fields = ["statnaam"] + [c for c in df_res.columns if c != "Gemeente_fix"]
    m = folium.Map(location=[52.2, 5.3], zoom_start=7, tiles="cartodbpositron")

    gj = folium.GeoJson(
        gjson,
        name="Gemeenten",
        style_function=style_function,
        highlight_function=lambda f: {"weight": 2, "color": "#222222"},
        tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_fields,
                                    localize=True, sticky=True, labels=True),
    )
    gj.add_to(m)

    # legenda
    if use_categorical:
        legend_html = """
        <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999;
                    background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 14px;">
        <b>populairste_coalitie</b><br>
        <span style="background:#ff0000;display:inline-block;width:12px;height:12px;margin-right:6px;"></span>1 (links)<br>
        <span style="background:#800080;display:inline-block;width:12px;height:12px;margin-right:6px;"></span>2 (midden)<br>
        <span style="background:#ff0000;display:inline-block;width:12px;height:12px;margin-right:6px;"></span>3 (rechts)
        </div>
        """
        folium.map.Marker([0,0], icon=folium.DivIcon(html=legend_html)).add_to(m)
    else:
        cmap.add_to(m)

    st.markdown(f"## Kaart: {metric}")
    st_folium(m, returned_objects=[])

def make_map_oud(df_res, jaar, metric):
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
    
def plot_scatter(df_res_all,xaxis,yaxis,extra_info=True):
   
    # Voorbeeld dataframe
    # df_res_all bevat o.a. kolommen: Gemeente, Rank_Chi2_rtl_2023, Rank_Chi2_rtl_2025
    # df_res_all = pd.DataFrame(...)
    show_text = st.checkbox("Toon tekstlabels", key =f"show_text299r{random.randint(10,1000000) }", value=True)
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
                    "X: %{x}<br>"
                    "Y: %{y}<extra></extra>"
                ),
                marker=dict(size=8, color="teal", opacity=0.7, line=dict(width=0.5, color="white"))
            )
        ]
    )

        
    # Referentielijn x=y
    min_val = min(df_res_all[xaxis].min(), df_res_all[yaxis].min())
    max_val = max(df_res_all[xaxis].max(), df_res_all[yaxis].max())

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

    if extra_info:
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
        title=f"Vergelijking {xaxis} vs {yaxis}",
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        template="plotly_white",
        height=600
    )

    st.plotly_chart(fig, width=True)
 
def plot_scatter_correlation(df_, x_axis, y_axis, partij, indicator,mode_, log_inkomen):
    # kopie zodat df zelf niet verandert
    df = df_.copy()

    # log-transform als de kolom ink_inw heet
    if x_axis == "ink_inw" and log_inkomen:
        df[x_axis] = np.log(df[x_axis].astype(float))
        x_label = f"log({x_axis})"
    else:
        x_label = x_axis

    # Bereken lineaire regressie
    x = df[x_axis].astype(float)
    y = df[y_axis].astype(float)
    coeffs = np.polyfit(x, y, 1)
    slope, intercept = coeffs
    y_pred = slope * x + intercept

    # R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot

    # Maak plot
    fig = go.Figure()

    # Scatterpunten
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode=mode_,
            name="Gemeenten",
            text=df["Naam"],
            textposition="top center",
            marker=dict(size=8, color="teal", opacity=0.7, line=dict(width=0.5, color="white")),
            hovertemplate="<b>%{text}</b><br>"
                          f"{x_axis}: "+"%{x}<br>"
                          f"{y_axis}: "+"%{y}<extra></extra>",
        )
    )

    # Trendline
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred,
            mode="lines",
            name="Trendline",
            line=dict(color="gray", width=2, dash="dash"),
            hoverinfo="skip"
        )
    )

    # Layout
    fig.update_layout(
        title=f"{y_axis}<br> vs {x_axis} | (R² = {r2:.3f})",
        xaxis_title=x_label,
        yaxis_title=y_axis,
        template="plotly_white",
        height=600
    )
    # # Log-schaal als x-axis "ink_inw" is
    # if x_axis == "ink_inw":
    #     fig.update_xaxes(type="log")

   
    st.plotly_chart(fig)
       
def obesitas_inkomen():
  
    # https://www.vzinfo.nl/overgewicht/regionaal/obesitas#obesitas-volwassenen
    # https://www.cbs.nl/nl-nl/maatwerk/2023/35/inkomen-per-gemeente-en-wijk-2020
    #url_obesitas = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gemeente_overgewicht.csv"
    url_obesitas = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gemeente_overgewicht.csv"
    # https://www.clo.nl/indicatoren/nl210016-hbo-en-wo-gediplomeerden-2024
    # Bevolking 15 tot 75 jaar met als behaalde opleiding hbo of wo per gemeente, 2024

    url_opleiding = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gemeente_opleiding.csv"
    df_opleiding = pd.read_csv(url_opleiding)
    df_obesitas = pd.read_csv(url_obesitas)
    #url_inkomen = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gemeente_inkomen.csv"
    url_inkomen = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/gemeente_inkomen.csv"
    df_inkomen = pd.read_csv(url_inkomen)

    df_votes=  load_votes(2025)
    
    df_votes["aantal_votes"] =  round(df_votes["Waarde"] / df_votes["percentage_votes"],0) *100
    df_merge = df_obesitas.merge(df_votes, left_on="Naam", right_on="Regio", how="inner")
    df_merge = df_merge.merge(df_inkomen, left_on="Naam", right_on="Regio", how="inner")
    df_merge = df_merge.merge(df_opleiding, left_on="Naam", right_on="Regio", how="inner")
    col1,col2,col3=st.columns(3)
    with col1:
        partij = st.selectbox("Partij", sorted(df_merge["LijstNaam"].unique().tolist()), key="afdadsf", index=0)
    with col2:
        indicator_ = st.selectbox("Indicator gewicht", sorted(df_merge["Indicator"].unique().tolist()), key="aresf", index=0)
    with col3:
        show_text = st.checkbox("Toon tekstlabels", key="affadsf4", value=True)
        log_inkomen =  st.checkbox("Log inkomen", key="affadsf3",  value=False)
    if show_text:
        mode_="markers+text"
    else:
        mode_="markers"

    df_res=df_merge[(df_merge["Indicator"]==indicator_)& (df_merge["LijstNaam"]==partij)]
    df_res[f"Percentage_{indicator_}"] = df_res["Percentage"]
    df_res[f"percentage_votes_{partij}"] = df_res["percentage_votes"]
    col1,col2, =st.columns(2)
    with col1:
        plot_scatter_correlation(df_res,f"Percentage_{indicator_}",f"percentage_votes_{partij}", partij, indicator_,mode_,log_inkomen)
    

    with col2:
        plot_scatter_correlation(df_res,"HBO_WO_2024",f"percentage_votes_{partij}", partij,"",mode_,log_inkomen)
    col1,col2, =st.columns(2)
    with col1:
        plot_scatter_correlation(df_res,"ink_inw",f"percentage_votes_{partij}", partij, "",mode_,log_inkomen)
    with col2:
        plot_scatter_correlation(df_res,"ink_inw",f"Percentage_{indicator_}", partij, indicator_,mode_,log_inkomen)

    # plot_scatter_correlation(df_res,"percentage_votes","Percentage", partij, indicator_)
    # plot_scatter_correlation(df_res,"percentage_votes","ink_inw", partij, "")
    ols_corr(df_res, partij,indicator_)
    st.write(df_res)

def ols_corr(df, partij,indicator_):
    """Make an multiple lineair regression analyses (OLS)
    Args:
        df (dataframe): dataframe with the survey results
    """
    # Select predictors and target
    
    X = df[[f"Percentage_{indicator_}", "ink_inw", "HBO_WO_2024"]]
    y = df[f"percentage_votes_{partij}"]
    # y = df["nps"]

    # Drop rows with missing values
    data = pd.concat([X, y], axis=1).dropna()

    # Your original switch to predict Instructions from the others kept as-is
    X = data[[f"Percentage_{indicator_}", "ink_inw", "HBO_WO_2024"]]
    y = data[f"percentage_votes_{partij}"]
    # y=data["nps"]

    # Add constant for regression
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()
    st.write(model.summary())
    st.write(
        df[
            [
                f"percentage_votes_{partij}", f"Percentage_{indicator_}", "ink_inw", "HBO_WO_2024"
            ]
        ].corr()
    )
    df["y"] = (df[f"percentage_votes_{partij}"] / 100).clip(1e-6, 1 - 1e-6)
    for col in [f"Percentage_{indicator_}", "HBO_WO_2024", "ink_inw"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    model = sm.GLM(df["y"], X, family=sm.families.Binomial(), var_weights=df["aantal_votes"])
    res = model.fit()
    st.write(res.summary())

    
    beta_=False
    if beta_ :

        # voorbeeld: percentage stemmen op SP
        df = df.copy()
        epsilon = 1e-6
        df["y"] = (df[ f"percentage_votes_{partij}"] / 100).clip(epsilon, 1 - epsilon)

        # formule notatie zoals bij R
        formula = f"y ~  Percentage_{indicator_} + ink_inw + HBO_WO_2024"

        model = glm(formula=formula, data=df, family=Beta()).fit()
        st.write(model.summary())

        df["y_pred"] = model.predict(df)

        # Omrekenen naar procenten
        df["pred_percentage"] = df["y_pred"] * 100

        # Vergelijk werkelijk vs voorspeld
        import plotly.express as px
        fig = px.scatter(
            df,
            x=f"percentage_votes_{partij}",
            y="pred_percentage",
            hover_name="Regio",
            title=f"Beta-regressie: werkelijke vs voorspelde {partij}-stemmen",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="gray", dash="dash"))
        st.plotly_chart(fig)

        model_w = glm(formula=formula, data=df, family=Beta(), freq_weights=df["aantal_votes"]).fit()
        st.write(model_w.summary())

def all_results():
 
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
        df_res = calculate_results_landelijk(jaar, df)
        if jaar ==jaren[-1]:
            metric_=f"{metric}_{jaar}"
            make_map(df_res, jaar, metric_)
            st.write(f"Aantal gemeentes in {jaar} : {len(df_res)}")
            st.dataframe(df_res.style.format({"Chi2": "{:.4f}"}))
        if df_res_all.empty:
            df_res_all = df_res
        else:
            df_res_all = df_res_all.merge(df_res, on="Gemeente", how="inner")
    st.markdown("## Vergelijkingen")
    #st.dataframe(df_res_all.style.format({"Chi2": "{:.4f}"}))
    plot_scatter(df_res_all,xaxis=f"Rank_Chi2_rtl_2023", yaxis=f"Rank_Chi2_rtl_2025")
    plot_scatter(df_res_all,f"Rank_Chi2_rtl_2025",f"Rank_Chi2_prop_2025", False)
    plot_scatter(df_res_all,f"Rank_Chi2_rtl_2023",f"Rank_Chi2_test_2023", False)
    st.write(df_res_all)
    #plot_scatter(df_res_all,xaxis=f"Chi2_rtl_2023", yaxis=f"Chi2_rtl_2025")

def voorkeurscoalitie_per_gemeente():
    """Grootste coalitie per gemeente. Geinspireerd door https://x.com/maanvis81/status/1985848338588049737"""
    
    df = load_votes(2025)
    print (df)
    df_pivot = df.pivot_table(
        index="Regio",
        columns="LijstNaam",
        values="percentage_votes",
        aggfunc="sum"
    ).fillna(0)
   
    
    partijen = sorted(df_pivot.columns.tolist())

    # Standaardsets (filter op bestaande kolommen, "PVVD" wordt automatisch genegeerd als onbestaand)
    defaults = {
        "links":  ["GL-PvdA","D66", "VOLT"],   # 'PVVD' bestaat waarschijnlijk niet -> gefilterd
        "midden": [],
        "rechts": ["FvD","JA21", "PVV"],
    }
 
    # defaults = {
    #     "links":  ["GL-PvdA","D66","SP", "CDA"],   
    #     "midden": ["D66", "VVD", "CDA","GL-PvdA"],
    #     "rechts": ["D66", "VVD", "CDA", "JA21"],
    # }

    defaults = {
    "links": [
        "GL-PvdA",
        "D66",
        "SP",
        "PvdD",
        "BIJ1",
        "Volt",
    ],
    "midden": [
        "NSC",
        "CDA",
        "CU",
        "BBB",
    ],
    "rechts": [
        "VVD",
        "PVV",
        "JA21",
        "FvD",
        "SGP",
    ],
}

    
    defaults = {k: [p for p in v if p in partijen] for k, v in defaults.items()}

    col_l, col_m, col_r = st.columns(3)
    with col_l:
        sel_links  = st.multiselect("Links",  options=partijen, default=defaults["links"])
    with col_m:
        sel_midden = st.multiselect("Midden", options=partijen, default=defaults["midden"])
    with col_r:
        sel_rechts = st.multiselect("Rechts", options=partijen, default=defaults["rechts"])

    # Nieuwe kolommen met som per blok
    def sum_or_zero(df, cols):
        return df[cols].sum(axis=1) if len(cols) else 0

    df_pivot = df_pivot.copy()
    df_pivot["centrum_links_total"]  = sum_or_zero(df_pivot, sel_links)
    df_pivot["centrum_midden_total"] = sum_or_zero(df_pivot, sel_midden)
    df_pivot["centrum_rechts_total"] = sum_or_zero(df_pivot, sel_rechts)

   
    
    # kolommen met sommen uit je vorige stap
    cols = ["centrum_links_total", "centrum_midden_total", "centrum_rechts_total"]

    mx = df_pivot[cols].max(axis=1)
    ties = df_pivot[cols].eq(mx, axis=0).sum(axis=1) > 1
    ix = df_pivot[cols].values.argmax(axis=1)
    winner = pd.Series(ix, index=df_pivot.index).map({0:1, 1:2, 2:3})
    df_pivot["populairste_coalitie"] = np.where(ties, 0, winner)
    df_pivot = df_pivot.reset_index()
    df_pivot["Gemeente"] =df_pivot["Regio"]
    make_map(df_pivot,2025,"populairste_coalitie")
    st.dataframe(df_pivot.round(2))
    



def kaart_per_partij():
    """Maak een kaart per partij"""
 
    jaren=[2023,2025]
    jaar = st.radio("Jaar", jaren, index=1, horizontal=True, key="partij_jaar")
    df_j = load_votes(jaar)
    
    

    #check = df.groupby("Regio")["percentage"].sum().round(2)
    df_j["Gemeente"]=df_j["Regio"]
    
    partij = st.selectbox("Partij", sorted(df_j["LijstNaam"].unique().tolist()), index=0)
    df_p=df_j[df_j["LijstNaam"] == partij]
    
    if len(df_p)>0:
      

        df_p=df_p[["Gemeente","Waarde","percentage_votes"]].sort_values("percentage_votes", ascending=False)
        df_p["Zetels"] = round(df_p["percentage_votes"]/0.66667,1)
        #df_p[f"Percentage_{partij}"] = df_p["percentage"]
        
        df_p = df_p.rename(columns={"percentage_votes": f"Percentage_{partij}"})
        
        
        try:
            make_map(df_p, jaar, f"Percentage_{partij}")
        except:
            #error VRIJVER 2025
            st.error("Fout bij het maken van de kaart")
        st.write(df_p)
    else:
        st.error("Partij heeft geen stemmen")
        st.stop()

def main():
    """Main functie
    """  

    jaren=[2023,2025] 
    tab1, tab2, tab3,tab4,tab5,tab6 = st.tabs(["Resultaten", "Enkele gemeente","Partij","voorkeurscoalitie","obesitas/inkomen/opleiding", "Info"])

    with tab6:
        show_info()
    with tab4:
        voorkeurscoalitie_per_gemeente()
    with tab5:
         obesitas_inkomen()
    with tab1:
        all_results()   
    with tab2:
        jaar = st.radio("Jaar", jaren, index=1, horizontal=True, key="gemeente_jaar")
        df_j = load_votes(jaar)
        calculate_results_gemeente(df_j, jaar)
    with tab3:
        kaart_per_partij()
        
if __name__ == "__main__":
    main()
