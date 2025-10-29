import pandas as pd
from scipy.stats import chi2_contingency
import streamlit as st
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
import branca.colormap as cm


def show_info():
    st.info("Reproductie van https://www.rtl.nl/nieuws/politiek/artikel/5535930/apeldoorn-nederland-het-klein-zo-gemiddeld-stemt-jouw-gemeente")
    st.info("Data  https://data.overheid.nl/dataset/verkiezingsuitslag-tweede-kamer-2023?utm_source=chatgpt.com#panel-description")
    st.info("Gemeentegrenzen  https://cartomap.github.io/nl/")

    st.info("""FORMULES
            
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
    """)
    
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
def get_df():
    #url = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\TK2023_uitslag_gemeente.csv"
    url = "https://https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/TK2023_uitslag_gemeente.csv"
    df = pd.read_csv(url, sep=";")

    
    
    df=df[df["LijstNaam"] != None] 
    df = df[df["RegioCode"].str.startswith("G", na=False)]
    df=df[df["VeldType"] == "LijstAantalStemmen"] 
    return df

def calculate_results_gemeente(df):
    # Pas deze kolomnamen aan als ze anders zijn
    kol_regio = "Regio"
    kol_partij = "LijstNaam"
    kol_stemmen = "Waarde"
    gemeentes = sorted(df[kol_regio].unique().tolist())
    index_leiden = gemeentes.index("Leiden")  # geeft positie van 'Leiden'
    uitgelichte_gemeente = st.selectbox("Gemeente", gemeentes, index=index_leiden)


    u_df = df[df[kol_regio] == uitgelichte_gemeente]
    st.write(f"Totaal aantal geldige stemmen in {uitgelichte_gemeente} = {u_df['Waarde'].sum()}")


    # Som stemmen per partij per regio
    agg = (
        df.groupby([kol_regio, kol_partij], as_index=False)[kol_stemmen]
        .sum()
    )

    # Selecteer uitgelicht
    apel = agg.query(f"{kol_regio} == '{uitgelichte_gemeente}'")[[kol_partij, kol_stemmen]].rename(columns={kol_stemmen: uitgelichte_gemeente})

    # Landelijk totaal
    landelijk = (
        agg.groupby(kol_partij, as_index=False)[kol_stemmen]
        .sum()
        .rename(columns={kol_stemmen: "Nederland"})
    )

    # Merge
    m = pd.merge(landelijk, apel, on=kol_partij, how="inner").set_index(kol_partij)

    # Bereken percentages
    m["% Nederland"] =  m["Nederland"] / m["Nederland"].sum()
    m[f"% {uitgelichte_gemeente}"] =  m[uitgelichte_gemeente] / m[uitgelichte_gemeente].sum()
    m["Verschil (pp)"] = m[f"% {uitgelichte_gemeente}"] - m["% Nederland"]

    # Chi-kwadraattoets
    obs = m[[uitgelichte_gemeente, "Nederland"]].T.values
    chi2, p, dof, expected = chi2_contingency(obs)

    # Chi2 op basis van proporties
    chi2_prop = ((m[f"% {uitgelichte_gemeente}"] - m["% Nederland"])**2 / m["% Nederland"]).sum()
    chi2_rtl = (   abs ( m[f"% {uitgelichte_gemeente}"] - m["% Nederland"]) ).sum()


    # Resultaat tonen
    st.subheader(f"{uitgelichte_gemeente} vs Nederland – Tweede Kamer 2023")
    st.dataframe(m.sort_values("% Nederland", ascending=False).round(2))



    st.markdown(f"""
    **Chi-kwadraattoets**

    - χ² = {chi2_prop:.3f} (proporties)
    - RTL = {chi2_rtl:.3f} (RTL-methode)
    
    """)


def calculate_results_landelijk(df):
    # Pas deze kolomnamen aan als ze anders zijn
    kol_regio = "Regio"
    kol_partij = "LijstNaam"
    kol_stemmen = "Waarde"
    agg = df.groupby([kol_regio, kol_partij], as_index=False)[kol_stemmen].sum()
    landelijk = agg.groupby(kol_partij, as_index=False)[kol_stemmen].sum()
    landelijk.columns = [kol_partij, "Nederland"]

    # Landelijke verdeling in fracties
    landelijk["p_landelijk"] = landelijk["Nederland"] / landelijk["Nederland"].sum()

    resultaten = []
    gemeenten = agg[kol_regio].unique()

    for g in gemeenten:
        lokaal = agg.query(f"{kol_regio} == @g")[[kol_partij, kol_stemmen]].rename(columns={kol_stemmen: g})
        m = pd.merge(landelijk, lokaal, on=kol_partij, how="inner").fillna(0)
        m["p_gemeente"] = m[g] / m[g].sum()


        # Chi-kwadraattoets
        obs = m[[g, "Nederland"]].T.values
        chi2, p, dof, expected = chi2_contingency(obs)
        # Chi2 op basis van proporties
        chi2_prop = ((m["p_gemeente"] - m["p_landelijk"])**2 / m["p_landelijk"]).sum()
        chi2_rtl = (   abs ( m["p_gemeente"] - m["p_landelijk"]) ).sum()

        #resultaten.append({"Gemeente": g, "Chi2": chi2, "Chi2_prop": chi2_prop})
        resultaten.append({"Gemeente": g, "Chi2_prop": chi2_prop,"Chi2_rtl": chi2_rtl})


    df_res = pd.DataFrame(resultaten).sort_values("Chi2_rtl", ascending=False)
    for fieldname in ["Chi2_rtl","Chi2_prop"]:
    # Rangorde op basis van Chi2_prop, hoogste eerst
        df_res[f"Rank_{fieldname}"] = df_res[fieldname].rank(method="dense", ascending=True).astype(int)
        df_res[f"Percentiel_{fieldname}"] = df_res[fieldname].rank(pct=True, ascending=True) * 100
    # df_res_chosen = df_res.query(f"Gemeente == '{uitgelichte_gemeente}'").iloc[0]
    # st.write(df_res_chosen)
    
    return df_res
def make_plot(df_res):
    columns_metrics = ['Chi2_prop', 'Chi2_rtl', 'Rank_Chi2_rtl', 'Percentiel_Chi2_rtl', 'Rank_Chi2_prop', 'Percentiel_Chi2_prop']

    GEO_PATH = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\gemeente_2023.geojson"  # zorg dat GM_NAAM in properties staat
    GEO_PATH="https://https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/gemeente_2023.geojson"
    with open(GEO_PATH, "r", encoding="utf-8") as f:
        gjson = json.load(f)

    fix = {
        "s-Hertogenbosch": "'s-Hertogenbosch",
        "Nuenen, Gerwen en Nederwetten": "Nuenen, Gerwen en Nederwetten",  # voorbeeld
    }
    df_res["Gemeente_fix"] = df_res["Gemeente"].replace(fix)



        

    # 2) Keuze metriek
    metric = st.radio("Kleur op", columns_metrics,2, horizontal=True)
    title = "Waar wordt het meest gemiddeld gestemd?" if metric=="Chi2_prop" else "Afwijking t.o.v. Nederland"

    # 3) Harmoniseer naamvelden
    # Zorg dat namen exact matchen: df_res.Gemeente <-> feature.properties.GM_NAAM
    # Eventueel kleine fixes:

    
    # 4) Range en colormap
    vmin = float(df_res[metric].min())
    vmax = float(df_res[metric].max())
    cmap = cm.LinearColormap(
        colors=["#e8f3ec", "#a9d0c3", "#6fb0a1", "#3e7e80", "#214b5a"],
        vmin=vmin, vmax=vmax
    ).to_step(7)
    cmap.caption = "gemiddeld            zeer afwijkend" if metric=="Chi2_prop" else "laag             hoog"

    # 5) Map
    m = folium.Map(location=[52.2, 5.3], zoom_start=7, tiles="cartodbpositron", control_scale=True)

    # 1) Map DataFrame-data naar GeoJSON
    # Voeg jouw df_res-data toe aan de GeoJSON via 'statnaam'
    data_dict = df_res.set_index("Gemeente_fix").to_dict(orient="index")

    for feature in gjson["features"]:
        name = feature["properties"].get("statnaam")
        if name in data_dict:
            feature["properties"].update(data_dict[name])

    # 2) Range en colormap
    vmin, vmax = float(df_res[metric].min()), float(df_res[metric].max())
    cmap = cm.LinearColormap(
        colors=["#e8f3ec", "#a9d0c3", "#6fb0a1", "#3e7e80", "#214b5a"],
        vmin=vmin, vmax=vmax
    ).to_step(7)
    cmap.caption = "gemiddeld → zeer afwijkend" if metric == "Chi2_prop" else "laag → hoog"

    # 3) Kaart en stijl
    def style_function(feature):
        val = feature["properties"].get(metric)
        color = "#cccccc" if val is None else cmap(val)
        return {"fillColor": color, "color": "#ffffff", "weight": 0.6, "fillOpacity": 0.85}

    # 4) Tooltip met ALLE velden uit df_res + statnaam
    tooltip_fields = ["statnaam"] + [c for c in df_res.columns if c != "Gemeente_fix"]
    tooltip_aliases = tooltip_fields

    m = folium.Map(location=[52.2, 5.3], zoom_start=7, tiles="cartodbpositron", control_scale=True)

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
            labels=True
        )
    )
    gj.add_to(m)
    cmap.add_to(m)

    st.markdown(f"## Kaart: {metric}")
    st_folium(m, height=700)

    st.dataframe(df_res.style.format({"Chi2": "{:.4f}"}))

def main():
    tab1, tab2,tab3 = st.tabs(   ["Resultaten", "Enkele gemeente", "Info"] )
    
    with tab3:
        show_info()
    with tab1:
        df = get_df()
        df_res = calculate_results_landelijk(df)

       
        make_plot(df_res)
    with tab2:
        calculate_results_gemeente(df)

if __name__ == "__main__":
    main()