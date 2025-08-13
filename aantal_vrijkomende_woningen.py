import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------------------- Helpers ----------------------------
def prepare_mortality(df):
    df = df.copy()
    df["age"] = df["age"].astype(int)
    for c in df.columns:
        if str(c).isdigit():
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def simuleer_overlijden_70_120(population_data, mort_f, mort_m, jaren=15):
    """
    Simuleer cohort 70..120 vanaf 2024 met jaar-specifieke sterftekansen
    Retourneert per jaar de overlijdens in dit cohort
    """
    pop24 = population_data[population_data["jaar"] == 2024].copy()
    
    cohort_f = pop24[(pop24["geslacht"] == "F") & (pop24["leeftijd"].between(70, 120))][["leeftijd", "aantal"]].copy()
    cohort_m = pop24[(pop24["geslacht"] == "M") & (pop24["leeftijd"].between(70, 120))][["leeftijd", "aantal"]].copy()
    cohort_f["survivors"] = cohort_f["aantal"]
    cohort_m["survivors"] = cohort_m["aantal"]

    years = [str(y) for y in range(2025, 2024 + jaren + 1)]
    rows = []
    for y in years:
        cohort_f["leeftijd"] += 1
        cohort_m["leeftijd"] += 1

        f = cohort_f.merge(mort_f[["age", y]], left_on="leeftijd", right_on="age", how="left")
        m = cohort_m.merge(mort_m[["age", y]], left_on="leeftijd", right_on="age", how="left")
        f[y] = f[y].fillna(0.0)
        m[y] = m[y].fillna(0.0)

        deaths_f = f["survivors"] * f[y]
        deaths_m = m["survivors"] * m[y]

        cohort_f = f.drop(columns=["age", y])
        cohort_m = m.drop(columns=["age", y])

        cohort_f["survivors"] -= deaths_f
        cohort_m["survivors"] -= deaths_m

        rows.append({"jaar": int(y), "deaths_total": float(deaths_f.sum() + deaths_m.sum())})

    return pd.DataFrame(rows)

def simuleer_populatie_0_120_met_geboortes(population_data, mort_f, mort_m, jaren=15):
    """
    Simuleer totale populatie met constante geboortes = aantal 1-jarigen in 2024 per geslacht
    Tel per jaar wie 24 is NA veroudering en VOOR sterfte
    """
    pop24 = population_data[population_data["jaar"] == 2024].copy()

    base_f = pop24[pop24["geslacht"] == "F"][["leeftijd", "aantal"]].rename(columns={"leeftijd": "age", "aantal": "pop"}).copy()
    base_m = pop24[pop24["geslacht"] == "M"][["leeftijd", "aantal"]].rename(columns={"leeftijd": "age", "aantal": "pop"}).copy()
    base_f["age"] = base_f["age"].astype(int)
    base_m["age"] = base_m["age"].astype(int)

    births_f_const = float(pop24[(pop24["geslacht"] == "F") & (pop24["leeftijd"] == 1)]["aantal"].sum())
    births_m_const = float(pop24[(pop24["geslacht"] == "M") & (pop24["leeftijd"] == 1)]["aantal"].sum())

    years = [str(y) for y in range(2025, 2024 + jaren + 1)]
    rows = []
    pop_f = base_f.copy()
    pop_m = base_m.copy()

    for y in years:
        year_int = int(y)
        pop_f["age"] += 1
        pop_m["age"] += 1
        pop_f.loc[pop_f["age"] > 120, "age"] = 120
        pop_m.loc[pop_m["age"] > 120, "age"] = 120

        n24 = pop_f.loc[pop_f["age"] == 24, "pop"].sum() + pop_m.loc[pop_m["age"] == 24, "pop"].sum()
        rows.append({"jaar": year_int, "n24": float(n24)})

        f = pop_f.merge(mort_f[["age", y]], on="age", how="left")
        m = pop_m.merge(mort_m[["age", y]], on="age", how="left")
        f[y] = f[y].fillna(0.0)
        m[y] = m[y].fillna(0.0)
        f["pop"] = f["pop"] * (1.0 - f[y])
        m["pop"] = m["pop"] * (1.0 - m[y])
        pop_f = f.drop(columns=[y])
        pop_m = m.drop(columns=[y])

        pop_f = pd.concat([pop_f, pd.DataFrame({"age": [0], "pop": [births_f_const]})], ignore_index=True)
        pop_m = pd.concat([pop_m, pd.DataFrame({"age": [0], "pop": [births_m_const]})], ignore_index=True)
        pop_f = pop_f.groupby("age", as_index=False)["pop"].sum()
        pop_m = pop_m.groupby("age", as_index=False)["pop"].sum()

    return pd.DataFrame(rows)

def format_int(x):
    return f"{int(round(x)):,}".replace(",", ".")

# ---------------------------- App ----------------------------
def main():
    st.title("Woningstromen 70–120 overlijdens, 24-jarigen, immigratie en emigratie")

    # Data
    population_data = pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/bevolking_leeftijd_NL.csv",
        sep=";"
    )
    mort_f = prepare_mortality(pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_female.csv"
    ))
    mort_m = prepare_mortality(pd.read_csv(
        "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/AG2024DefinitiefGevalideerd_male.csv"
    ))

    jaren = st.slider("Aantal jaren vooruit", 5, 30, 15, 1)
    # Instellingen
    cB, cC = st.columns(2)
    
    
    
    with cB:
        woningfactor_overlijden = st.slider("Woningen per overlijden", 0.20, 1.00, 0.60, 0.05)
    with cC:
        factor_24 = st.slider("Woningen voor elke nieuwe 24-jarige", 0.20, 1.00, 0.50, 0.05)

    # Nieuwe schuiven immigratie en emigratie
    # https://www.cbs.nl/nl-nl/nieuws/2025/05/lagere-bevolkingsgroei-in-2024
    st.subheader("Immigratie en emigratie per jaar")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        emigranten_per_jaar = st.number_input("Emigranten per jaar", min_value=0, max_value=1_000_000, value=200_000, step=5_000)
    with d2:
        factor_emigratie = st.slider("Factor woningen vrij emigratie", 0.00, 1.00, 0.50, 0.05)

    with d3:
        immigranten_per_jaar = st.number_input("Immigranten per jaar", min_value=0, max_value=1_000_000, value=300_000, step=5_000)
    with d4:
        factor_immigratie = st.slider("Factor woningvraag immigratie", 0.00, 1.00, 0.50, 0.05)
    
    # Overlijdens 70–120 en woningen vrij
    with st.spinner("Simuleer overlijdens 70–120"):
        df_deaths = simuleer_overlijden_70_120(population_data, mort_f, mort_m, jaren=jaren)
    df_deaths["woningen_vrij_overlijden"] = df_deaths["deaths_total"] * woningfactor_overlijden

    # 24-jarigen en woningvraag
    with st.spinner("Simuleer 24-jarigen"):
        df_24 = simuleer_populatie_0_120_met_geboortes(population_data, mort_f, mort_m, jaren=jaren)
    df_24["woningen_vraag_24"] = df_24["n24"] * factor_24

    # Immigratie en emigratie per jaar
    jaren_lijst = list(range(2025, 2024 + jaren + 1))
    df_migr = pd.DataFrame({
        "jaar": jaren_lijst,
        "woningen_vraag_immigratie": [immigranten_per_jaar * factor_immigratie] * len(jaren_lijst),
        "woningen_vrij_emigratie": [emigranten_per_jaar * factor_emigratie] * len(jaren_lijst),
    })

    # Merge alle stromen
    df = df_deaths.merge(df_24, on="jaar", how="outer").merge(df_migr, on="jaar", how="outer").sort_values("jaar")

    # Bereken saldo per jaar
    df["woningen_vrij_totaal"] = df["woningen_vrij_overlijden"].fillna(0) + df["woningen_vrij_emigratie"].fillna(0)
    df["woningen_vraag_totaal"] = df["woningen_vraag_24"].fillna(0) + df["woningen_vraag_immigratie"].fillna(0)
    df["saldo_woningen"] = df["woningen_vrij_totaal"] - df["woningen_vraag_totaal"]

    st.subheader("Per jaar")
    st.dataframe(
        df[[
            "jaar",
            "woningen_vrij_overlijden",
            "woningen_vrij_emigratie",
            "woningen_vrij_totaal",
            "woningen_vraag_24",
            "woningen_vraag_immigratie",
            "woningen_vraag_totaal",
            "saldo_woningen"
        ]].round(0).rename(columns={
            "woningen_vrij_overlijden": "vrij_overlijden",
            "woningen_vrij_emigratie": "vrij_emigratie",
            "woningen_vrij_totaal": "vrij_totaal",
            "woningen_vraag_24": "vraag_24",
            "woningen_vraag_immigratie": "vraag_immigratie",
            "woningen_vraag_totaal": "vraag_totaal",
            "saldo_woningen": "saldo"
        }).style.format("{:,.0f}")
    )

    st.subheader("Grafiek")
    melt = df.melt(
        id_vars="jaar",
        value_vars=["woningen_vrij_totaal", "woningen_vraag_totaal", "saldo_woningen"],
        var_name="type", value_name="woningen"
    )
    fig = px.line(melt, x="jaar", y="woningen", color="type",
                  title="Woningen vrij totaal vs woningvraag totaal en saldo")
    st.plotly_chart(fig, use_container_width=True)



    # Totalen
    overlijden_vrij = df["woningen_vrij_overlijden"].sum()
    in_vraag = df["woningen_vraag_24"].sum()
    saldo_tot =  df["woningen_vrij_overlijden"].sum() - df["woningen_vraag_24"].sum()

    st.subheader(f"Natuurlijk verloop {jaren} jaar")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Woningen vrij overlijden", format_int(overlijden_vrij))
    with t2:
        st.metric("Woningen vraag 24 jarigen", format_int(in_vraag))
    with t3:
        st.metric("Saldo", format_int(saldo_tot))


    # Totalen Migratie
    m_vrij = df["woningen_vrij_emigratie"].sum()
    m_vraag = df["woningen_vraag_immigratie"].sum()
    m_saldo_tot = df["woningen_vrij_emigratie"].sum() - df["woningen_vraag_immigratie"].sum() 

    st.subheader(f"Migratie {jaren} jaar")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Woningen vrij emigratie", format_int(m_vrij))
    with t2:
        st.metric("Woningen vraag immigratie", format_int(m_vraag))
    with t3:
        st.metric("Saldo", format_int(m_saldo_tot))

    # Totalen
    totaal_vrij = df["woningen_vrij_totaal"].sum()
    totaal_vraag = df["woningen_vraag_totaal"].sum()
    saldo_tot = df["saldo_woningen"].sum()

    st.subheader(f"Totaal {jaren} jaar")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.metric("Woningen vrij totaal", format_int(totaal_vrij))
    with t2:
        st.metric("Woningen vraag totaal", format_int(totaal_vraag))
    with t3:
        st.metric("Saldo", format_int(saldo_tot))

    st.caption("24-jarigen geteld na verouderen en voor sterfte. Geboortes constant op 1-jarigen 2024 per geslacht. Migratie als vaste jaarstroom met factor.")

    st.divider()
    st.write("**Bronnen **")
    
    
    
    st.write("* CBS bevolking per leeftijd 2024 : https://opendata.cbs.nl/#/CBS/nl/dataset/03759ned/table?dl=39E0B")
    st.write("* https://www.cbs.nl/nl-nl/nieuws/2025/05/lagere-bevolkingsgroei-in-2024")
    st.write("*  AG2024 prognosetafel - https://www.actuarieelgenootschap.nl/kennisbank/prognosetafel-ag2024-2")

if __name__ == "__main__":

    main()
