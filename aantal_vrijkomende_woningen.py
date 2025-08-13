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
    Tel per jaar wie 18 is NA veroudering en VOOR sterfte
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

        n18 = pop_f.loc[pop_f["age"] == 18, "pop"].sum() + pop_m.loc[pop_m["age"] == 18, "pop"].sum()
        rows.append({"jaar": year_int, "n18": float(n18)})

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
    st.title("Woningstromen 70–120 overlijdens, 18-jarigen, immigratie en emigratie")

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

    # Instellingen
    cA, cB, cC = st.columns(3)
    with cA:
        jaren = st.slider("Aantal jaren vooruit", 5, 30, 15, 1)
    with cB:
        woningfactor_overlijden = st.slider("Woningen per overlijden", 0.20, 1.00, 0.60, 0.05)
    with cC:
        factor_18 = st.slider("Factor 18-jarigen", 0.20, 1.00, 0.80, 0.05)

    # Nieuwe schuiven immigratie en emigratie
    # https://www.cbs.nl/nl-nl/nieuws/2025/05/lagere-bevolkingsgroei-in-2024
    st.subheader("Immigratie en emigratie per jaar")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        immigranten_per_jaar = st.number_input("Immigranten per jaar", min_value=0, max_value=1_000_000, value=300_000, step=5_000)
    with d2:
        factor_immigratie = st.slider("Factor woningvraag immigratie", 0.20, 1.00, 0.50, 0.05)
    with d3:
        emigranten_per_jaar = st.number_input("Emigranten per jaar", min_value=0, max_value=1_000_000, value=200_000, step=5_000)
    with d4:
        factor_emigratie = st.slider("Factor woningen vrij emigratie", 0.20, 1.00, 0.50, 0.05)

    # Overlijdens 70–120 en woningen vrij
    with st.spinner("Simuleer overlijdens 70–120"):
        df_deaths = simuleer_overlijden_70_120(population_data, mort_f, mort_m, jaren=jaren)
    df_deaths["woningen_vrij_overlijden"] = df_deaths["deaths_total"] * woningfactor_overlijden

    # 18-jarigen en woningvraag
    with st.spinner("Simuleer 18-jarigen"):
        df_18 = simuleer_populatie_0_120_met_geboortes(population_data, mort_f, mort_m, jaren=jaren)
    df_18["woningen_vraag_18"] = df_18["n18"] * factor_18

    # Immigratie en emigratie per jaar
    jaren_lijst = list(range(2025, 2024 + jaren + 1))
    df_migr = pd.DataFrame({
        "jaar": jaren_lijst,
        "woningen_vraag_immigratie": [immigranten_per_jaar * factor_immigratie] * len(jaren_lijst),
        "woningen_vrij_emigratie": [emigranten_per_jaar * factor_emigratie] * len(jaren_lijst),
    })

    # Merge alle stromen
    df = df_deaths.merge(df_18, on="jaar", how="outer").merge(df_migr, on="jaar", how="outer").sort_values("jaar")

    # Bereken saldo per jaar
    df["woningen_vrij_totaal"] = df["woningen_vrij_overlijden"].fillna(0) + df["woningen_vrij_emigratie"].fillna(0)
    df["woningen_vraag_totaal"] = df["woningen_vraag_18"].fillna(0) + df["woningen_vraag_immigratie"].fillna(0)
    df["saldo_woningen"] = df["woningen_vrij_totaal"] - df["woningen_vraag_totaal"]

    st.subheader("Per jaar")
    st.dataframe(
        df[[
            "jaar",
            "woningen_vrij_overlijden",
            "woningen_vrij_emigratie",
            "woningen_vrij_totaal",
            "woningen_vraag_18",
            "woningen_vraag_immigratie",
            "woningen_vraag_totaal",
            "saldo_woningen"
        ]].round(0).rename(columns={
            "woningen_vrij_overlijden": "vrij_overlijden",
            "woningen_vrij_emigratie": "vrij_emigratie",
            "woningen_vrij_totaal": "vrij_totaal",
            "woningen_vraag_18": "vraag_18",
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

    st.caption("18-jarigen geteld na verouderen en voor sterfte. Geboortes constant op 1-jarigen 2024 per geslacht. Migratie als vaste jaarstroom met factor.")

    st.divider()
    st.write("Bronnen  CBS bevolking per leeftijd 2024  AG2024 prognosetafel")

if __name__ == "__main__":
    main()
