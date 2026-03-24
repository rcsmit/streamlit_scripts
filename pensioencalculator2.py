"""
Pensioen Calculator — Cohortmodel
===================================
Simuleert een cohort van N personen.
Overledenen onttrekken geen uitkering meer; hun aandeel in de pot
komt ten goede aan de overlevenden (solidariteitsprincipe).

Doel: de collectieve pensioenpot eindigt op €0.
"""

import csv
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import brentq

# ── Paginaconfiguratie ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pensioen calculator",
    page_icon=":material/savings:",
    layout="wide",
)

# ── AG2024 sterftetafel laden ─────────────────────────────────────────────────

AG2024_PATH = Path("AG2024DefinitiefGevalideerd_male.csv")


@st.cache_data
def load_mortality(url: str) -> dict:
    df = pd.read_csv(url)
    mortality = {}
    for _, row in df.iterrows():
        age = int(row["age"])
        mortality[age] = {int(k): float(v) for k, v in row.items() if k != "age"}
    return mortality

@st.cache_data
def load_mortality_(path: Path) -> dict:
    mortality = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            age = int(row["age"])
            mortality[age] = {int(k): float(v) for k, v in row.items() if k != "age"}
    return mortality


def get_mort(mortality, age, year):
    table = mortality.get(age, {})
    if year in table:
        return table[year]
    if table:
        return table[min(table.keys(), key=lambda y: abs(y - year))]
    return 0.0


# ── Cohortmodel ───────────────────────────────────────────────────────────────

def simulate_cohort(inleg_per_jaar, beleggingsresultaat, inflatie, uitkering_jaar1,
                    start_leeftijd, pensioenleeftijd, max_leeftijd, mortality,
                    n_start=1000, start_year=2024):
    n_opbouw = pensioenleeftijd - start_leeftijd
    n_totaal = max_leeftijd - start_leeftijd

    pot = 0.0
    levenden = float(n_start)
    records = []

    for t in range(n_totaal):
        leeftijd = start_leeftijd + t
        year = start_year + t
        fase = "opbouw" if t < n_opbouw else "uitkering"

        q = get_mort(mortality, leeftijd, year)
        overlijdend = levenden * q
        levenden_eind = levenden - overlijdend

        if fase == "opbouw":
            inleg_pp = inleg_per_jaar * (inflatie ** t)
            inleg_totaal = inleg_pp * levenden
            uitkering_pp = 0.0
            uitkering_totaal = 0.0
            pot = pot * beleggingsresultaat + inleg_totaal
        else:
            u_jaar = t - n_opbouw
            inleg_pp = 0.0
            inleg_totaal = 0.0
            uitkering_pp = uitkering_jaar1 * (inflatie ** u_jaar)
            uitkering_totaal = uitkering_pp * levenden_eind
            pot = pot * beleggingsresultaat - uitkering_totaal

        records.append({
            "jaar": t + 1,
            "leeftijd": leeftijd,
            "fase": fase,
            "overlijdenskans": q,
            "levenden_begin": levenden,
            "overlijdend": overlijdend,
            "levenden_eind": levenden_eind,
            "inleg_pp": inleg_pp,
            "inleg_totaal": inleg_totaal,
            "uitkering_pp": uitkering_pp,
            "uitkering_totaal": uitkering_totaal,
            "pensioenpot": pot,
        })
        levenden = levenden_eind

    return records


def solve_uitkering(inleg_per_jaar, beleggingsresultaat, inflatie,
                    start_leeftijd, pensioenleeftijd, max_leeftijd, mortality, n_start):
    def eindwaarde(u):
        rows = simulate_cohort(inleg_per_jaar, beleggingsresultaat, inflatie, u,
                               start_leeftijd, pensioenleeftijd, max_leeftijd, mortality, n_start)
        return rows[-1]["pensioenpot"]
    try:
        return brentq(eindwaarde, 1.0, 10_000_000.0, xtol=0.01)
    except ValueError:
        return 0.0


# ── Grafieken ─────────────────────────────────────────────────────────────────

def grafiek_pot(df, pensioenleeftijd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["leeftijd"], y=df["pensioenpot"] / 1_000_000,
        name="Collectieve pot (M€)",
        line={"color": "#2563EB", "width": 2.5},
        fill="tozeroy", fillcolor="rgba(37,99,235,0.08)",
    ))
    opbouw = df[df["fase"] == "opbouw"]
    uitkering_df = df[df["fase"] == "uitkering"]
    fig.add_trace(go.Bar(
        x=opbouw["leeftijd"], y=opbouw["inleg_totaal"] / 1_000,
        name="Inleg totaal (K€/jaar)",
        marker_color="rgba(16,185,129,0.65)", yaxis="y2",
    ))
    fig.add_trace(go.Bar(
        x=uitkering_df["leeftijd"], y=uitkering_df["uitkering_totaal"] / 1_000,
        name="Uitkering totaal (K€/jaar)",
        marker_color="rgba(245,158,11,0.65)", yaxis="y2",
    ))
    fig.add_vline(x=pensioenleeftijd, line_dash="dash", line_color="#6B7280",
                  annotation_text=f"Pensioen {pensioenleeftijd}", annotation_position="top right")
    fig.update_layout(
        title="Collectieve pensioenpot",
        xaxis_title="Leeftijd",
        yaxis={"title": "Pensioenpot (M€)"},
        yaxis2={"title": "Jaarlijkse stroom (K€)", "overlaying": "y",
                "side": "right", "showgrid": False},
        barmode="overlay",
        legend={"orientation": "h", "y": -0.18},
        hovermode="x unified", height=460,
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"t": 50, "b": 90},
    )
    return fig


def grafiek_levenden(df, pensioenleeftijd):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["leeftijd"], y=df["levenden_eind"],
        name="Levenden (eind jaar)",
        line={"color": "#059669", "width": 2},
        fill="tozeroy", fillcolor="rgba(5,150,105,0.08)",
    ))
    fig.add_trace(go.Bar(
        x=df["leeftijd"], y=df["overlijdend"],
        name="Overledenen per jaar",
        marker_color="rgba(220,38,38,0.5)", yaxis="y2",
    ))
    fig.add_vline(x=pensioenleeftijd, line_dash="dash", line_color="#6B7280")
    fig.update_layout(
        title="Cohort: overleving",
        xaxis_title="Leeftijd",
        yaxis={"title": "Levenden"},
        yaxis2={"title": "Overledenen per jaar", "overlaying": "y",
                "side": "right", "showgrid": False},
        legend={"orientation": "h", "y": -0.18},
        hovermode="x unified", height=360,
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"t": 50, "b": 90},
    )
    return fig


def grafiek_uitkering_pp(df):
    uit = df[df["fase"] == "uitkering"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=uit["leeftijd"], y=uit["uitkering_pp"],
        name="Uitkering per persoon (€/jaar)",
        line={"color": "#F59E0B", "width": 2.5},
        fill="tozeroy", fillcolor="rgba(245,158,11,0.10)",
    ))
    fig.update_layout(
        title="Uitkering per persoon per jaar (stijgt met inflatie)",
        xaxis_title="Leeftijd",
        yaxis_title="Uitkering (€/jaar)",
        height=320,
        plot_bgcolor="white", paper_bgcolor="white",
        margin={"t": 50, "b": 40},
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

st.title(":material/savings: Pensioen calculator — cohortmodel")
st.caption(
    "Een cohort van N personen legt samen in. Overledenen onttrekken geen "
    "uitkering meer; hun aandeel blijft in de collectieve pot (solidariteitsprincipe). "
    "De uitkering wordt zo bepaald dat de pot op de eindleeftijd precies €0 is."
)


# # Sterftetafel laden
# if AG2024_PATH.exists():
#     mortality = load_mortality(AG2024_PATH)
# else:
#     st.error(f":material/error: Sterftetafel niet gevonden: `{AG2024_PATH}`. Upload hieronder.")
#     uploaded = st.file_uploader("Upload AG2024DefinitiefGevalideerd_male.csv", type="csv")
#     if uploaded is None:
#         st.stop()
#     content = uploaded.read().decode("utf-8-sig")
#     reader = csv.DictReader(io.StringIO(content))
#     mortality = {}
#     for row in reader:
#         age = int(row["age"])
#         mortality[age] = {int(k): float(v) for k, v in row.items() if k != "age"}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header(":material/tune: Parameters")

    st.subheader("Cohort")
    # voeg toe onder "Cohort":
    geslacht = st.radio("Geslacht", ["Man", "Vrouw"], horizontal=True)

    n_start = st.number_input("Aantal personen", min_value=10, max_value=1_000_000,
                               value=1000, step=100)

    st.subheader("Leeftijden")
    start_leeftijd = st.number_input("Startleeftijd", min_value=16, max_value=70, value=25, step=1)
    pensioenleeftijd = st.number_input("Pensioenleeftijd",
                                        min_value=start_leeftijd + 1, max_value=85, value=67, step=1)
    max_leeftijd = st.number_input("Eindleeftijd (horizon)",
                                    min_value=pensioenleeftijd + 1, max_value=120, value=110, step=1)

    st.subheader("Financieel")
    inleg_per_jaar = st.number_input("Inleg per persoon per jaar (€)",
                                      min_value=0, max_value=500_000,
                                      value=14 * 12, step=100,
                                      help="Stijgt jaarlijks mee met inflatie.")

    st.subheader("Economische aannames")
    rendement_pct = st.slider("Beleggingsrendement per jaar (%)",
                               min_value=0.0, max_value=15.0, value=5.75, step=0.1, format="%.1f%%")
    inflatie_pct = st.slider("Inflatie per jaar (%)",
                              min_value=0.0, max_value=10.0, value=2.25, step=0.1, format="%.1f%%")

    beleggingsresultaat = 1 + rendement_pct / 100
    inflatie = 1 + inflatie_pct / 100
    reeel = rendement_pct - inflatie_pct
    kleur = "green" if reeel >= 0 else "red"
    st.markdown(f"Reëel rendement: :{kleur}[**{reeel:+.1f}%** per jaar]")

BASE_URL = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/"


url = BASE_URL + ("AG2024DefinitiefGevalideerd_male.csv" if geslacht == "Man"
                  else "AG2024DefinitiefGevalideerd_female.csv")
mortality = load_mortality(url)

# ── Berekening ────────────────────────────────────────────────────────────────

with st.spinner("Berekenen…"):
    optimale_uitkering = solve_uitkering(
        inleg_per_jaar, beleggingsresultaat, inflatie,
        start_leeftijd, pensioenleeftijd, max_leeftijd, mortality, n_start,
    )
    records = simulate_cohort(
        inleg_per_jaar, beleggingsresultaat, inflatie, optimale_uitkering,
        start_leeftijd, pensioenleeftijd, max_leeftijd, mortality, n_start,
    )

df = pd.DataFrame(records)

idx_pensioen = pensioenleeftijd - start_leeftijd - 1
pot_bij_pensioen = df.iloc[idx_pensioen]["pensioenpot"]
levenden_bij_pensioen = df.iloc[idx_pensioen]["levenden_eind"]
levenden_eind = df.iloc[-1]["levenden_eind"]
uitkering_laatste = optimale_uitkering * (inflatie ** (max_leeftijd - pensioenleeftijd - 1))
totaal_ingelegd = df["inleg_totaal"].sum()
totaal_uitgekeerd = df["uitkering_totaal"].sum()

# ── KPI's ─────────────────────────────────────────────────────────────────────

st.subheader("Uitkomsten")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(":material/euro: Uitkering jaar 1 (p.p.)", f"€ {optimale_uitkering:,.0f}/jaar")
    st.metric(":material/calendar_month: Per maand", f"€ {optimale_uitkering / 12:,.0f}/maand")

with col2:
    st.metric(":material/account_balance: Pot bij pensioen",
              f"€ {pot_bij_pensioen / 1_000_000:,.2f}M",
              help=f"Collectieve pot op leeftijd {pensioenleeftijd}.")
    st.metric(":material/group: Levenden bij pensioen",
              f"{levenden_bij_pensioen:,.1f} / {n_start:,}")

with col3:
    st.metric(":material/trending_up: Uitkering laatste jaar (p.p.)",
              f"€ {uitkering_laatste:,.0f}/jaar",
              help=f"Per persoon op leeftijd {max_leeftijd - 1}.")
    st.metric(":material/group: Levenden op eindleeftijd", f"{levenden_eind:,.1f}")

with col4:
    st.metric(":material/payments: Totaal ingelegd", f"€ {totaal_ingelegd / 1_000_000:,.2f}M")
    st.metric(":material/redeem: Totaal uitgekeerd", f"€ {totaal_uitgekeerd / 1_000_000:,.2f}M")

# ── Grafieken ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    ":material/account_balance: Pensioenpot",
    ":material/group: Cohort overleving",
    ":material/euro: Uitkering per persoon",
])

with tab1:
    st.plotly_chart(grafiek_pot(df, pensioenleeftijd), use_container_width=True)
with tab2:
    st.plotly_chart(grafiek_levenden(df, pensioenleeftijd), use_container_width=True)
with tab3:
    st.plotly_chart(grafiek_uitkering_pp(df), use_container_width=True)

# ── Detailtabel ───────────────────────────────────────────────────────────────

with st.expander(":material/table_chart: Detailtabel per jaar", expanded=False):
    toon = df.copy()
    for col in ["levenden_begin", "overlijdend", "levenden_eind"]:
        toon[col] = toon[col].round(1)
    for col in ["inleg_pp", "inleg_totaal", "uitkering_pp", "uitkering_totaal", "pensioenpot"]:
        toon[col] = toon[col].round(2)

    st.dataframe(
        toon,
        use_container_width=True,
        hide_index=True,
        column_config={
            "jaar": st.column_config.NumberColumn("Jaar", format="%d"),
            "leeftijd": st.column_config.NumberColumn("Leeftijd", format="%d"),
            "fase": st.column_config.TextColumn("Fase"),
            "overlijdenskans": st.column_config.NumberColumn("Sterftekans q", format="%.6f"),
            "levenden_begin": st.column_config.NumberColumn("Levenden begin", format="%.1f"),
            "overlijdend": st.column_config.NumberColumn("Overlijdend", format="%.1f"),
            "levenden_eind": st.column_config.NumberColumn("Levenden eind", format="%.1f"),
            "inleg_pp": st.column_config.NumberColumn("Inleg p.p. (€)", format="€ %,.2f"),
            "inleg_totaal": st.column_config.NumberColumn("Inleg totaal (€)", format="€ %,.0f"),
            "uitkering_pp": st.column_config.NumberColumn("Uitkering p.p. (€)", format="€ %,.2f"),
            "uitkering_totaal": st.column_config.NumberColumn("Uitkering totaal (€)", format="€ %,.0f"),
            "pensioenpot": st.column_config.NumberColumn("Pensioenpot (€)", format="€ %,.0f"),
        },
    )
    csv_data = toon.to_csv(index=False, sep=";", decimal=",").encode("utf-8")
    st.download_button(":material/download: Download als CSV", data=csv_data,
                       file_name="pensioen_cohort.csv", mime="text/csv")

# ── Methodiek ─────────────────────────────────────────────────────────────────

with st.expander(":material/info: Methodiek", expanded=False):
    n_opbouw = pensioenleeftijd - start_leeftijd
    n_uitkering = max_leeftijd - pensioenleeftijd
    st.markdown(f"""
**Cohortmodel** — {n_start:,} personen starten op leeftijd {start_leeftijd}.

**Opbouwfase** ({n_opbouw} jaar)
- Inleg totaal = inleg_pp × inflatie^t × levenden_begin
- `pot = pot × {beleggingsresultaat:.3f} + inleg_totaal`
- Aandeel van overledenen blijft in de collectieve pot

**Uitkeringsfase** ({n_uitkering} jaar)
- Uitkering alleen aan wie het jaar **overleeft**
- Uitkering p.p. stijgt met inflatie ({inflatie_pct:.1f}%/jaar)
- `pot = pot × {beleggingsresultaat:.3f} − uitkering_pp × levenden_eind`

**Doelstelling**

`uitkering_jaar1` opgelost met Brent's methode zodat de collectieve pot op leeftijd {max_leeftijd} = **€0**.

**Sterftetafel**

AG2024 Definitief Gevalideerd (mannen). Diagonale lezing: jaar t gebruikt
leeftijd {start_leeftijd}+t in kalenderjaar 2024+t.
""")