"""
Huis Kopen vs. Huren / Vermogensopbouw Model
Vergelijkt kosten van huizenbezit met waardestijging over 30 jaar.

Usage:
    streamlit run huis_model.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Huis Kopen Model", page_icon="🏠", layout="wide")


def main() -> None:
    """Hoofdfunctie voor het huis kopen model."""

    st.title("🏠 Huis Kopen — Kosten vs. Waardestijging")
    st.caption("Model gebaseerd op Nederlandse woningmarkt · parameters bijgewerkt april 2026")

    # ─── Sidebar: parameters ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Parameters")

        st.subheader("🏡 Woning")
        aankoopprijs = st.number_input(
            "Aankoopprijs (€)", min_value=100_000, max_value=2_000_000,
            value=450_000, step=10_000, format="%d",
            help="NL gemiddelde 2025: ~€480k (CBS). €450k als neutrale default."
        )
        eigenwoningforfait_pct = st.slider(
            "Eigenwoningforfait (%)", 0.1, 1.0, 0.35, 0.05,
            help="2026: 0,35% voor WOZ €75k–€1.350k (Belastingdienst)"
        )

        st.subheader("💰 Hypotheek")
        eigen_inbreng_pct = st.slider(
            "Eigen inbreng (%)", 0, 30, 10,
            help="Percentage van aankoopprijs als eigen inbreng"
        )
        rente_pct = st.slider(
            "Hypotheekrente (%)", 1.0, 8.0, 4.0, 0.1,
            help="Laagste 10j vast april 2026: ~3,81% (Independer). 4,0% als neutrale default."
        )
        looptijd_jaar = st.slider("Looptijd hypotheek (jaar)", 10, 30, 30)
        hypotheek_aftrek_pct = st.slider(
            "Belastingvoordeel renteaftrek (%)", 0, 52, 37,
            help="Max. aftrek 2026: 37,56%. Geldt alleen bij annuïtaire/lineaire hypotheek."
        )
        gebruik_nhg = st.checkbox(
            "NHG (Nationale Hypotheek Garantie)",
            value=True,
            help="NHG-grens 2026: €470.000. Borgtochtprovisie: 0,4% van hypotheekbedrag."
        )

        st.subheader("🏗️ Kosten")
        gebruik_aankoopmakelaar = st.checkbox(
            "Aankoopmakelaar inschakelen",
            value=False,
            help="Niet verplicht. Courtage ~1,2% van koopprijs. Veel kopers doen het zonder."
        )
        onderhoud_pct = st.slider(
            "Jaarlijks onderhoud (% WOZ)", 0.5, 2.0, 1.0, 0.1,
            help="Vuistregel: 1% per jaar. Oudere woningen zitten eerder op 1,5–2%."
        )
        verzekering_jaar = st.number_input(
            "Opstal + inboedelverzekering (€/jr)", 200, 3000, 700, 50,
            help="NL gemiddelde gecombineerd pakket 2025: ~€630/jr (Independer). €700 als voorzichtige schatting."
        )
        vve_maand = st.number_input(
            "VvE bijdrage (€/maand, 0 = geen)", 0, 1000, 0, 25
        )

        st.subheader("📈 Waardestijging")
        waardestijging_pct = st.slider(
            "Jaarlijkse waardestijging (%)", 0.0, 10.0, 3.5, 0.1,
            help="Neutraal: 3,0–3,5% · Optimistisch: 4,5% · Conservatief: 1,5%. "
                 "Recent (jan 2026): +5,4% j-o-j, maar 30-jaars gemiddelde incl. crises is lager."
        )
        inflatie_pct = st.slider(
            "Inflatie (%)", 0.0, 6.0, 2.5, 0.1,
            help="Voor reële berekening. ECB-target: 2%."
        )

        st.subheader("🏦 Alternatief: Huren + Beleggen")
        # Suggestie: price-to-rent ratio ~25 is NL gemiddeld voor vrije sector
        huur_suggestie = int(round(aankoopprijs / 25 / 12 / 50) * 50)
        huur_maand = st.number_input(
            "Maandelijkse huur (€)", min_value=500, max_value=5000,
            value=huur_suggestie, step=50,
            help=f"Suggestie op basis van aankoopprijs: €{huur_suggestie:,.0f}/maand "
                 f"(price-to-rent ratio 25, gangbaar NL vrije sector). "
                 f"NL gemiddeld vrije sector 2025: ~€1.400–€1.600/maand."
        )
        huurstijging_pct = st.slider(
            "Jaarlijkse huurstijging (%)", 0.0, 8.0, 3.0, 0.1,
            help="Vrije sector: contractueel vaak CPI + opslag. Historisch ~3–4%/jr."
        )
        spaar_rente_pct = st.slider(
            "Rendement beleggen (%)", 0.0, 10.0, 4.0, 0.1,
            help="Wat de huurder verdient op het vrijgekomen startkapitaal + koopkosten. "
                 "Spaarrekening: ~1,5% · AEX incl. div.: ~7% · Wereld ETF: ~7–8%"
        )

        st.subheader("📅 Tijdshorizon")
        jaren = st.slider("Aantal jaar", 5, 30, 30)

    # ─── Berekeningen ──────────────────────────────────────────────────────
    eigen_inbreng = aankoopprijs * eigen_inbreng_pct / 100
    hypotheek = aankoopprijs - eigen_inbreng

    # Aankoopkosten (eenmalig)
    overdrachtsbelasting = aankoopprijs * 0.02   # 2% eigenaar-bewoner (0% starter ≤35jr + ≤€555k)
    notaris_transport = 1_200   # transportakte (leveringsakte)
    notaris_hypotheek = 900     # hypotheekakte
    taxatie = 600
    makelaarscourtage = aankoopprijs * 0.012 if gebruik_aankoopmakelaar else 0
    # NHG 2026: grens €470.000, provisie 0,4% van hypotheekbedrag (bron: nhg.nl)
    nhg_grens = 470_000
    nhg_provisie = 0.004
    nhg_kosten = hypotheek * nhg_provisie if (gebruik_nhg and hypotheek <= nhg_grens) else 0
    aankoopkosten_totaal = (
        overdrachtsbelasting + notaris_transport + notaris_hypotheek +
        taxatie + makelaarscourtage + nhg_kosten
    )

    # Maandelijkse annuïteit
    r = rente_pct / 100 / 12
    n = looptijd_jaar * 12
    if r > 0:
        annuiteit_maand = hypotheek * (r * (1 + r) ** n) / ((1 + r) ** n - 1)
    else:
        annuiteit_maand = hypotheek / n

    # OZB (gemeentelijk, gemiddeld NL)
    ozb_pct_jaar = 0.1   # ~0,1% van WOZ-waarde (eigenaar)

    # ─── Jaarlijkse simulatie ──────────────────────────────────────────────
    rows: list[dict] = []
    woningwaarde = aankoopprijs
    restschuld = hypotheek
    cumulatief_kosten = aankoopkosten_totaal + eigen_inbreng
    cumulatief_kosten_excl_eigen = aankoopkosten_totaal
    totaal_rente_betaald = 0.0
    totaal_aflossing = 0.0

    # Spaarpot: startkapitaal = eigen inbreng + aankoopkosten
    spaarpot = eigen_inbreng + aankoopkosten_totaal
    huur_jaar_huidig = huur_maand * 12
    cumulatief_bespaarde_huur = 0.0
    cumulatief_woonkosten_excl_afl = 0.0

    for jaar in range(1, jaren + 1):
        # Woningwaarde groeit
        woningwaarde_start = woningwaarde
        woningwaarde = woningwaarde_start * (1 + waardestijging_pct / 100)

        # Annuïteit splitsen in rente & aflossing (begin jaar)
        rente_jaar = 0.0
        aflossing_jaar = 0.0
        restschuld_start = restschuld

        for _ in range(12):
            if restschuld > 0:
                rente_maand = restschuld * rente_pct / 100 / 12
                aflossing_maand = min(annuiteit_maand - rente_maand, restschuld)
                rente_jaar += rente_maand
                aflossing_jaar += aflossing_maand
                restschuld = max(0, restschuld - aflossing_maand)

        totaal_rente_betaald += rente_jaar
        totaal_aflossing += aflossing_jaar

        # Belastingvoordeel renteaftrek (afgebouwd na 2023, maar parameter)
        renteaftrek_voordeel = rente_jaar * hypotheek_aftrek_pct / 100
        netto_rente = rente_jaar - renteaftrek_voordeel

        # Eigenwoningforfait belasting (bijtelling)
        ewf_belasting = (woningwaarde_start * eigenwoningforfait_pct / 100) * 0.37

        # OZB
        ozb = woningwaarde_start * ozb_pct_jaar / 100

        # Onderhoud
        onderhoud = woningwaarde_start * onderhoud_pct / 100

        # Verzekering
        verzekering = verzekering_jaar

        # VvE
        vve = vve_maand * 12

        # Totale jaarkosten (netto, na renteaftrek)
        jaarkosten = (
            netto_rente + aflossing_jaar + ewf_belasting +
            ozb + onderhoud + verzekering + vve
        )
        jaarkosten_excl_aflossing = (
            netto_rente + ewf_belasting + ozb + onderhoud + verzekering + vve
        )

        cumulatief_kosten += jaarkosten
        cumulatief_kosten_excl_eigen += jaarkosten

        # Vermogen = waarde - restschuld
        overwaarde = woningwaarde - restschuld
        netto_vermogen = overwaarde - aankoopkosten_totaal

        # Reële waarde (gecorrigeerd voor inflatie)
        reele_waarde = woningwaarde / ((1 + inflatie_pct / 100) ** jaar)

        # ── Spaarpot: startkapitaal + alle koopkosten belegd ──
        spaarpot = spaarpot * (1 + spaar_rente_pct / 100) + jaarkosten

        # ── Koper totaalvermogen: overwaarde + cumulatief niet-betaalde huur ──
        cumulatief_bespaarde_huur += huur_jaar_huidig
        koper_totaalvermogen = overwaarde + cumulatief_bespaarde_huur

        # ── Netto rendement: overwaarde − verkoopkosten − aankoopkosten − alle woonkosten ──
        # Verkoopkosten bij verkoop op dit moment: makelaar ~1% + notaris ~€500
        verkoopkosten = woningwaarde * 0.01 + 500
        cumulatief_woonkosten_excl_afl += jaarkosten_excl_aflossing
        netto_rendement = overwaarde - verkoopkosten - aankoopkosten_totaal - cumulatief_woonkosten_excl_afl

        # ── Netto rendement incl. woonvoordeel: netto rendement + cumulatief niet-betaalde huur ──
        netto_rendement_incl_woonvoordeel = netto_rendement + cumulatief_bespaarde_huur

        # Huur stijgt volgend jaar
        huur_jaar_huidig = huur_jaar_huidig * (1 + huurstijging_pct / 100)

        rows.append({
            "Jaar": jaar,
            "Woningwaarde": round(woningwaarde),
            "Reële waarde": round(reele_waarde),
            "Restschuld": round(restschuld),
            "Overwaarde": round(overwaarde),
            "Rente (netto)": round(netto_rente),
            "Aflossing": round(aflossing_jaar),
            "OZB": round(ozb),
            "Onderhoud": round(onderhoud),
            "Verzekering": round(verzekering),
            "VvE": round(vve),
            "EWF belasting": round(ewf_belasting),
            "Totale jaarkosten": round(jaarkosten),
            "Kosten excl. aflossing": round(jaarkosten_excl_aflossing),
            "Cumulatieve kosten": round(cumulatief_kosten),
            "Netto vermogen": round(netto_vermogen),
            "Spaarpot": round(spaarpot),
            "Koper totaalvermogen": round(koper_totaalvermogen),
            "Cumulatief woonvoordeel": round(cumulatief_bespaarde_huur),
            "Huur dit jaar": round(huur_jaar_huidig / (1 + huurstijging_pct / 100)),
            "Netto rendement": round(netto_rendement),
            "Netto rendement incl. woonvoordeel": round(netto_rendement_incl_woonvoordeel),
        })

    df = pd.DataFrame(rows)

    # ─── KPI's bovenaan ────────────────────────────────────────────────────
    eindwaarde = df["Woningwaarde"].iloc[-1]
    eindoverwaarde = df["Overwaarde"].iloc[-1]
    totale_kosten = df["Totale jaarkosten"].sum()
    totale_kosten_excl_afl = df["Kosten excl. aflossing"].sum()
    winst = eindoverwaarde - aankoopkosten_totaal - totale_kosten_excl_afl
    roi = (eindoverwaarde / (eigen_inbreng + aankoopkosten_totaal) - 1) * 100

    st.subheader("📊 Samenvatting na {} jaar".format(jaren))

    with st.container(horizontal=True):
        st.metric(
            "Woningwaarde", f"€ {eindwaarde:,.0f}",
            f"+{eindwaarde - aankoopprijs:,.0f}",
            border=True,
            chart_data=df["Woningwaarde"].tolist(),
            chart_type="line"
        )
        st.metric(
            "Overwaarde", f"€ {eindoverwaarde:,.0f}",
            border=True,
            chart_data=df["Overwaarde"].tolist(),
            chart_type="line"
        )
        st.metric(
            "Restschuld", f"€ {df['Restschuld'].iloc[-1]:,.0f}",
            f"-{hypotheek - df['Restschuld'].iloc[-1]:,.0f} afgelost",
            border=True,
            chart_data=df["Restschuld"].tolist(),
            chart_type="line"
        )
        st.metric(
            "ROI op eigen inbreng", f"{roi:.1f}%",
            border=True
        )
        st.metric(
            "Spaarpot / beleggen 🏦",
            f"€ {df['Spaarpot'].iloc[-1]:,.0f}",
            help="Startkapitaal (eigen inbreng + aankoopkosten) + alle jaarlijkse koopkosten belegd",
            border=True,
            chart_data=df["Spaarpot"].tolist(),
            chart_type="line"
        )

    with st.container(horizontal=True):
        st.metric(
            "Aankoopkosten (eenmalig)", f"€ {aankoopkosten_totaal:,.0f}",
            border=True
        )
        st.metric(
            "Totale woonkosten (excl. aflossing)",
            f"€ {totale_kosten_excl_afl:,.0f}",
            help="Rente, OZB, onderhoud, verzekering",
            border=True,
            # chart_data=df["Kosten excl. aflossing"].tolist(),
            # chart_type="bar"
        )
        st.metric(
            "Totale rente betaald",
            f"€ {totaal_rente_betaald:,.0f}",
            border=True
        )
        # st.metric(
        #     "Netto rendement*", f"€ {winst:,.0f}",
        #     help="Overwaarde minus aankoopkosten minus alle woonkosten excl. aflossing",
        #     border=True
        # )
        st.metric(
            "Netto rendement incl. verkoopkosten",
            f"€ {df['Netto rendement'].iloc[-1]:,.0f}",
            help="Overwaarde − verkoopkosten (1% + €500) − aankoopkosten − alle woonkosten excl. aflossing",
            border=True,
            chart_data=df["Netto rendement"].tolist(),
            chart_type="line"
        )

    st.caption(
        "\\* Netto rendement = Overwaarde − aankoopkosten − alle kosten excl. aflossing. "
        "Aflossing telt niet als 'verloren geld' — het is vermogensopbouw."
    )

    # ─── Eerlijke vergelijking: koper vs huurder ──────────────────────────
    # netto_rendement_eind = overwaarde − verkoopkosten − aankoopkosten − cumulatieve woonkosten
    netto_rendement_eind = df["Netto rendement"].iloc[-1]
    netto_rendement_woon_eind = df["Netto rendement incl. woonvoordeel"].iloc[-1]
    huurder_eind = df["Spaarpot"].iloc[-1]
    woonvoordeel_eind = df["Cumulatief woonvoordeel"].iloc[-1]
    verschil_puur = netto_rendement_eind - huurder_eind
    verschil_incl_woon = netto_rendement_woon_eind - huurder_eind

    if verschil_puur > 0:
        conclusie_puur = f"✅ Kopen wint: **€ {verschil_puur:,.0f} meer** netto rendement dan spaarpot na {jaren} jaar."
    else:
        conclusie_puur = f"🏦 Sparen/beleggen wint: **€ {abs(verschil_puur):,.0f} meer** dan netto rendement na {jaren} jaar."

    if verschil_incl_woon > 0:
        conclusie_eerlijk = f"✅ Kopen wint: **€ {verschil_incl_woon:,.0f} meer** als woonvoordeel wordt meegeteld."
    else:
        conclusie_eerlijk = f"🏦 Sparen/beleggen wint: **€ {abs(verschil_incl_woon):,.0f} meer**, ook met woonvoordeel."

    with st.container(border=True):
        st.markdown("### ⚖️ Kopen vs. Sparen/Beleggen")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(
                "🏠 Netto rendement kopen",
                f"€ {netto_rendement_eind:,.0f}",
                help=f"Overwaarde €{eindoverwaarde:,.0f} − verkoopkosten − aankoopkosten − alle woonkosten excl. aflossing"
            )
        with col_b:
            st.metric(f"🏦 Spaarpot ({spaar_rente_pct}%)", f"€ {huurder_eind:,.0f}")
        with col_c:
            st.metric("Verschil", f"€ {verschil_puur:,.0f}")
        st.markdown(conclusie_puur)

        if woonvoordeel_eind != 0:
            st.divider()
            st.caption("Inclusief woonvoordeel: netto rendement + alle huur die de koper niet heeft betaald (gecumuleerd):")
            col_d, col_e, col_f = st.columns(3)
            with col_d:
                st.metric(
                    "🏠 Netto rendement incl. woonvoordeel",
                    f"€ {netto_rendement_woon_eind:,.0f}",
                    help=f"Netto rendement €{netto_rendement_eind:,.0f} + cumulatief niet-betaalde huur €{woonvoordeel_eind:,.0f}"
                )
            with col_e:
                st.metric(f"🏦 Spaarpot ({spaar_rente_pct}%)", f"€ {huurder_eind:,.0f}")
            with col_f:
                st.metric("Verschil incl. woonvoordeel", f"€ {verschil_incl_woon:,.0f}")
            st.markdown(conclusie_eerlijk)
        st.caption(
            f"Huurprijs jaar 1: €{huur_maand:,.0f}/maand · "
            f"Huurprijs jaar {jaren}: €{df['Huur dit jaar'].iloc[-1]:,.0f}/jr "
            f"(€{df['Huur dit jaar'].iloc[-1]/12:,.0f}/maand)"
        )

    # ─── Grafiek 1: Vermogensvergelijking ─────────────────────────────────
    st.subheader("📈 Vermogensvergelijking: Kopen vs. Sparen")

    toon_woonvoordeel = st.checkbox(
        "Toon ook koper totaalvermogen (incl. woonvoordeel)",
        value=True,
        help="Voegt een derde lijn toe: overwaarde + cumulatief woonvoordeel "
             "(huurprijs − woonkosten excl. aflossing, per jaar opgeteld)."
    )

    fig1 = go.Figure()

    # Nullijn als referentie
    fig1.add_shape(
        type="line", x0=1, x1=jaren, y0=0, y1=0,
        line=dict(color="gray", width=1, dash="dot")
    )

    fig1.add_trace(go.Scatter(
        x=df["Jaar"], y=df["Woningwaarde"], visible="legendonly",
        name="Woningwaarde", line=dict(color="#2ecc71", width=2, dash="dot"),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.05)"
    ))
    fig1.add_trace(go.Scatter(
        x=df["Jaar"], y=df["Overwaarde"], visible="legendonly",
        name="🏠 Overwaarde (kopen)",
        line=dict(color="#3498db", width=3),
    ))
    fig1.add_trace(go.Scatter(
        x=df["Jaar"], y=df["Spaarpot"],
        name=f"🏦 Spaarpot / beleggen ({spaar_rente_pct}%)",
        line=dict(color="#e74c3c", width=3, ),
    ))
    if toon_woonvoordeel:
        fig1.add_trace(go.Scatter(
            x=df["Jaar"], y=df["Koper totaalvermogen"], visible="legendonly",
            name="🏠 Koper totaalvermogen (overwaarde + niet-betaalde huur)",
            line=dict(color="#1abc9c", width=2, dash="dash"),
        ))

    # Netto rendement lijnen
    fig1.add_trace(go.Scatter(
        x=df["Jaar"], y=df["Netto rendement"], visible="legendonly",
        name="💰 Netto rendement (na alle kosten + verkoopkosten)",
        line=dict(color="#f39c12", width=2, dash="dashdot"),
    ))
    fig1.add_trace(go.Scatter(
        x=df["Jaar"], y=df["Netto rendement incl. woonvoordeel"],
        name="💰 Netto rendement incl. woonvoordeel",
        line=dict(color="#9b59b6", width=2, dash="dashdot"),
    ))

    fig1.update_layout(
        xaxis_title="Jaar",
        yaxis_title="Bedrag (€)",
        yaxis_tickprefix="€ ",
        yaxis_tickformat=",.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig1, width="stretch")

    # ─── Grafiek 1b: Netto rendement ──────────────────────────────────────
    # st.subheader("💰 Netto rendement bij verkoop op jaar X")
    # st.caption(
    #     "Overwaarde − verkoopkosten (makelaar ~1% + notaris €500) "
    #     "− aankoopkosten − cumulatieve woonkosten excl. aflossing. "
    #     "Negatief in de beginjaren is normaal: dan wegen de eenmalige kosten nog zwaar."
    # )

    # fig1b = go.Figure()
    # fig1b.add_shape(type="line", x0=0, x1=jaren, y0=0, y1=0,
    #                 line=dict(color="gray", width=1, dash="dot"))
    # fig1b.add_trace(go.Scatter(
    #     x=df["Jaar"], y=df["Netto rendement"],
    #     name="Netto rendement",
    #     line=dict(color="#f39c12", width=3),
    #     fill="tozeroy",
    #     fillcolor="rgba(243,156,18,0.08)"
    # ))
    # fig1b.add_trace(go.Scatter(
    #     x=df["Jaar"], y=df["Netto rendement incl. woonvoordeel"],
    #     name="Netto rendement incl. woonvoordeel",
    #     line=dict(color="#9b59b6", width=2, dash="dash"),
    #     fill="tozeroy",
    #     fillcolor="rgba(155,89,182,0.05)"
    # ))
    # fig1b.add_trace(go.Scatter(
    #     x=df["Jaar"], y=df["Spaarpot"],
    #     name=f"Spaarpot ({spaar_rente_pct}%)",
    #     line=dict(color="#e74c3c", width=2, dash="dash"),
    # ))
    # fig1b.update_layout(
    #     xaxis_title="Jaar",
    #     yaxis_title="Bedrag (€)",
    #     yaxis_tickprefix="€ ",
    #     yaxis_tickformat=",.0f",
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02),
    #     height=400,
    #     margin=dict(l=10, r=10, t=30, b=10),
    # )
    # st.plotly_chart(fig1b, width="stretch")

    # ─── Grafiek 2: Kostenverdeling gestapeld ─────────────────────────────
    st.subheader("💸 Jaarlijkse kostenverdeling")

    fig2 = go.Figure()
    kostenposten = {
        "Rente (netto)": "#e74c3c",
        "OZB": "#e67e22",
        "Onderhoud": "#f39c12",
        "Verzekering": "#f1c40f",
        "VvE": "#d4ac0d",
        "EWF belasting": "#a93226",
    }
    for post, kleur in kostenposten.items():
        if df[post].sum() > 0:
            fig2.add_trace(go.Bar(
                x=df["Jaar"], y=df[post],
                name=post, marker_color=kleur
            ))

    fig2.update_layout(
        barmode="stack",
        xaxis_title="Jaar",
        yaxis_title="Kosten (€)",
        yaxis_tickprefix="€ ",
        yaxis_tickformat=",.0f",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig2, width="stretch")

    # ─── Grafiek 3: Taartdiagram totale kosten ────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🥧 Verdeling totale kosten")
        labels = ["Rente (netto)", "OZB", "Onderhoud", "Verzekering", "VvE", "EWF belasting", "Aankoopkosten"]
        values = [
            df["Rente (netto)"].sum(),
            df["OZB"].sum(),
            df["Onderhoud"].sum(),
            df["Verzekering"].sum(),
            df["VvE"].sum(),
            df["EWF belasting"].sum(),
            aankoopkosten_totaal,
        ]
        kleuren = ["#e74c3c", "#e67e22", "#f39c12", "#f1c40f", "#d4ac0d", "#a93226", "#8e44ad"]
        fig3 = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.45,
            marker_colors=kleuren,
            textinfo="label+percent",
        ))
        fig3.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=10),
                           showlegend=False)
        st.plotly_chart(fig3, width="stretch")

    with col2:
        st.subheader("📋 Aankoopkosten detail")
        aankoop_df = pd.DataFrame({
            "Post": [
                "Overdrachtsbelasting (2%)",
                "Notaris — transportakte",
                "Notaris — hypotheekakte",
                "Taxatie",
                "Aankoopmakelaar (~1,2%)" if gebruik_aankoopmakelaar else "Aankoopmakelaar (niet gekozen)",
                f"NHG provisie (0,4%, grens €470k)" if nhg_kosten > 0 else "NHG (niet van toepassing)",
                "**Totaal**",
            ],
            "Bedrag": [
                f"€ {overdrachtsbelasting:,.0f}",
                f"€ {notaris_transport:,.0f}",
                f"€ {notaris_hypotheek:,.0f}",
                f"€ {taxatie:,.0f}",
                f"€ {makelaarscourtage:,.0f}",
                f"€ {nhg_kosten:,.0f}",
                f"**€ {aankoopkosten_totaal:,.0f}**",
            ]
        })
        st.dataframe(aankoop_df, hide_index=True, width="stretch")

        st.subheader("📊 Hypotheek")
        hyp_df = pd.DataFrame({
            "Post": [
                "Hypotheekbedrag",
                "Eigen inbreng",
                "Maandlast (annuïteit)",
                "Looptijd",
            ],
            "Waarde": [
                f"€ {hypotheek:,.0f}",
                f"€ {eigen_inbreng:,.0f} ({eigen_inbreng_pct}%)",
                f"€ {annuiteit_maand:,.0f}",
                f"{looptijd_jaar} jaar",
            ]
        })
        st.dataframe(hyp_df, hide_index=True, width="stretch")

    # ─── Detailtabel ───────────────────────────────────────────────────────
    with st.expander("📄 Jaarlijkse detailtabel"):


        #     "Jaar": jaar,
        #     "Woningwaarde": round(woningwaarde),
        #     "Reële waarde": round(reele_waarde),
        #     "Restschuld": round(restschuld),
        #     "Overwaarde": round(overwaarde),
        #     "Rente (netto)": round(netto_rente),
        #     "Aflossing": round(aflossing_jaar),
        #     "OZB": round(ozb),
        #     "Onderhoud": round(onderhoud),
        #     "Verzekering": round(verzekering),
        #     "VvE": round(vve),
        #     "EWF belasting": round(ewf_belasting),
        #     "Totale jaarkosten": round(jaarkosten),
        #     "Kosten excl. aflossing": round(jaarkosten_excl_aflossing),
        #     "Cumulatieve kosten": round(cumulatief_kosten),
        #     "Netto vermogen": round(netto_vermogen),
        #     "Spaarpot": round(spaarpot),
        #     "Koper totaalvermogen": round(koper_totaalvermogen),
        #     "Cumulatief woonvoordeel": round(cumulatief_bespaarde_huur),
        #     "Huur dit jaar": round(huur_jaar_huidig / (1 + huurstijging_pct / 100)),
        #     "Netto rendement": round(netto_rendement),
        #     "Netto rendement incl. woonvoordeel": round(netto_rendement_incl_woonvoordeel),
        # })

        weergave_df = df[[
            "Jaar", "Woningwaarde", "Restschuld", "Overwaarde","Aflossing",
            "Rente (netto)", "OZB", "Onderhoud", "Verzekering",
            "EWF belasting", "Kosten excl. aflossing", "Totale jaarkosten",
            "Netto rendement", "Netto rendement incl. woonvoordeel","Spaarpot"
        ]].copy()

        # Formateer als euro
        euro_cols = [c for c in weergave_df.columns if c != "Jaar"]
        for col in euro_cols:
            weergave_df[col] = weergave_df[col].apply(lambda x: f"€ {x:,.0f}")

        st.dataframe(weergave_df, hide_index=True, width="stretch")

    # ─── Historische context ───────────────────────────────────────────────
    with st.expander("📚 Historische context NL woningmarkt (1994–2026)"):
        st.markdown("""
        | Periode | Gem. jaarlijkse stijging | Context |
        |---------|--------------------------|---------|
        | 1994–2008 | ~+7% | Lage rente, liberalisering hypotheekmarkt |
        | 2008–2013 | ~−3% | Financiële crisis, prijsdaling ~20% |
        | 2013–2022 | ~+8% | Historisch lage rente, woningtekort |
        | 2022–2023 | ~−5% | Rentestijging ECB |
        | 2023–2026 | ~+5–6% | Herstel, aanhoudend woningtekort |
        | **Gemiddeld 30 jaar** | **~4,0–4,5% nominaal** | Incl. crisisperiodes |

        **Scenarios voor waardestijging:**
        - 🟢 Optimistisch: 4,5% (historisch gemiddelde excl. crises)
        - 🟡 Neutraal: 3,5% ← **default in dit model**
        - 🔴 Conservatief: 1,5–2,0% (incl. crisisperiodes, reëel)

        **Aannames en bronnen (april 2026):**
        - OZB: ~0,1% WOZ (varieert gemeente: 0,06–0,19%, bron: COELO Atlas Lokale Lasten 2025)
        - Onderhoud: 1% per jaar vuistregel (oudere woning: 1,5–2%)
        - Eigenwoningforfait: 0,35% WOZ voor woningen €75k–€1.350k (Belastingdienst 2026)
        - Renteaftrek max. 37,56% (2026). Wet Hillen loopt af: zonder schuld minder EWF-voordeel.
        - Overdrachtsbelasting: 2% eigenaar-bewoner. Starter ≤35jr + woning ≤€555k = **0%**
        - NHG-grens 2026: **€470.000**, provisie **0,4%** (nhg.nl)
        - Gemiddelde transactieprijs bestaande woning NL 2025: ~€480.000 (CBS)
        - Hypotheekrente 10jr vast april 2026: laagste ~3,81% (Independer)
        - Verzekering opstal+inboedel: gemiddeld ~€630/jr in 2025 (Independer)

        ⚠️ **Dit model is een verkenner, geen financieel advies.**
        Raadpleeg een hypotheekadviseur voor persoonlijke beslissingen.
        Huuralternatief is niet opgenomen — voor een volledige koop-vs-huur analyse zijn huurprijs en huurstijging nodig.
        """)

    # ─── Rendement referentie ──────────────────────────────────────────────
    with st.expander("📊 Referentie: historisch rendement sparen & beleggen (voor spaarparameter)"):
        st.markdown("""
        Gebruik deze cijfers als houvast bij het instellen van de **spaarrekening/beleggen** parameter.

        | Beleggingsvorm | Gem. jaarrendement (30 jaar) | Opmerking |
        |---|---|---|
        | 🏦 Spaarrekening NL (variabel gemiddelde) | ~1,5–2,0% | Nominaal; reëel na inflatie vaak negatief |
        | 📈 AEX — koersrendement excl. dividend | ~5,3% | 1994–2024 (Finansjaal/CBS) |
        | 📈 AEX — totaalrendement incl. herbelegde div. | ~7,3% | Dividend ~2–3%/jr herbelegd (Langzaam Rijker) |
        | 🌍 MSCI World ETF (wereldwijd gespreid) | ~8–9% | Historisch gemiddelde lange termijn |
        | 🇺🇸 S&P 500 | ~10% | Gemiddeld 100 jaar nominaal (USD) |

        **Historisch spaarrente verloop NL:**
        - 1994: ~5% · 2000: ~3% · 2008: piek ~5% (spaarrenteoorlog) · 2012–2021: dalend naar ~0%
        - 2019–2022: vrijwel 0%, soms negatief · 2023–2024: herstel naar ~1,5% · 2026: ~1,3%
        - 30-jaars gemiddelde: **~1,5–2,0% nominaal**

        **Kanttekeningen AEX:**
        - In ~10 van de 41 jaar was het rendement negatief (gem. -21% in die jaren)
        - Wie instapte in 1999 stond eind 2019 nog op verlies (koers)
        - AEX heeft slechts 25 bedrijven — weinig spreiding vs. wereldwijde ETF
        - Dividendrendement historisch ~2–3%/jr; herbeleggen maakt groot verschil op lange termijn

        **Conclusie voor dit model:**
        - Spaarrekening realistisch: **1,5%**
        - AEX beleggen (incl. dividend): **7%**
        - Conservatief beleggen (wereldwijd ETF): **6–7%**
        - De default van **4%** zit tussen sparen en beleggen in — kies bewust.
        """)

    # ─── Eerlijkheid van de vergelijking ──────────────────────────────────
    with st.expander("⚖️ Is de vergelijking eigenlijk wel eerlijk? — Het woongenot-argument"):
        st.markdown("""
        **Nee, de vergelijking huis vs. spaarpot is niet volledig eerlijk — in het voordeel van het huis.**

        De spaarpot-lijn vergelijkt alleen *vermogen*. Maar als je een huis koopt, heb je al die jaren
        ook ergens gewoond. Dat is niet gratis.

        ---

        ### De eerlijke vergelijking is:
        **Huis kopen** vs. **Huren + het verschil beleggen**

        | Bij kopen betaal je | Bij huren betaal je |
        |---|---|
        | Hypotheekannuïteit | Huur |
        | OZB, onderhoud, verzekering | (geen) |
        | Aankoopkosten (eenmalig) | Borg (eenmalig, terugkrijgbaar) |

        Als huren goedkoper is dan de maandlast bij kopen, kan de huurder het verschil beleggen.
        Die beleggingsopbrengst moet je dan optellen bij de spaarpot — dan pas is de vergelijking eerlijk.

        ---

        ### Wat dit model doet:
        De spaarpot-lijn gaat ervan uit dat de kosten op een spaarrekening wordt gezet of belegt met een bepaald rendement.
        Dit wordt vergeleken met de waardestijging van het huis minus de kosten. De niet betaalde huur wordt hierbij opgeteld voor
        een zo eerlijk mogelijke vergelijking.

        ---

        ### Vuistregel:
        > *Kopen is financieel aantrekkelijker naarmate de huur hoger is t.o.v. de hypotheeklasten,
        > de woningwaarde harder stijgt, en je lang op dezelfde plek woont.*

        > *Huren is financieel aantrekkelijker naarmate je flexibeler wilt zijn, de huizenmarkt duur is
        > t.o.v. huurprijzen (hoge 'price-to-rent ratio'), en je het verschil daadwerkelijk belegt.*

        ---

        ### Price-to-rent ratio Nederland (2026)
        In Nederland liggen koopprijzen historisch hoog t.o.v. huurprijzen.
        Een woning van €450.000 die voor €1.500/maand verhuurd zou worden heeft een
        price-to-rent ratio van **25** (450.000 / 18.000). Internationaal geldt >20 als 'duur om te kopen'. """)

    st.info("""⚠️ **Dit model is een verkenner, geen financieel advies.**
        Raadpleeg een hypotheekadviseur voor persoonlijke beslissingen.""")


if __name__ == "__main__":
    main()