"""
uitleg.py — "Should I Be Trading?" — Uitleg van het systeem
Een leesbaar artikel dat alle cijfers, grafieken en beslislogica uitlegt.

Gebruik:
  streamlit run uitleg.py
"""

import streamlit as st
from theme import apply_article_theme
  
try:
    st.set_page_config(
        page_title="Should I Be Trading? — Uitleg",
        page_icon="📖",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
except:
    pass

def uitleg_backtest():
    apply_article_theme()
    st.markdown("<style>.block-container{padding:1rem 1.5rem!important;max-width:100%!important}</style>",
            unsafe_allow_html=True)

    
    # ════════════════════════════════════════════════════════════════════
    # TITEL
    # ════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="bodytext">
    <div class="art-title">Should I Be Trading?</div>
    <div class="art-sub">◈ TECHNISCHE UITLEG — SCORING ENGINE · GRAFIEKEN · BACKTESTRESULTATEN</div>
    <div class="art-lead">
      Dit dashboard beantwoordt één vraag: is het vandaag een goed moment om actief te handelen in de
      Amerikaanse aandelenmarkt? Het systeem berekent dagelijks een score op basis van vijf
      marktfactoren en geeft een concreet advies: <strong>YES</strong>, <strong>CAUTION</strong> of <strong>NO</strong>.
      Dit artikel legt uit hoe die score werkt, wat elke grafiek toont en hoe je de backtestresultaten
      moet lezen.
    </div>
    <hr class="div"/>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # 1. HET IDEE
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 1. Het idee")

    st.markdown("""
    Niet elke dag is een goede dag om te handelen. In rustige, stijgende markten werken technische
    setups beter: breakouts houden stand, stops worden minder snel geraakt en de wind waait mee.
    In chaotische of dalende markten falen dezelfde setups vaker, zelfs als de technische analyse
    klopt.

    Dit systeem probeert die *marktomgeving* te meten — los van welk aandeel je overweegt.
    Het antwoord is niet "koop aandeel X", maar "is dit een moment waarop handelen überhaupt zinvol is?"

    De filosofie is defensief: **het is beter om goede kansen te missen dan om slechte kansen te nemen
    in een vijandige markt.**
    """)

    st.markdown("""
    <div class="callout callout-blue">
      ◈ Het systeem volgt de S&P 500 (via SPY) als proxy voor de brede Amerikaanse markt.
      Alle signalen zijn gebaseerd op publiek beschikbare dagelijkse data via Yahoo Finance.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # 2. DE VIJF FACTOREN
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 2. De vijf factoren en hun gewichten")

    st.markdown("""
    SPY is de ticker van de SPDR S&P 500 ETF Trust — het grootste en meest verhandelde beleggingsfonds 
    ter wereld. SPY volgt de S&P 500 — een index van de 500 grootste beursgenoteerde bedrijven in Amerika.
    Denk aan Apple, Microsoft, Amazon, Google, JPMorgan, Johnson & Johnson, etc. 
    Samen vertegenwoordigen ze roughly 80% van de totale Amerikaanse beurswaarde.
                """)
    st.markdown("""
    De totaalscore — de **Market Quality Score (MQS)** — is een gewogen gemiddelde van vijf
    sub-scores, elk tussen 0 en 100. De gewichten reflecteren hoe sterk elke factor de
    kans op succesvolle trades beïnvloedt:
                """)
    

    st.markdown("""
    <table class="score-table">
    <thead>
      <tr>
        <th>#</th><th>Factor</th><th>Gewicht</th><th>Wat het meet</th><th>Goed teken</th><th>Slecht teken</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>1</td>
        <td class="cat">⚡ Volatiliteit</td>
        <td class="wt">25%</td>
        <td>Marktrumoer en angst</td>
        <td class="pos">VIX &lt; 18, dalend</td>
        <td class="neg">VIX &gt; 28, stijgend</td>
      </tr>
      <tr>
        <td>2</td>
        <td class="cat">📈 Trend</td>
        <td class="wt">20%</td>
        <td>Richting van SPY t.o.v. zijn voortschrijdende gemiddelden</td>
        <td class="pos">SPY &gt; MA20 &gt; MA50 &gt; MA200</td>
        <td class="neg">SPY onder MA200 en MA50</td>
      </tr>
      <tr>
        <td>3</td>
        <td class="cat">🌊 Marktbreedte</td>
        <td class="wt">20%</td>
        <td>Hoeveel sectoren doen mee aan de beweging</td>
        <td class="pos">&gt; 7/11 sectoren positief</td>
        <td class="neg">&lt; 4/11 sectoren positief</td>
      </tr>
      <tr>
        <td>4</td>
        <td class="cat">🚀 Momentum</td>
        <td class="wt">25%</td>
        <td>Kracht en richting van de recente beweging</td>
        <td class="pos">SPY +2% in 5 dagen, brede sectorspreiding</td>
        <td class="neg">SPY &minus;2% in 5 dagen, vlak</td>
      </tr>
      <tr>
        <td>5</td>
        <td class="cat">🏦 Macro</td>
        <td class="wt">10%</td>
        <td>Rente en dollaromgeving</td>
        <td class="pos">10-jaarsrente dalend, Fed dovish</td>
        <td class="neg">Rente &gt; 5% en stijgend</td>
      </tr>
    </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("### Hoe wordt elke sub-score berekend?")

    st.markdown("""
    Elke factor begint op een basiswaarde en krijgt punten toegevoegd of afgetrokken op basis van
    drempelwaarden. De uitkomst wordt altijd begrensd tussen 0 en 100.
    """)

    st.markdown("""
    **⚡ Volatiliteit (25%)**

    De basis is het VIX-niveau — de "angstindex" van de markt, afgeleid van optieprijzen op de S&P 500:

    - VIX < 15 → basiscore **90** (ideaal)
    - VIX 15–18 → **75**
    - VIX 18–22 → **60** (normaal)
    - VIX 22–26 → **42**
    - VIX 26–32 → **22**
    - VIX > 32 → **8** (extreme angst)

    Daarna correcties op basis van *richting* (stijgt of daalt de VIX de laatste 5 dagen?) en
    *historisch percentiel* (is dit een hoge of lage VIX voor dit jaar?). Een VIX van 22 die hard
    stijgt in een jaar dat al gespannen was, scoort lager dan dezelfde VIX die kalmeert vanuit een
    piektop.
    """)

    st.markdown("""
    **📈 Trend (20%)**

    Startwaarde **50**, dan opgeteld of afgetrokken per voorwaarde:

    - SPY boven MA200: **+18** punten (onder MA200: **−22**)
    - SPY boven MA50: **+13** (onder: **−13**)
    - SPY boven MA20: **+9** (onder: **−9**)
    - Bullish MA-stapel (MA20 > MA50 > MA200): **+10**
    - Bearish MA-stapel: **−10**
    - RSI > 60: **+5**, RSI < 40: **−8**, RSI < 30: **−16**
    - QQQ (tech) boven eigen MA50: **+5** (anders **−5**)

    Een volledig uitgelijnde bull-markt scoort dus ver boven 80; een neergaande trend onder MA200 zakt
    snel naar 0–20.
    """)

    st.markdown("""
    **🌊 Marktbreedte (20%)**

    Breedte meet of de marktbeweging *breed gedragen* wordt of slechts door enkele grote namen.
    Het systeem gebruikt vier indicatoren:

    - **% sectoren positief (5 dagen):** van de 11 S&P-sectoren (XLK, XLF, XLE...), hoeveel stegen
      de afgelopen week?
    - **A/D-ratio:** verhouding stijgende vs. dalende sectoren
    - **McClellan Oscillator (schatting):** gebaseerd op SPY's 10-daags vs. 19-daags exponentieel
      gemiddelde — meet of de breedte versnelt of vertraagt
    - **New Highs minus New Lows (schatting):** afgeleid van sectorparticipatie

    *Opmerking: echte NH/NL en A/D data vereisen dure datafeeds. Het systeem gebruikt schatting
    op basis van sectorgedrag als proxy.*
    """)

    st.markdown("""
    **🚀 Momentum (25%)**

    - **Sectorspreiding:** verschil tussen de best en slechtst presterende sector (5 dagen). Grote
      spreiding betekent duidelijke rotatie en energie — goed. Geen spreiding betekent vlakke,
      richtingloze markt.
    - **SPY 5-daags rendement:** directe maatstaf voor recente kracht (+15 punten bij > +2%,
      −15 bij < −2%).
    - **Aantal positieve sectoren:** idem als breedte, maar nu als momentumsignaal (hoe breed is
      de positieve beweging *nu*?)
    """)

    st.markdown("""
    **🏦 Macro (10%)**

    - **Fed-stance:** DOVISH (+20), NEUTRAL (+5), HAWKISH (−15). Afgeleid uit het niveau en de
      richting van de 10-jaarsrente.
    - **10-jaarsrente slope:** daalt de rente de laatste 5 dagen? (+8). Stijgt hij hard? (−10).
    - **DXY (dollar):** een dalende dollar is doorgaans gunstig voor aandelen.
    - **Absoluut niveau rente:** rente > 5,2% geeft een penalty (duur geld remt aandelenmarkt).

    Macro weegt het minst zwaar (10%) omdat rentebewegingen pas vertraagd doorwerken in koersen,
    en omdat de andere factoren de marktreactie al grotendeels *verwerken*.
    """)


    # ════════════════════════════════════════════════════════════════════
    # 3. VAN SCORE NAAR BESLUIT
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 3. Van score naar besluit")

    st.markdown("""
    De gewogen totaalscore bepaalt het eindbesluit. Er zijn twee modi met eigen drempels:
    """)

    st.markdown("""
    <div class="formula">
      MQS = (Volatiliteit × 0,25) + (Trend × 0,20) + (Breedte × 0,20) + (Momentum × 0,25) + (Macro × 0,10)
      <br/><br/>
      <span class="dim">Swing trading:</span>
        MQS ≥ 80  →  <span style="color:#00ff9d">YES</span> &nbsp;|&nbsp;
        60–79     →  <span style="color:#ffb700">CAUTION</span> &nbsp;|&nbsp;
        &lt; 60   →  <span style="color:#ff3a3a">NO</span>
      <br/>
      <span class="dim">Day trading:&nbsp;</span>
        MQS ≥ 75  →  <span style="color:#00ff9d">YES</span> &nbsp;|&nbsp;
        55–74     →  <span style="color:#ffb700">CAUTION</span> &nbsp;|&nbsp;
        &lt; 55   →  <span style="color:#ff3a3a">NO</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Day trading heeft lagere drempels omdat daarin sneller in en uit gestapt wordt —
    de omgeving hoeft minder perfect te zijn voor een dagkans.

    **Wat betekent elk signaal concreet?**
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
    <div class="callout callout-green">
      <strong>YES</strong><br/>
      Alle of de meeste factoren zijn gunstig.
      Volledige positiegrootte. Druk bij op A+
      setups. De markt werkt mee.
    </div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
    <div class="callout callout-yellow">
      <strong>CAUTION</strong><br/>
      Gemengde signalen. Handel alleen de sterkste
      setups, halfgrootte positie, strakke stops.
      Niet agressief zijn.
    </div>
    """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
    <div class="callout callout-red">
      <strong>NO</strong><br/>
      Omgeving is ongunstig. Blijf aan de zijlijn.
      Bewaar kapitaal. Bouw je watchlist op
      voor betere tijden.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # 4. EXECUTION WINDOW SCORE
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 4. De Execution Window Score")

    st.markdown("""
    Naast de MQS berekent het systeem een aparte **Execution Window Score** (0–100). Dit is geen
    onderdeel van het YES/NO-besluit, maar een aanvullende maatstaf voor de *kwaliteit van setups
    op dit moment*.

    De formule begint op 50 en past aan op basis van:

    - **Regime:** UPTREND geeft +20, DOWNTREND −20, CHOP ±0
    - **5-daags rendement van SPY:** sterk positief = setups die gevolgd worden; negatief = veel
      false breakouts
    - **Sectorparticipatie:** per sector boven of onder 5 actieve sectoren: ±3 punten per sector

    Een MQS van 75 (CAUTION) kan samengaan met een Execution Window van 30 — dat betekent: de markt
    is niet verschrikkelijk, maar setups falen nu veel. Andersom kan een YES-dag toch een lagere
    Execution Window hebben als de participatie smal is.
    """)

    st.markdown("""
    <div class="callout callout-blue">
      ◈ Gebruik de Execution Window als tiebreaker. Bij twee gelijkwaardige setups, kies degene op de
      dag met de hogere Execution Window Score.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # 5. DE LIVE DASHBOARD GRAFIEKEN (app.py)
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 5. De live dashboard — wat zie je in app.py?")

    st.markdown("### Scrollende ticker bovenaan")
    st.markdown("""
    De zwarte balk toont real-time prijzen en 1-daagse of 5-daagse veranderingen van:
    SPY, QQQ, IWM, VIX, TNX (10-jaarsrente) en DXY (dollar), plus alle 11 sector-ETFs.
    Groen = positief, rood = negatief, pijl omhoog/omlaag = richting.
    """)

    st.markdown("### De drie ringen (hero-sectie)")
    st.markdown("""
    - **YES / CAUTION / NO badge:** het directe antwoord, groot en duidelijk
    - **Linkerring — Market Quality Score:** het gewogen totaalcijfer (0–100). Groen boven 70,
      geel 50–70, rood onder 50
    - **Rechterring — Execution Window:** de kwaliteit van setupomgeving op dit moment
    - **Regime-kaart:** UPTREND (bullish MA-stapel), DOWNTREND, of CHOP — plus RSI14,
      VIX-percentiel, Fed-stance en sectoraantal
    """)

    st.markdown("### De vijf detailpanelen")
    st.markdown("""
    Elk paneel toont de actuele waarden die de sub-score bepalen, met pijltjes die de richting
    van de 5-daagse beweging aangeven (↑ goed, ↓ slecht, of omgekeerd voor VIX). De gekleurde
    balk rechtsboven in elk paneel is de sub-score zelf (groen/geel/rood).

    Het interpretatie-label onderaan elk paneel (bijv. "ELEVATED — REDUCE SIZE") is een directe
    vertaling van de score naar beheersadvies.
    """)

    st.markdown("### Sector heatmap")
    st.markdown("""
    Een horizontale balkgrafiek met alle 11 sector-ETFs en hun 5-daags rendement. De middenlijn
    is nul. Groene balken naar rechts = sector steeg; rode balken naar links = sector daalde.
    De grootste balk bepaalt de schaal.

    **Wat je zoekt:** als defensieve sectoren (XLU = Utilities, XLP = Staples, XLV = Health)
    leiden en groei-sectoren (XLK = Tech, XLY = Discretionary) achterblijven, is dat een teken
    van risicoaversie — slechte omgeving voor actief handelen.
    """)

    st.markdown("### Score breakdown")
    st.markdown("""
    Toont de bijdrage van elke factor: de balk is de ruwe sub-score, het getal ernaast is de gewogen
    bijdrage (sub-score × gewicht). De som geeft de MQS.
    """)

    st.markdown("### Terminal Analysis")
    st.markdown("""
    Een automatisch gegenereerde tekstvattening in terminal-stijl. Beschrijft het regime, de
    volatiliteitsomgeving, de breedte en de aanbevolen actie. Geen AI — puur regels op basis
    van de score-uitkomsten.
    """)


    # ════════════════════════════════════════════════════════════════════
    # 6. DE BACKTEST GRAFIEKEN (backtest.py)
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 6. De backtestgrafieken — wat zie je in backtest.py?")

    st.markdown("""
    De backtest draait dezelfde scoring engine op elke handelsdag van de afgelopen 5 jaar.
    Zo kun je zien hoe het systeem historisch presteerde — en wat de signalen *achteraf* waard waren.
    """)

    st.markdown("### KPI-rij bovenaan")
    st.markdown("""
    Negen kerncijfers, opgesplitst per signaaltype (YES / CAUTION / NO):

    - **Aantal dagen:** hoeveel handelsdagen viel in deze categorie?
    - **Gemiddeld rendement:** wat deed SPY *daarna* (over het gekozen venster: 1, 5, 10 of 20 dagen)?
    - **% positief:** op hoeveel van die dagen was het rendement daarna positief?

    Dit zijn de meest directe maatstaven voor de *voorspellende waarde* van het systeem.
    """)

    st.markdown("""
    <div class="callout callout-yellow">
      ◈ Let op: correlatie is geen causaliteit. Het systeem <em>reageert</em> op marktomstandigheden —
      het voorspelt niet de toekomst. Een hoog gemiddeld rendement op YES-dagen betekent dat die
      dagen samenhingen met gunstige marktomstandigheden, niet dat het systeem die omstandigheden
      voorspelde.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### €1000 beleggingscalculator")
    st.markdown("""
    De drie grote kaarten tonen wat er van een startbedrag overblijft als je:

    1. **Buy & Hold SPY:** elke dag belegd blijft, ongeacht het signaal
    2. **Handelen op YES + CAUTION:** uit de markt op NO-dagen
    3. **Handelen alleen op YES:** uit de markt op CAUTION en NO-dagen

    *Aanname:* je belegt in SPY (of een equivalent) en stapt in/uit tegen de slotkoers. Geen
    transactiekosten, geen slippage, geen dividend.

    **CAGR** = Compound Annual Growth Rate — het gemiddeld jaarlijks rendement als percentage.
    Een CAGR van 12% betekent dat je vermogen gemiddeld 12% per jaar groeide.

    **Max drawdown** = de grootste daling van een piek naar een dal, als percentage van de piek.
    Een max drawdown van −25% betekent dat je op het slechtste moment 25% onder je vorige
    hoogste stand stond. Lagere drawdown = minder pijnlijke periodes.

    De grafiek eronder toont de dagelijkse waardeontwikkeling van elk scenario, zodat je ziet
    wanneer de strategieën divergeren.
    """)

    st.markdown("""
    <div class="callout callout-green">
      ◈ De meeste waarde van "YES only" zit doorgaans <em>niet</em> in een hoger eindrendement, maar
      in een lagere max drawdown — je zat niet in de markt tijdens de ergste crashes.
      Minder stress, minder kapitaalverlies op het slechtste moment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Grafiek 1 — SPY-koers met signaalstipjes")
    st.markdown("""
    De blauwe lijn is de SPY-slotkoers. De gestippelde grijze lijn is de MA200.
    De gekleurde stippen geven per dag het signaal:

    - 🟢 **Groen (YES):** gunstige omgeving
    - 🟡 **Geel (CAUTION):** gemengd
    - 🔴 **Rood (NO):** ongunstig

    Je ziet meteen dat rode clusters samenvallen met marktneergang (COVID-crash maart 2020,
    neergang 2022) en groene clusters met rustige stijgingen (2023–2024 bull run).
    """)

    st.markdown("### Grafiek 2 — MQS over tijd + VIX")
    st.markdown("""
    Bovenste paneel: de dagelijkse MQS over 5 jaar. De gestippelde groene lijn markeert 80 (YES),
    de gele lijn 60 (CAUTION-grens). Wanneer de score onder 60 duikt, is het NO.

    Onderste paneel: de VIX. Je ziet de sterke negatieve correlatie: als VIX piekt (angst),
    duikt de MQS. Dit is deels tautologisch — volatiliteit weegt 25% in de score — maar de andere
    factoren (trend, breedte) versterken het effect.

    De 10-daagse smoothing maakt de trendstructuur zichtbaarder dan de ruwe dagwaarden.
    """)

    st.markdown("### Grafiek 3a — Rendementsverdeling per signaal (violin)")
    st.markdown("""
    Een violin-grafiek toont de *volledige verdeling* van rendementen — breder = meer dagen met
    dat rendement. De box in het midden toont mediaan en interkwartielafstand.

    Wat je zoekt:
    - Is de mediaan van YES-dagen hoger dan die van NO-dagen?
    - Is de spreiding (breedte van de viool) bij NO-dagen groter? (meer onzekerheid)
    - Zijn er grote negatieve uitschieters bij NO-dagen? (staartrisico)
    """)

    st.markdown("### Grafiek 3b — Gemiddeld rendement per MQS-bucket")
    st.markdown("""
    De horizontale as toont MQS-bereiken (0–10, 10–20, ..., 90–100). De verticale as het
    gemiddelde forward-rendement. Groene staven = positief gemiddeld rendement na die MQS,
    rood = negatief.

    Een ideaal systeem laat een duidelijke stijgende lijn zien: hoge MQS → hoger rendement.
    In de praktijk is de relatie rommelig, maar de richting is informatief.
    """)

    st.markdown("### Grafiek 4 — Sub-scores over tijd")
    st.markdown("""
    Vijf lijnen tonen de 10-daags rollend gemiddelde van elke sub-score. Dit onthult welke factoren
    de MQS het meest bewegen.

    Typische patronen:
    - **Volatiliteit en momentum** reageren snel — ze schommelen het meest
    - **Macro** is stabiel — rentebewegingen zijn traag
    - **Trend** daalt geleidelijk bij een neergaande markt en herstelt pas na MA-kruisingen
    - Als *alle* lijnen tegelijk instorten, is de marktomgeving extreem slecht
    """)

    st.markdown("### Grafiek 5 — Maandelijkse YES%-heatmap")
    st.markdown("""
    Een kalenderweergave: per maand en jaar, welk percentage van de handelsdagen was YES?
    Donkergroen = veel YES-dagen (gunstige maand), donkerrood = weinig (slechte maand).

    Dit toont seizoenspatronen en specifieke crisisperiodes:
    - Maart 2020: donkerrood (COVID-crash)
    - Eind 2022: donkerrood (renteschok)
    - 2023–2024: overwegend groen (herstelbull)

    Het is geen basis voor "sell in May" — maar het toont wel in welke periodes het systeem
    historisch weinig handelskansen zag.
    """)

    st.markdown("### Grafiek 6 — Cumulatieve equity-curve")
    st.markdown("""
    Drie lijnen tonen de waardeontwikkeling van €1 startkapitaal:

    - **Gestippeld grijs — Buy & Hold:** altijd belegd, elke dag het volledige SPY-rendement
    - **Geel — YES + CAUTION:** uit de markt op NO-dagen, verder mee met SPY
    - **Groen — YES only:** alleen belegd op YES-dagen

    De fill (groene schaduw) is het verschil tussen YES-only en YES+CAUTION.

    **Let op de schaal van de Y-as** — de eindwaardes worden ook getoond als annotatie rechts
    van elke lijn, in euro's.
    """)


    # ════════════════════════════════════════════════════════════════════
    # 7. BEPERKINGEN
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 7. Beperkingen en kanttekeningen")

    st.markdown("""
    **Look-ahead bias is afwezig** — de scoring gebruikt alleen data die op dat moment beschikbaar
    was. MA's worden berekend over histórica, forward returns worden *alleen* gebruikt voor
    evaluatie, niet voor scoring.

    **Data-kwaliteit:** de breedte-indicatoren (McClellan, NH/NL, A/D-ratio) zijn *schattingen*
    op basis van de 11 sector-ETFs. Professionele backtests gebruiken volledige NYSE/Nasdaq-data.
    De richting van de indicatoren klopt, maar de absolute waarden zijn approximaties.

    **Transactiekosten:** de equity-curves houden geen rekening met kosten, slippage of
    dividendherbelgging. In de praktijk maakt dit een klein negatief verschil voor de strategieën
    die vaker handelen.

    **Overfitting:** de drempelwaarden en gewichten zijn *niet* geoptimaliseerd op historische data.
    Ze zijn bepaald op basis van marktkennis. Een geoptimaliseerd systeem zou op de backtest beter
    presteren maar slechtere toekomstige resultaten geven.

    **Het is geen handelssysteem:** het systeem zegt niets over welk aandeel te kopen, wanneer
    precies in te stappen of hoe de positie te beheren. Het geeft alleen de marktomgeving aan.
    """)

    st.markdown("""
    <div class="callout callout-red">
      ◈ EDUCATIEF GEBRUIK ALLEEN. GEEN BELEGGINGSADVIES. Historische prestaties geven geen garantie
      voor toekomstige resultaten. Handel altijd op basis van eigen analyse en risicotolerantie.
    </div>
    """, unsafe_allow_html=True)


    # ════════════════════════════════════════════════════════════════════
    # 8. TECHNISCH
    # ════════════════════════════════════════════════════════════════════
    st.markdown("## 8. Technische structuur")

    st.markdown("""
    Het systeem bestaat uit vier bestanden die samen worden gebruikt:

    | Bestand | Inhoud |
    |---|---|
    | `scoring.py` | De scoring engine en constanten — **hier wijzig je drempels en gewichten** |
    | `market_data.py` | Dataverzameling: live fetch (yfinance) en historische data |
    | `app.py` | Het live Bloomberg-terminal dashboard |
    | `backtest.py` | De 5-jaar backtest met alle grafieken |

    **Aanpassen:** als je een gewicht wilt veranderen — zeg, volatiliteit van 25% naar 30% —
    doe je dat op één plek in `scoring.py`. Beide apps gebruiken dan automatisch de nieuwe waarde.
    Hetzelfde geldt voor de YES/NO-drempels in `THRESHOLDS`.

    **Data:** bij het eerste gebruik van `backtest.py` worden 6 jaar aan dagdata gedownload
    (~30 seconden). Dit wordt 1 uur gecached. Bij `app.py` wordt alleen het afgelopen jaar
    opgehaald en elke 30 seconden bijgewerkt.
    """)

    st.markdown("""
    <div class="disclaimer">
      ⚠ DISCLAIMER — Dit systeem en deze documentatie zijn uitsluitend bedoeld voor educatieve en
      informatieve doeleinden. Niets in dit artikel of de bijbehorende software vormt beleggingsadvies,
      financieel advies of een aanbeveling om te kopen of verkopen. De auteur is geen financieel
      adviseur. Handel altijd verantwoord en in overeenstemming met uw eigen risicoprofiel.
      Verleden resultaten bieden geen garantie voor de toekomst.
    </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    uitleg_backtest()


if __name__ == "__main__":
    main()
