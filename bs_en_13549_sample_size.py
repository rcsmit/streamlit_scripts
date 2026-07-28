"""Seizoensbeoordeling - AQL-niveau & bonus/malus calculator.

Bepaalt voor een gegeven aantal cleans en aantal afgekeurd het AQL-niveau
en de bijbehorende bonus/malus. De steekproefgrootte (n) komt uit de
officiele BS EN 13549 Tabel C.1 (Level 2).

De niveaudrempels (acceptatiegetal Ac per niveau) kunnen op twee manieren
berekend worden:
  - Binomiaal: net als de officiele Tabel C.2 in BS EN 13549 (onafhankelijk
    van de lotgrootte N -- geverifieerd tegen de gepubliceerde PAQ-waarden).
  - Hypergeometrisch: houdt wel rekening met de exacte, eindige lotgrootte N.

PAQ (Probability of Acceptance at AQL) en PLQ (Probability of Acceptance
at Limiting Quality) zijn beide BEREKENDE resultaten, symmetrisch aan
elkaar:
  - AQL is een vast, zelf/contractueel gekozen kwaliteitspunt -> PAQ wordt
    daaruit berekend.
  - LQ is een vast, zelf gekozen "te slecht"-referentiepunt (LQ = AQL x
    instelbare multiplier) -> PLQ wordt daaruit berekend.
Geen van beide wordt omgekeerd opgelost naar een vooraf vastgezet
percentage -- dat zou de kolom tot een echo van een instelling maken in
plaats van een output. (Een losse find_lq()-hulpfunctie is beschikbaar
voor wie wil weten bij welke kwaliteit Pa daalt tot een gekozen
consumentrisico, conform de ISO 2859-1/2859-2 definitie van LQ -- maar
dat is een apart vraagstuk, niet de PLQ-kolom in de standaardtabel.)

Let op: Pa(AQL) is GEEN vaste 95% -- dat is slechts een vuistregel die in
de praktijk vaak (niet altijd) ongeveer klopt. Sommige officiele plannen
(bijv. n=50, Ac=0) hebben een producer's risico van wel 39,5%.
"""

from __future__ import annotations

# version : 20260715-120000 - Initial version: tabel-reproductie en calculator
# version : 20260715-133000 - Niveaudrempels worden nu berekend (floor(n*AQL/100))
#                             i.p.v. hardcoded, en n komt uit BS EN 13549 Tabel C.1
#                             zodat de app voor elk aantal cleans werkt
# version : 20260715-141500 - Niveaudrempels nu berekend met de hypergeometrice
#                             verdeling (acceptatiegetal bij gekozen zekerheid)
#                             i.p.v. de vuistregel floor(n*AQL/100)
# version : 20260715-153000 - Schakelaar binomial/hypergeometric toegevoegd;
#                             from __future__ import annotations nu direct na de
#                             docstring (was per ongeluk na current_version, wat
#                             een echte SyntaxError veroorzaakte); nieuwe PAQ/PLQ
#                             tabellen toegevoegd in beide tabbladen
# version : 20260715-161500 - LQ eerst omgezet naar "kwaliteit waarbij Pa =
#                             consumentrisico" (ISO-definitie), maar dat maakte
#                             PLQ een echo van de ingestelde risico-slider
# version : 20260715-163000 - Teruggedraaid: LQ is weer een vast, zelf gekozen
#                             referentiepunt (AQL x multiplier); PLQ is puur het
#                             berekende resultaat daarvan, net als PAQ bij AQL.
#                             find_lq() blijft beschikbaar als apart hulpmiddel.
# version : 20260720-000000 - Derde tabblad "Officiele tabellen" toegevoegd met
#                             letterlijk overgenomen Tabel C.2 (normal), C.3
#                             (tightened) en C.7 (AQL=7% conformity index) uit
#                             BS EN 13549:2001 Annex C, ter verificatie van de
#                             zelf berekende waarden. Default "Zekerheid bij
#                             AQL" slider aangepast van 95% naar 98%.
# version : 20260720-010000 - Elke officiele tabel krijgt nu een expander met
#                             de eigen herberekende PAQ (probability_of_acceptance,
#                             binomial) op basis van dezelfde (n, Ac, AQL),
#                             met Delta-kolom en check-icoon per rij, plus een
#                             via find_lq() teruggerekende impliciete LQ voor
#                             de PLQ-kolom (die de standaard zelf niet vermeldt).
# version : 20260720-013000 - Bug: impliciete LQ gaf "n.b." zodra de officiele
#                             PLQ afgerond 0,0% was (find_lq had dan geen exacte
#                             oplossing). Nu wordt in dat geval een ONDERGRENS
#                             (>=) berekend: het kwaliteitsniveau waarbij Pa net
#                             onder de afrondingsgrens van 0,05% duikt.
# version : 20260720-020000 - Vergelijking vereenvoudigd op verzoek: het
#                             terugzoeken van de impliciete LQ (find_lq) is
#                             vervangen door een directe berekening: PLQ
#                             berekend = probability_of_acceptance(n, Ac,
#                             lq_target_pct), exact dezelfde functie als PAQ
#                             maar met een ander doelpercentage. Dat doel-
#                             percentage is nu een instelbare number_input in
#                             het tabblad ("LQ (doel-kwaliteitspercentage %)"),
#                             met eigen Delta- en check-kolom naast PAQ.
# version : 20260720-021500 - Bug: Check-kolommen toonden de kale tekst
#                             ":material/check_circle:" / ":material/warning:"
#                             in plaats van een icoon, omdat die shortcodes
#                             alleen in st.markdown/st.button/tab-labels
#                             worden gerenderd, niet in st.dataframe-cellen.
#                             _check_symbol() geeft nu echte unicode-tekens
#                             terug (checkmark / waarschuwingsdriehoek).
# version : 20260720-023000 - LQ voor Tabel C.2/C.3 is niet langer een
#                             gedeelde waarde maar drie losse number_inputs,
#                             een per AQL-kolom (4%, 6,5%, 10%), want elke
#                             kolom heeft zijn eigen impliciete LQ die niet
#                             allemaal met dezelfde factor schalen.
#                             build_comparison_df() accepteert nu een
#                             lq_targets-dict i.p.v. een losse float. Alle
#                             PAQ/PLQ-percentages in de officiele en
#                             vergelijkingstabellen tonen nu 2 in plaats van
#                             1 cijfer achter de komma.
# version : 20260720-024500 - Gebruiker viel op dat n=50/10% in Tabel C.2 een
#                             PLQ van 32,80% toont, wat het monotoon dalende
#                             patroon binnen die kolom breekt (21,20% bij
#                             n=32, 0,80% bij n=80); eigen berekening geeft
#                             7,89% wat wel in de reeks past. Cel wordt nu
#                             gemarkeerd met een "*" en voetnoot in de tabel
#                             (SUSPECT_CELLS_C2) i.p.v. stilzwijgend
#                             overgenomen -- de brontabel is niet aangepast
#                             omdat de scan niet met zekerheid herlezen kan
#                             worden.
# version : 20260720-030000 - Vierde tabblad "Overtuig de directie" toegevoegd:
#                             combinatie van blogpost, (populair)wetenschappelijk
#                             artikel en managementpitch die uitlegt waarom een
#                             vast controlepercentage geen kwaliteitscontrole
#                             is, de ISO 2859/BS EN 13549-geschiedenis schetst,
#                             en met Ainslie/delay-discounting, Kahneman &
#                             Tversky/loss-aversion, Ostrom/gegradueerde
#                             sancties en Juran/cost-of-quality onderbouwt
#                             waarom een maandelijks, vooral belonend
#                             bonus/malus-systeem beter werkt dan een vaag
#                             seizoensdreigement. Toont de live AQL/BONUS_MALUS
#                             niveautabel rechtstreeks uit de code (geen
#                             hardcoded duplicaat) plus een bronnenlijst.
# version : 20260720-051500 - Corrected an overreaching claim in "What is
#                             this table based on?": the text used to assert
#                             a rigorous 1/sqrt(n) statistical derivation for
#                             why sample size grows sub-linearly with lot
#                             size, which isn't actually how these
#                             code-letter tables were derived. Replaced with
#                             an honest, empirically-computed demonstration:
#                             a log-log scatter (bracket midpoint vs Level II
#                             sample size, from TABLE_C1) with a fitted
#                             trend line, live-computed slope (k) and R^2 --
#                             showing the real regularity (a power law with
#                             k roughly 0.5-0.6, close to but not literally
#                             the square-root law) instead of asserting an
#                             unfounded formula. New import: math.
# version : 20260720-050000 - Whole app translated to English: title, all
#                             tab labels, captions, dataframe columns, chart
#                             titles/legends, widget labels, and error
#                             messages. Internal method tokens "binomiaal"/
#                             "hypergeometrisch" renamed to "binomial"/
#                             "hypergeometric" throughout. Code comments and
#                             docstrings (developer-facing, not shown in the
#                             app) were left in Dutch. "What is this table
#                             based on?" is now inside its own bordered
#                             container. Added a demerit-rating scoring rule
#                             (score = fatal x 9 + major x 3 + minor x 1,
#                             threshold = 9 -> reject above 9) in a bordered
#                             container with a full fatal/major/minor
#                             examples table.
# version : 20260720-044500 - BS EN 13549 Tabel C.1 toegevoegd in de sectie
#                             "Het probleem met 'we controleren 25%'":
#                             de volledige lotgrootte/steekproefgrootte-tabel
#                             (TABLE_C1) als dataframe, met uitleg over de
#                             herkomst (MIL-STD-105E / ANSI Z1.4 / ISO
#                             2859-1) en de twee ontwerpkeuzes erachter: de
#                             wortelwet (1/√n-schaling) en de
#                             voorkeurgetallenreeks (~1,6x per stap) achter
#                             de steekproefgroottes, plus uitleg van de
#                             drie inspection levels.
# version : 20260720-044000 - OC-tabel en -grafiek in tab4: n x AQL (geeft
#                             PAQ) en 3 x n x AQL (geeft PLQ, bij de
#                             standaard LQ-multiplier) toegevoegd als extra
#                             regels in de Markering-kolom en als extra
#                             gestippelde verticale lijnen (bruin/blauw)
#                             naast de bestaande rode Ac-lijn. Het getoonde
#                             traject wordt zo nodig verruimd zodat alle
#                             drie referentiepunten altijd zichtbaar
#                             blijven, ook als ze samenvallen met dezelfde
#                             rij (dan worden de markeringen gecombineerd).
# version : 20260720-043500 - Caption bij de OC-grafiek gecorrigeerd: noemde
#                             het getoonde stuk nog "S-vormig", terwijl na
#                             het inkorten van het bereik (zie vorige
#                             versie) juist alleen het steile, vrijwel
#                             rechtlijnige middenstuk zichtbaar is. Tekst
#                             legt nu uit dat de vlakke plateaus bewust zijn
#                             afgesneden en dat de rechtlijnigheid binnen dit
#                             venster dus klopt.
# version : 20260720-043000 - OC-tabel en -grafiek in tab4 ingekort: begint
#                             nu bij de eerste rij waar de kans net onder
#                             99,95% zakt en stopt bij de laatste rij waar de
#                             kans nog net boven 0,05% zit (de vlakke
#                             uiteinden bij ~100%/~0% voegden niets toe). Ac
#                             blijft altijd binnen het getoonde traject,
#                             ook als dat verder in de staart zou liggen.
# version : 20260720-042500 - OC-tabel en -grafiek in tab4: x-as en 1e kolom
#                             zijn nu hele getallen van 0 t/m n (in plaats van
#                             AQL-veelvouden). Ac wordt gemarkeerd als
#                             tabelregel ("<- Ac") en als aparte rode
#                             verticale lijn in de grafiek, naast de
#                             bestaande bruine n x AQL-lijn.
# version : 20260720-041500 - OC-grafiek in tab4: x-as toont nu het (verwachte)
#                             aantal vieze huisjes in de steekproef (n x
#                             werkelijk %) in plaats van het werkelijke
#                             percentage. De referentietabel ernaast toont
#                             beide (percentage en aantal); de gestippelde
#                             AQL-referentielijn staat nu op n x AQL, dezelfde
#                             waarde als de "n x AQL"-metric hierboven.
# version : 20260720-040000 - Tweede grafiek toegevoegd onder de OC-curve in
#                             tabblad "Overtuig de directie": een staafdiagram
#                             van de cumulatieve binomiale kans P(X<=c) per
#                             acceptatiegetal c (0 t/m Ac+3), met een lijn op
#                             de 98%-zekerheidsdrempel, zodat in een oogopslag
#                             te zien is waarom find_acceptance_number() bij
#                             n=20 en AQL=6,5% precies Ac=4 teruggeeft (c=3
#                             haalt de drempel nog niet, c=4 wel).
# version : 20260720-031500 - Kader toegevoegd onder "Dit gereedschap, in de
#                             praktijk" dat uitlegt waarom een AQL van 6,5%
#                             NIET betekent dat je maximaal 6,5% vieze huisjes
#                             tolereert: PAQ/PLQ uitgelegd op middelbare-
#                             school-niveau (bloedproef-analogie), met een live
#                             berekend OC-curve-voorbeeld (100 cleans, n=20,
#                             Ac=4 bij AQL=6,5%/98% zekerheid) dat laat zien
#                             dat zelfs bij 19,5% werkelijk vieze huisjes de
#                             goedkeuringskans nog ~65% is.
# version : 20260720-032500 - Grafiek toegevoegd onder de OC-curve-tabel in
#                             dat kader: een Altair line chart van de kans op
#                             goedkeuring (0-40% werkelijk vuil-percentage, in
#                             stappen van 0,25pp) met een gestippelde
#                             referentielijn bij de AQL-grens (6,5%). Nieuwe
#                             import: altair.
current_version = "20260720-051500"

import math

import altair as alt
import pandas as pd
import streamlit as st
from scipy.optimize import brentq
from scipy.stats import binom, hypergeom

st.set_page_config(page_title="AQL Level Table", page_icon=":material/fact_check:", layout="wide")

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
# BS EN 13549 Tabel C.1 (Annex C): (lot_from, lot_until, level1, level2, level3)
TABLE_C1: list[tuple[int, int, int, int, int]] = [
    (2, 8, 2, 2, 3),
    (9, 15, 2, 3, 5),
    (16, 25, 3, 5, 8),
    (26, 50, 5, 8, 13),
    (51, 90, 5, 13, 20),
    (91, 150, 8, 20, 32),
    (151, 280, 13, 32, 50),
    (281, 500, 20, 50, 80),
    (501, 1200, 32, 80, 125),
    (1201, 3200, 50, 125, 200),
    (3201, 10000, 80, 200, 315),
    (10001, 35000, 125, 315, 500),
    (35001, 150000, 200, 500, 800),
    (150001, 500000, 315, 800, 1250),
    (500001, 600000000, 500, 1250, 2000),
]
LEVEL_COLUMN: dict[int, int] = {1: 2, 2: 3, 3: 4}

# Eigen contractuele AQL-niveaus en bijbehorende bonus/malus.
NIVEAUS: list[str] = ["I", "II", "III", "IV", "V"]
AQL: dict[str, float] = {"I": 10, "II": 10, "III": 6.5, "IV": 4, "V": 2}
BONUS_MALUS: dict[str, int] = {"I": -10, "II": -5, "III": 0, "IV": 5, "V": 10}

DEFAULT_LEVEL = 2  # BS EN 13549 sampling level (1, 2 of 3)
DEFAULT_CONFIDENCE = 0.98  # zekerheid dat een lot op AQL-niveau geaccepteerd wordt
DEFAULT_LQ_MULTIPLIER = 3.0  # LQ = AQL x deze multiplier -- zelf gekozen referentiepunt
VOORBEELD_CLEANS: list[int] = [50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 10000]

# --------------------------------------------------------------------------
# Officiele referentietabellen (BS EN 13549:2001, Annex C)
# --------------------------------------------------------------------------
# Deze drie tabellen zijn LETTERLIJK overgenomen uit de standaard (niet
# berekend) en dienen als controle op de zelf berekende PAQ/PLQ-waarden
# elders in deze app. Een cel is None wanneer de standaard voor die
# combinatie van steekproefgrootte/AQL geen Ac definieert.
#
# Elke waarde: (Ac, PAQ in %, PLQ in %)
_TableEntry = "tuple[int, float, float] | None"

# Tabel C.2 - Single sampling: normal inspection
TABLE_C2_NORMAL: dict[int, dict[str, tuple[int, float, float] | None]] = {
    5:   {"4%": None,             "6,5%": None,             "10%": (1, 91.9, 52.8)},
    8:   {"4%": None,             "6,5%": (1, 90.9, 51.8),  "10%": (2, 96.2, 55.2)},
    13:  {"4%": (1, 90.7, 52.6),  "6,5%": (2, 95.2, 52.0),  "10%": (3, 96.6, 42.1)},
    20:  {"4%": (2, 95.6, 56.3),  "6,5%": (3, 96.3, 43.3),  "10%": (5, 98.8, 41.6)},
    32:  {"4%": (3, 96.2, 45.4),  "6,5%": (5, 98.4, 38.7),  "10%": (7, 98.8, 21.2)},
    50:  {"4%": (5, 98.6, 43.5),  "6,5%": (7, 98.5, 21.5),  "10%": (10, 99.0, 32.8)},
    80:  {"4%": (7, 98.5, 24.2),  "6,5%": (10, 98.6, 7.0),  "10%": (14, 98.8, 0.8)},
    125: {"4%": (10, 98.8, 10.3), "6,5%": (14, 98.4, 1.0),  "10%": (21, 99.0, 0.1)},
    200: {"4%": (14, 98.5, 1.5),  "6,5%": (21, 98.9, 0.0),  "10%": None},
    315: {"4%": (21, 99.6, 0.1),  "6,5%": None,             "10%": None},
    500: {"4%": None,             "6,5%": None,             "10%": None},
}

# Tabel C.3 - Single sampling: tightened inspection
TABLE_C3_TIGHTENED: dict[int, dict[str, tuple[int, float, float] | None]] = {
    5:   {"4%": None,             "6,5%": None,             "10%": None},
    8:   {"4%": None,             "6,5%": None,             "10%": None},
    13:  {"4%": None,             "6,5%": None,             "10%": (1, 81.3, 25.5)},
    20:  {"4%": (1, 81.0, 28.9),  "6,5%": (2, 86.3, 22.2),  "10%": (3, 86.7, 10.7)},
    32:  {"4%": (2, 86.5, 24.4),  "6,5%": (3, 84.8, 10.5),  "10%": (5, 90.6, 5.1)},
    50:  {"4%": (3, 86.1, 13.5),  "6,5%": (5, 89.6, 5.7),   "10%": (8, 94.2, 1.8)},
    80:  {"4%": (5, 89.9, 7.1),   "6,5%": (8, 92.5, 1.7),   "10%": (12, 94.6, 0.2)},
    125: {"4%": (8, 93.6, 2.9),   "6,5%": (12, 93.7, 0.2),  "10%": (18, 95.7, 0.0)},
    200: {"4%": (12, 94.0, 0.4),  "6,5%": (18, 93.7, 0.0),  "10%": None},
    315: {"4%": (18, 94.9, 0.0),  "6,5%": None,             "10%": None},
    500: {"4%": None,             "6,5%": None,             "10%": None},
}

# Tabel C.7 - Conformity index table for AQL = 7 %
# LET OP: dit is exact wat leesbaar was op de aangeleverde pagina-scan
# (pagina 19). De volgende reeksen ontbreken omdat ze buiten de
# gefotografeerde kolommen vielen en zijn dus NIET in deze lijst
# opgenomen: sample size 26-65, 93-130 en 159-195. Sample size 72 (tussen
# 71 en 73) en de laatste waarden van sample size 158 waren niet leesbaar
# en zijn eveneens weggelaten. Vul aan met een scan van de ontbrekende
# regels als je de volledige tabel nodig hebt.
# Elke rij: (sample size, Ac, PAQ %, PLQ %, PAQ> % of None)
TABLE_C7_AQL_7: list[tuple[int, int, float, float, float | None]] = [
    (1, 0, 93.0, 79.0, 90.0),
    (2, 1, 99.5, 95.6, None),
    (3, 1, 98.6, 88.6, None),
    (4, 1, 97.3, 80.4, None),
    (5, 1, 95.8, 71.7, None),
    (6, 1, 93.9, 63.1, None),
    (7, 1, 91.9, 54.9, None),
    (8, 2, 98.5, 77.5, None),
    (9, 2, 97.9, 71.1, None),
    (10, 2, 97.2, 64.7, None),
    (11, 2, 96.3, 58.4, None),
    (12, 2, 95.3, 52.3, None),
    (13, 2, 94.2, 46.5, None),
    (14, 2, 93.0, 41.1, None),
    (15, 2, 91.7, 36.1, None),
    (16, 3, 97.3, 50.7, None),
    (17, 3, 96.7, 45.9, None),
    (18, 3, 96.0, 41.2, None),
    (19, 3, 95.3, 36.9, None),
    (20, 3, 94.5, 32.9, None),
    (21, 3, 93.6, 29.1, None),
    (22, 3, 92.7, 25.7, None),
    (23, 3, 91.7, 22.7, None),
    (24, 4, 96.3, 30.2, None),
    (25, 4, 95.7, 27.1, None),
    (66, 8, 96.0, 4.6, None),
    (67, 8, 95.7, 4.1, None),
    (68, 8, 95.3, 3.6, None),
    (69, 9, 97.9, 6.5, None),
    (70, 9, 97.6, 5.8, None),
    (71, 10, 99.0, 9.5, 98.0),
    (73, 10, 98.9, 8.6, None),
    (74, 10, 98.8, 7.8, None),
    (75, 10, 98.5, 6.3, None),
    (76, 10, 98.3, 5.7, None),
    (77, 10, 98.2, 5.1, None),
    (78, 10, 98.0, 4.5, None),
    (79, 11, 99.1, 7.5, None),
    (80, 11, 99.0, 6.8, None),
    (81, 11, 98.9, 6.1, None),
    (82, 11, 98.8, 5.5, None),
    (83, 11, 98.7, 5.0, None),
    (84, 11, 98.6, 4.5, None),
    (85, 11, 98.5, 4.0, None),
    (86, 11, 98.3, 3.6, None),
    (87, 11, 98.2, 3.2, None),
    (88, 11, 98.0, 2.9, None),
    (89, 12, 99.1, 4.8, None),
    (90, 12, 99.0, 4.4, None),
    (91, 12, 98.9, 3.9, None),
    (92, 12, 98.8, 3.5, None),
    (131, 16, 99.0, 0.7, None),
    (132, 16, 98.9, 0.6, None),
    (133, 16, 98.8, 0.5, None),
    (134, 16, 98.7, 0.5, None),
    (135, 16, 98.6, 0.4, None),
    (136, 16, 98.6, 0.4, None),
    (137, 16, 98.4, 0.3, None),
    (138, 16, 98.3, 0.3, None),
    (139, 16, 98.2, 0.3, None),
    (140, 16, 98.1, 0.2, None),
    (141, 17, 98.9, 0.4, None),
    (142, 17, 98.9, 0.4, None),
    (143, 17, 98.9, 0.3, None),
    (144, 17, 98.8, 0.3, None),
    (145, 17, 98.7, 0.2, None),
    (146, 17, 98.6, 0.2, None),
    (147, 17, 98.5, 0.2, None),
    (148, 17, 98.4, 0.2, None),
    (149, 17, 98.3, 0.2, None),
    (150, 17, 98.2, 0.1, None),
    (151, 18, 99.1, 0.1, None),
    (152, 18, 99.0, 0.2, None),
    (153, 18, 98.9, 0.2, None),
    (154, 18, 98.8, 0.2, None),
    (155, 18, 98.8, 0.1, None),
    (156, 18, 98.7, 0.1, None),
    (157, 18, 98.6, 0.1, None),
    (196, 21, 98.0, 0.0, None),
    (197, 21, 98.9, 0.0, None),
    (198, 22, 98.8, 0.0, None),
    (199, 22, 98.7, 0.0, None),
    (200, 22, 98.7, 0.0, None),
    (201, 22, 98.6, 0.0, None),
    (202, 22, 98.5, 0.0, None),
    (203, 22, 98.4, 0.0, None),
    (204, 22, 98.3, 0.0, None),
    (205, 22, 98.2, 0.0, None),
    (206, 22, 98.2, 0.0, None),
    (207, 23, 98.9, 0.0, None),
    (208, 23, 98.8, 0.0, None),
    (209, 23, 98.8, 0.0, None),
    (210, 23, 98.7, 0.0, None),
    (211, 23, 98.7, 0.0, None),
    (212, 23, 98.6, 0.0, None),
    (213, 23, 98.5, 0.0, None),
    (214, 23, 98.4, 0.0, None),
    (215, 23, 98.3, 0.0, None),
    (216, 23, 98.3, 0.0, None),
    (217, 23, 98.2, 0.0, None),
    (218, 23, 98.1, 0.0, None),
    (219, 24, 98.9, 0.0, None),
    (220, 24, 98.8, 0.0, None),
    (221, 24, 98.8, 0.0, None),
    (222, 24, 98.7, 0.0, None),
]


# --------------------------------------------------------------------------
# Logica: sample size (BS EN 13549 Tabel C.1)
# --------------------------------------------------------------------------
def get_sample_size(lot_size: int, level: int = DEFAULT_LEVEL) -> int:
    """Zoek de steekproefgrootte op in BS EN 13549 Tabel C.1.

    Args:
        lot_size: Totaal aantal cleans (lot size).
        level: Sampling level: 1, 2 of 3.

    Returns:
        De bijbehorende steekproefgrootte n.

    Raises:
        ValueError: Als lot_size buiten het bereik van de tabel valt.
    """
    col = LEVEL_COLUMN[level]
    for row in TABLE_C1:
        if row[0] <= lot_size <= row[1]:
            return row[col]
    raise ValueError(
        f"lot_size {lot_size} is outside the range of Table C.1 "
        f"({TABLE_C1[0][0]} to {TABLE_C1[-1][1]})"
    )


# --------------------------------------------------------------------------
# Logica: acceptatiekans (PAQ/PLQ) per methode
# --------------------------------------------------------------------------
def probability_of_acceptance(n: int, c: int, quality_pct: float, method: str,
                               lot_size: int | None = None) -> float:
    """Bereken de kans dat een lot met een gegeven kwaliteit wordt geaccepteerd.

    Args:
        n: Steekproefgrootte.
        c: Acceptatiegetal (max. toegestaan aantal afgekeurd in de steekproef).
        quality_pct: Werkelijk percentage afwijkend in de lot (bijv. de AQL
            of de LQ), als percentage (0-100).
        method: "binomial" (onafhankelijk van lotgrootte, zoals de
            officiele Tabel C.2) of "hypergeometric" (houdt rekening
            met de eindige lotgrootte N).
        lot_size: Lotgrootte N. Verplicht bij method="hypergeometric".

    Returns:
        De kans (0.0-1.0) dat de lot wordt geaccepteerd.

    Raises:
        ValueError: Als method="hypergeometric" en lot_size ontbreekt,
            of als method een onbekende waarde heeft.
    """
    p = quality_pct / 100
    if method == "binomial":
        return float(binom.cdf(c, n, p))
    if method == "hypergeometric":
        if lot_size is None:
            raise ValueError("lot_size is required when method='hypergeometric'")
        defects_at_quality = round(lot_size * p)
        return float(hypergeom.cdf(c, lot_size, defects_at_quality, n))
    raise ValueError(f"Unknown method: {method!r}")


def find_lq(n: int, ac: int, method: str, consumer_risk: float = 0.10,
            lot_size: int | None = None) -> float | None:
    """Zoek de Limiting Quality (LQ): het kwaliteitsniveau waarbij de kans
    op acceptatie gelijk is aan het gekozen consumentrisico.

    Dit volgt de officiele ISO 2859-1/2859-2 definitie: LQ is niet iets
    dat je vooraf kiest als vaste multiplier van de AQL -- het is het
    kwaliteitsniveau waarbij Pa (de acceptatiekans) daalt tot het
    consumentrisico (doorgaans 10%). Zie ISO 2859-2: "the sampling plans
    are indexed by a series of specified values of limiting quality (LQ),
    where the consumer's risk (probability of acceptance at the LQ) is
    usually below 10%".

    Args:
        n: Steekproefgrootte.
        ac: Acceptatiegetal.
        method: "binomial" of "hypergeometric".
        consumer_risk: Gewenst consumentrisico (kans op acceptatie bij LQ).
        lot_size: Lotgrootte N. Verplicht bij method="hypergeometric".

    Returns:
        Het kwaliteitsniveau (percentage, 0-100) waarbij Pa = consumer_risk,
        of None als er geen oplossing bestaat (bijv. als ac == n, waarbij
        de lot altijd wordt geaccepteerd, ongeacht de kwaliteit).
    """
    def f(q: float) -> float:
        return probability_of_acceptance(n, ac, q * 100, method, lot_size=lot_size) - consumer_risk

    lo, hi = 1e-6, 0.999
    if f(lo) < 0 or f(hi) > 0:
        return None
    return brentq(f, lo, hi) * 100


def find_acceptance_number(n: int, quality_pct: float, method: str, confidence: float,
                            lot_size: int | None = None) -> int:
    """Zoek het kleinste acceptatiegetal c waarbij de acceptatiekans op het
    gegeven kwaliteitsniveau minstens `confidence` is.

    Args:
        n: Steekproefgrootte.
        quality_pct: Kwaliteitsniveau (percentage afwijkend) om op te toetsen
            -- meestal de AQL.
        method: "binomial" of "hypergeometric".
        confidence: Gewenste minimale kans op acceptatie bij dit
            kwaliteitsniveau.
        lot_size: Lotgrootte N. Verplicht bij method="hypergeometric".

    Returns:
        Het kleinste acceptatiegetal c dat aan de eis voldoet (of n, als
        zelfs bij c=n de eis niet gehaald wordt).
    """
    c = 0
    while probability_of_acceptance(n, c, quality_pct, method, lot_size) < confidence:
        c += 1
        if c >= n:
            return n
    return c


def compute_thresholds(lot_size: int, n: int, method: str,
                        confidence: float = DEFAULT_CONFIDENCE) -> dict[str, int]:
    """Bereken de niveaudrempels (acceptatiegetal Ac) voor elk niveau.

    Args:
        lot_size: Totaal aantal cleans (lotgrootte N).
        n: Steekproefgrootte.
        method: "binomial" of "hypergeometric".
        confidence: Gewenste zekerheid bij AQL-kwaliteit.

    Returns:
        Dict met per niveau het acceptatiegetal Ac.
    """
    return {
        lv: find_acceptance_number(n, aql, method, confidence, lot_size=lot_size)
        for lv, aql in AQL.items()
    }


def determine_niveau(lot_size: int, aantal_afgekeurd: int, method: str,
                      level: int = DEFAULT_LEVEL,
                      confidence: float = DEFAULT_CONFIDENCE) -> tuple[str, int, dict[str, int]]:
    """Bepaal het AQL-niveau voor een gegeven maand.

    Args:
        lot_size: Totaal aantal cleans deze maand.
        aantal_afgekeurd: Rejectede accommodaties in de steekproef.
        method: "binomial" of "hypergeometric".
        level: BS EN 13549 sampling level (1, 2 of 3).
        confidence: Gewenste zekerheid bij het bepalen van de drempels.

    Returns:
        Tuple van (niveau, steekproefgrootte n, drempels per niveau).
    """
    n = get_sample_size(lot_size, level=level)
    thresholds = compute_thresholds(lot_size, n, method, confidence=confidence)

    niveau = "I"
    for lv in reversed(NIVEAUS):  # V, IV, III, II, I -- eerste match wint
        if aantal_afgekeurd <= thresholds[lv]:
            niveau = lv
            break

    return niveau, n, thresholds


def build_paq_plq_table(lot_size: int, n: int, thresholds: dict[str, int], method: str,
                         lq_multiplier: float) -> pd.DataFrame:
    """Bouw een tabel met AQL, Ac, PAQ, LQ en PLQ per niveau.

    LQ is hier een vast, zelf gekozen referentiepunt (LQ = AQL x
    lq_multiplier) -- net zoals AQL een vast referentiepunt is. PLQ is
    daarna puur het berekende resultaat (de acceptatiekans bij die LQ),
    en varieert dus vrij met n en de gekozen methode -- het wordt niet
    omgekeerd opgelost naar een vooraf vastgezet consumentrisico (dat
    zou PLQ tot een echo van een instelling maken in plaats van een
    output).

    Args:
        lot_size: Lotgrootte N (gebruikt bij method="hypergeometric").
        n: Steekproefgrootte.
        thresholds: Acceptatiegetal Ac per niveau.
        method: "binomial" of "hypergeometric".
        lq_multiplier: LQ = AQL x deze multiplier (het zelf gekozen
            "hoe slecht is te slecht"-referentiepunt).

    Returns:
        DataFrame met een rij per niveau.
    """
    rows = []
    for lv in NIVEAUS:
        aql = AQL[lv]
        ac = thresholds[lv]
        lq = aql * lq_multiplier
        paq = probability_of_acceptance(n, ac, aql, method, lot_size=lot_size)
        plq = probability_of_acceptance(n, ac, lq, method, lot_size=lot_size)
        rows.append(
            {
                "Level": lv,
                "AQL (%)": aql,
                "Ac": ac,
                "PAQ (%)": round(paq * 100, 1),
                "LQ (%)": round(lq, 2),
                "PLQ (%)": round(plq * 100, 1),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Logica: officiele referentietabellen -> DataFrame
# --------------------------------------------------------------------------
# Cellen in Tabel C.2/C.3 die vermoedelijk een fout bevatten in de bron
# (scan/OCR of een echte fout in de standaard) -- ze worden gemarkeerd met
# een voetnoot in plaats van stilzwijgend "gecorrigeerd", want de scan kan
# niet met zekerheid herlezen worden.
#
# (50, "10%") in TABLE_C2_NORMAL: gepubliceerde PLQ = 32,8%. Dat breekt het
# monotoon dalende PLQ-patroon binnen de 10%-kolom (21,2% bij n=32,
# 0,8% bij n=80), en de eigen herberekening met dezelfde LQ=30% die de rest
# van die kolom uitstekend reproduceert geeft hier 7,89% -- wat wel naadloos
# in de dalende reeks past. Zie build_comparison_df(), rij (50, "10%").
SUSPECT_CELLS_C2: set[tuple[int, str]] = {(50, "10%")}


def build_official_c2_c3_df(
    table: dict[int, dict[str, tuple[int, float, float] | None]],
    suspect_cells: set[tuple[int, str]] = frozenset(),
) -> pd.DataFrame:
    """Zet Tabel C.2 of C.3 (dict per steekproefgrootte/AQL) om naar een
    platte DataFrame die de lay-out van de standaard benadert.

    Args:
        table: TABLE_C2_NORMAL of TABLE_C3_TIGHTENED.
        suspect_cells: Set van (sample_size, aql_label) combinaties waarvan
            de PLQ vermoedelijk fout is in de bron. Die PLQ-waarde krijgt een
            "*"-voetnootmarkering in plaats van stilzwijgend gecorrigeerd te
            worden.

    Returns:
        DataFrame met een rij per steekproefgrootte en per AQL-kolom
        de Ac/PAQ/PLQ, of "-" waar de standaard niets definieert.
    """
    aql_labels = list(next(iter(table.values())).keys())
    rows = []
    for sample_size, per_aql in table.items():
        row: dict[str, object] = {"Sample size": sample_size}
        for aql_label in aql_labels:
            entry = per_aql[aql_label]
            if entry is None:
                row[f"{aql_label} Ac"] = "-"
                row[f"{aql_label} PAQ"] = "-"
                row[f"{aql_label} PLQ"] = "-"
            else:
                ac, paq, plq = entry
                row[f"{aql_label} Ac"] = ac
                row[f"{aql_label} PAQ"] = f"{paq:.2f}%".replace(".", ",")
                marker = "*" if (sample_size, aql_label) in suspect_cells else ""
                row[f"{aql_label} PLQ"] = f"{plq:.2f}%{marker}".replace(".", ",")
        rows.append(row)
    return pd.DataFrame(rows)


def build_official_c7_df() -> pd.DataFrame:
    """Zet Tabel C.7 (AQL = 7%) om naar een DataFrame.

    Returns:
        DataFrame met een rij per steekproefgrootte uit TABLE_C7_AQL_7.
        Ontbrekende steekproefgroottes (zie module-docstring bij
        TABLE_C7_AQL_7) staan er niet in.
    """
    rows = [
        {
            "Sample size": sample_size,
            "Ac": ac,
            "PAQ": f"{paq:.2f}%".replace(".", ","),
            "PLQ": f"{plq:.2f}%".replace(".", ","),
            "PAQ >": f"{paq_gt:.0f}%" if paq_gt is not None else "",
        }
        for sample_size, ac, paq, plq, paq_gt in TABLE_C7_AQL_7
    ]
    return pd.DataFrame(rows)


def _parse_aql_label(label: str) -> float:
    """'4%' -> 4.0, '6,5%' -> 6.5, '10%' -> 10.0."""
    return float(label.replace("%", "").replace(",", "."))


def _check_symbol(diff_pp: float, tolerance: float = 0.15) -> str:
    """Geeft een symbool terug voor gebruik IN een dataframe-cel.

    Let op: `:material/...:`-shortcodes worden alleen gerenderd door
    Streamlit's eigen markdown/tekstcomponenten (st.markdown, st.button,
    tab-labels, etc.), niet binnen `st.dataframe`-cellen -- daar verschijnt
    de kale shortcode-tekst. Vandaar hier een echt unicode-symbool.
    """
    return "\u2705" if abs(diff_pp) <= tolerance else "\u26a0\ufe0f"


def build_comparison_df(
    table: dict[int, dict[str, tuple[int, float, float] | None]],
    lq_targets: dict[str, float],
    method: str = "binomial",
) -> pd.DataFrame:
    """Herberekent PAQ en PLQ met de eigen `probability_of_acceptance()` op
    basis van de officiele (n, Ac) uit Tabel C.2/C.3, en vergelijkt beide
    met de gepubliceerde waarden.

    PAQ berekend = probability_of_acceptance(n, Ac, AQL) -- AQL komt direct
    uit de tabel, dus dit is een ondubbelzinnige vergelijking.

    PLQ berekend = probability_of_acceptance(n, Ac, lq_targets[AQL]) -- exact
    dezelfde berekening als PAQ, alleen met een ander doel-kwaliteitspercentage
    per AQL-kolom. De standaard vermeldt zelf niet welke LQ bij de PLQ-kolom
    hoort; door per AQL-kolom een eigen `lq_targets`-waarde mee te geven kun
    je die apart instellen (elke AQL-kolom heeft immers zijn eigen impliciete
    LQ, ze schalen niet allemaal met dezelfde factor).

    Args:
        table: TABLE_C2_NORMAL of TABLE_C3_TIGHTENED.
        lq_targets: Dict van AQL-label (bijv. "4%", "6,5%", "10%") naar het
            doel-kwaliteitspercentage (LQ) waarop PLQ voor die kolom getoetst
            wordt.
        method: "binomial" (standaard -- deze tabellen zijn
            lotgrootte-onafhankelijk) of "hypergeometric".

    Returns:
        Long-format DataFrame: een rij per (steekproefgrootte, AQL)
        combinatie waarvoor Tabel C.2/C.3 een Ac geeft.
    """
    rows = []
    for sample_size, per_aql in table.items():
        for aql_label, entry in per_aql.items():
            if entry is None:
                continue
            ac, paq_off, plq_off = entry
            aql_value = _parse_aql_label(aql_label)
            lq_target_pct = lq_targets[aql_label]

            paq_calc = (
                probability_of_acceptance(sample_size, ac, aql_value, method, lot_size=sample_size)
                * 100
            )
            diff_paq = paq_calc - paq_off

            plq_calc = (
                probability_of_acceptance(sample_size, ac, lq_target_pct, method, lot_size=sample_size)
                * 100
            )
            diff_plq = plq_calc - plq_off

            rows.append(
                {
                    "Sample size": sample_size,
                    "AQL": aql_label,
                    "Ac": ac,
                    "PAQ officieel": f"{paq_off:.2f}%".replace(".", ","),
                    "PAQ berekend": f"{paq_calc:.2f}%".replace(".", ","),
                    "Δ PAQ (pp)": round(diff_paq, 2),
                    "Check PAQ": _check_symbol(diff_paq),
                    "PLQ officieel": f"{plq_off:.2f}%".replace(".", ","),
                    "PLQ berekend": f"{plq_calc:.2f}%".replace(".", ","),
                    "Δ PLQ (pp)": round(diff_plq, 2),
                    "Check PLQ": _check_symbol(diff_plq),
                }
            )
    return pd.DataFrame(rows)


def build_comparison_c7_df(lq_target_pct: float, method: str = "binomial") -> pd.DataFrame:
    """Zelfde vergelijking als `build_comparison_df()`, maar voor Tabel C.7
    (vast AQL = 7 %). Zie die functie voor de uitleg per kolom.

    Args:
        lq_target_pct: Doel-kwaliteitspercentage (LQ) waarop PLQ getoetst
            wordt.
        method: "binomial" (standaard) of "hypergeometric".
    """
    rows = []
    for sample_size, ac, paq_off, plq_off, _paq_gt in TABLE_C7_AQL_7:
        paq_calc = (
            probability_of_acceptance(sample_size, ac, 7.0, method, lot_size=sample_size) * 100
        )
        diff_paq = paq_calc - paq_off

        plq_calc = (
            probability_of_acceptance(sample_size, ac, lq_target_pct, method, lot_size=sample_size)
            * 100
        )
        diff_plq = plq_calc - plq_off

        rows.append(
            {
                "Sample size": sample_size,
                "Ac": ac,
                "PAQ officieel": f"{paq_off:.2f}%".replace(".", ","),
                "PAQ berekend": f"{paq_calc:.2f}%".replace(".", ","),
                "Δ PAQ (pp)": round(diff_paq, 2),
                "Check PAQ": _check_symbol(diff_paq),
                "PLQ officieel": f"{plq_off:.2f}%".replace(".", ","),
                "PLQ berekend": f"{plq_calc:.2f}%".replace(".", ","),
                "Δ PLQ (pp)": round(diff_plq, 2),
                "Check PLQ": _check_symbol(diff_plq),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("AQL Level & Bonus/Malus", text_alignment="left")
st.caption(
    "Sample size n via BS EN 13549 Table C.1 (Level 2). "
    "PAQ = probability of acceptance at the AQL (not a fixed 95%, that's just a rule of thumb). "
    "LQ = a self-chosen 'too bad' reference point (AQL x multiplier); "
    "PLQ = the calculated probability of acceptance at that point."
)

with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4, vertical_alignment="bottom")
    with col1:
        level = st.segmented_control("Sampling level", options=[1, 2, 3], default=DEFAULT_LEVEL)
    with col2:
        method = st.segmented_control(
            "Method", options=["binomial", "hypergeometric"], default="binomial"
        )
    with col3:
        confidence_pct = st.slider(
            "Confidence at AQL (%)", min_value=80, max_value=99, value=98, step=1,
            help="Probability that a lot exactly at the AQL level is still accepted."
        )
    with col4:
        lq_multiplier = st.slider(
            "LQ = AQL x", min_value=1.5, max_value=5.0, value=DEFAULT_LQ_MULTIPLIER, step=0.5,
            help="Self-chosen 'how bad is too bad' reference point. PLQ (the probability "
                 "of acceptance at this LQ) is calculated from it, not the other way around."
        )

level = level or DEFAULT_LEVEL
method = method or "hypergeometric"
confidence = confidence_pct / 100

if method == "binomial":
    st.caption(
        ":material/info: Binomial: independent of lot size -- comparable "
        "to the official Table C.2 in BS EN 13549 (verified against the "
        "published PAQ values)."
    )
else:
    st.caption(
        ":material/info: Hypergeometric: takes into account the exact, "
        "finite lot size (number of cleans this month)."
    )

tab_calc, tab_table, tab_official, tab_uitleg = st.tabs(
    [
        ":material/calculate: Calculator",
        ":material/table: Reference table",
        ":material/verified: Official tables",
        ":material/campaign: Convince management",
    ]
)

with tab_calc:
    with st.container(border=True):
        col1, col2 = st.columns(2, vertical_alignment="bottom")
        with col1:
            aantal_cleans = st.number_input(
                "Number of cleans this month", min_value=2, max_value=600_000_000, value=1000, step=50
            )
        with col2:
            aantal_afgekeurd = st.number_input(
                "Number of rejected accommodations (in the sample)", min_value=0, value=0, step=1
            )

    niveau, n, thresholds = determine_niveau(
        int(aantal_cleans), int(aantal_afgekeurd), method, level=level, confidence=confidence
    )
    pct = round(100 * aantal_afgekeurd / n, 2) if n else 0.0

    st.space("small")
    with st.container(horizontal=True, border=True):
        st.metric("Checks (n)", n)
        st.metric("Rejected", f"{int(aantal_afgekeurd)} / {n}")
        st.metric("Percentage", f"{pct}%")
        st.metric("Level", niveau)
        st.metric("Bonus/malus", f"{'+' if BONUS_MALUS[niveau] > 0 else ''}{BONUS_MALUS[niveau]}%")

    st.space("small")
    color = "green" if BONUS_MALUS[niveau] > 0 else ("gray" if BONUS_MALUS[niveau] == 0 else "red")
    st.markdown(
        f":{color}-badge[Level {niveau} — AQL {AQL[niveau]}% — "
        f"{'+' if BONUS_MALUS[niveau] > 0 else ''}{BONUS_MALUS[niveau]}% bonus/malus]"
    )
    st.caption(
        f"For {int(aantal_cleans)} cleans (level {level}, {method}), the correct sample "
        f"size is {n} checks. Threshold for level {niveau} (at {confidence_pct}% confidence): "
        f"max {thresholds[niveau]} rejected. With {int(aantal_afgekeurd)} rejected "
        f"({pct}%), this falls within level {niveau}."
    )

    st.space("medium")
    st.markdown("**PAQ / PLQ per level (for this sample)**")
    paq_plq_df = build_paq_plq_table(int(aantal_cleans), n, thresholds, method, lq_multiplier)
    st.dataframe(
        paq_plq_df,
        hide_index=True,
        width="stretch",
        column_config={
            "PAQ (%)": st.column_config.ProgressColumn("PAQ (%)", min_value=0, max_value=100, format="%.1f%%"),
            "PLQ (%)": st.column_config.ProgressColumn("PLQ (%)", min_value=0, max_value=100, format="%.1f%%"),
        },
    )
    st.caption(
        "High PAQ = the producer is well protected (a good lot is rarely rejected). "
        "Low PLQ = the consumer is well protected (a bad lot is rarely accepted)."
    )

with tab_table:
    st.markdown("**AQL per level and bonus/malus**")
    header_df = pd.DataFrame(
        [
            [f"{AQL[lv]}%" for lv in NIVEAUS],
            [f"{'+' if BONUS_MALUS[lv] > 0 else ''}{BONUS_MALUS[lv]}%" for lv in NIVEAUS],
        ],
        columns=[f"Level {lv}" for lv in NIVEAUS],
        index=["AQL", "Bonus/malus"],
    )
    st.dataframe(header_df, width="stretch")

    st.markdown(f"**Calculated thresholds per number of cleans (Level {level}, {method}, {confidence_pct}% confidence)**")
    rows = []
    for cleans in VOORBEELD_CLEANS:
        n_ref = get_sample_size(cleans, level=level)
        th = compute_thresholds(cleans, n_ref, method, confidence=confidence)
        rows.append([cleans, n_ref] + [th[lv] for lv in NIVEAUS])
    table_df = pd.DataFrame(
        rows,
        columns=["Number of cleans", "Number of checks"] + [f"Level {lv} (<=)" for lv in NIVEAUS],
    )
    st.dataframe(
        table_df,
        hide_index=True,
        width="stretch",
        column_config={"Number of cleans": st.column_config.NumberColumn(pinned=True)},
    )
    st.caption(
        "These rows are for illustration -- the calculator computes n and the "
        "thresholds for any number of cleans, not just these examples."
    )

    st.space("medium")
    st.markdown("**PAQ / PLQ per number of cleans and level**")
    paq_plq_rows = []
    for cleans in VOORBEELD_CLEANS:
        n_ref = get_sample_size(cleans, level=level)
        th = compute_thresholds(cleans, n_ref, method, confidence=confidence)
        for lv in NIVEAUS:
            aql = AQL[lv]
            ac = th[lv]
            lq = aql * lq_multiplier
            paq = probability_of_acceptance(n_ref, ac, aql, method, lot_size=cleans)
            plq = probability_of_acceptance(n_ref, ac, lq, method, lot_size=cleans)
            paq_plq_rows.append(
                {
                    "Number of cleans": cleans,
                    "n": n_ref,
                    "Level": lv,
                    "AQL (%)": aql,
                    "Ac": ac,
                    "PAQ (%)": round(paq * 100, 1),
                    "LQ (%)": round(lq, 2),
                    "PLQ (%)": round(plq * 100, 1),
                }
            )
    paq_plq_ref_df = pd.DataFrame(paq_plq_rows)
    st.dataframe(
        paq_plq_ref_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Number of cleans": st.column_config.NumberColumn(pinned=True),
            "PAQ (%)": st.column_config.ProgressColumn("PAQ (%)", min_value=0, max_value=100, format="%.1f%%"),
            "PLQ (%)": st.column_config.ProgressColumn("PLQ (%)", min_value=0, max_value=100, format="%.1f%%"),
        },
    )

with tab_official:
    st.markdown(
        "The gray tables are **taken literally** from BS EN 13549:2001, "
        "Annex C. Below each table is the **app's own calculation**: the same "
        "`probability_of_acceptance()` function used elsewhere in the app, once "
        "tested against AQL (-> PAQ) and once against a self-set LQ percentage "
        "(-> PLQ) -- exactly the same calculation, just with a different target percentage."
    )

    st.markdown("**LQ per AQL column (target quality percentage %) for Table C.2 and C.3**")
    st.caption(
        "Each AQL column has its own implicit LQ -- they don't all scale by "
        "the same factor -- so each column below has its own adjustable value. "
        "PLQ calculated = probability_of_acceptance(n, Ac, this percentage)."
    )
    col_lq1, col_lq2, col_lq3 = st.columns(3)
    with col_lq1:
        lq_target_4 = st.number_input(
            "LQ at AQL 4%", min_value=0.1, max_value=100.0, value=12.0, step=0.1, key="lq_target_4"
        )
    with col_lq2:
        lq_target_65 = st.number_input(
            "LQ at AQL 6.5%", min_value=0.1, max_value=100.0, value=19.5, step=0.1, key="lq_target_65"
        )
    with col_lq3:
        lq_target_10 = st.number_input(
            "LQ at AQL 10%", min_value=0.1, max_value=100.0, value=30.0, step=0.1, key="lq_target_10"
        )
    lq_targets_c2c3 = {"4%": lq_target_4, "6,5%": lq_target_65, "10%": lq_target_10}

    st.markdown("**Table C.2 -- Single sampling: normal inspection**")
    st.dataframe(
        build_official_c2_c3_df(TABLE_C2_NORMAL, suspect_cells=SUSPECT_CELLS_C2),
        hide_index=True,
        width="stretch",
        column_config={"Sample size": st.column_config.NumberColumn(pinned=True)},
    )
    st.caption(
        "\\* n=50, 10%: the published PLQ (32.80%) breaks the monotonically "
        "decreasing pattern within that column (21.20% at n=32, 0.80% at n=80) "
        "and deviates strongly from the app's own recalculation (7.89% at "
        "LQ=30%, see expander below) -- likely an error in the scan/standard, "
        "not corrected but flagged."
    )
    comp_c2 = build_comparison_df(TABLE_C2_NORMAL, lq_targets=lq_targets_c2c3)
    max_diff_paq_c2 = comp_c2["Δ PAQ (pp)"].abs().max()
    max_diff_plq_c2 = comp_c2["Δ PLQ (pp)"].abs().max()
    with st.expander(
        f":material/function: Own calculation vs. official "
        f"(largest deviation PAQ: {max_diff_paq_c2:.2f} pp, PLQ: {max_diff_plq_c2:.2f} pp)"
    ):
        st.dataframe(
            comp_c2,
            hide_index=True,
            width="stretch",
            column_config={
                "Check PAQ": st.column_config.TextColumn(),
                "Check PLQ": st.column_config.TextColumn(),
                "Δ PAQ (pp)": st.column_config.NumberColumn(format="%.2f"),
                "Δ PLQ (pp)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "PAQ calculated = `probability_of_acceptance(n, Ac, AQL, 'binomial')` "
            "using the official (n, Ac, AQL) above -- an unambiguous comparison. "
            "PLQ calculated = the same function with LQ = the adjustable percentage per "
            "AQL column above, instead of AQL. Set each LQ so that Δ PLQ for "
            "that column is small everywhere, and you'll know which LQ the "
            "standard implicitly used for that column."
        )

    st.space("medium")
    st.markdown("**Table C.3 -- Single sampling: tightened inspection**")
    st.dataframe(
        build_official_c2_c3_df(TABLE_C3_TIGHTENED),
        hide_index=True,
        width="stretch",
        column_config={"Sample size": st.column_config.NumberColumn(pinned=True)},
    )
    comp_c3 = build_comparison_df(TABLE_C3_TIGHTENED, lq_targets=lq_targets_c2c3)
    max_diff_paq_c3 = comp_c3["Δ PAQ (pp)"].abs().max()
    max_diff_plq_c3 = comp_c3["Δ PLQ (pp)"].abs().max()
    with st.expander(
        f":material/function: Own calculation vs. official "
        f"(largest deviation PAQ: {max_diff_paq_c3:.2f} pp, PLQ: {max_diff_plq_c3:.2f} pp)"
    ):
        st.dataframe(
            comp_c3,
            hide_index=True,
            width="stretch",
            column_config={
                "Check PAQ": st.column_config.TextColumn(),
                "Check PLQ": st.column_config.TextColumn(),
                "Δ PAQ (pp)": st.column_config.NumberColumn(format="%.2f"),
                "Δ PLQ (pp)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "See the explanation for Table C.2 above -- same method and "
            "same LQ-per-AQL-column settings, now applied to the "
            "tightened-inspection rows."
        )


    st.space("medium")
    st.markdown("**Table C.7 -- Conformity index table for AQL = 7 %**")
    st.caption(
        ":material/warning: Incompletely transcribed: only the rows that were "
        "legible on the supplied page scan (page 19) are shown below. Sample sizes "
        "26-65, 93-130 and 159-195 are missing, and the rows for sample size 72 and 158 "
        "were not fully legible. Provide a scan of the remaining rows if you "
        "want the complete table."
    )
    st.dataframe(
        build_official_c7_df(),
        hide_index=True,
        width="stretch",
        column_config={"Sample size": st.column_config.NumberColumn(pinned=True)},
    )
    lq_target_c7 = st.number_input(
        "LQ (target quality percentage %) for Table C.7",
        min_value=7.1, max_value=100.0, value=21.0, step=0.1,
        help="AQL is fixed at 7% here; PLQ calculated = "
             "probability_of_acceptance(n, Ac, this percentage).",
        key="lq_target_c7",
    )
    comp_c7 = build_comparison_c7_df(lq_target_pct=lq_target_c7)
    max_diff_paq_c7 = comp_c7["Δ PAQ (pp)"].abs().max()
    max_diff_plq_c7 = comp_c7["Δ PLQ (pp)"].abs().max()
    with st.expander(
        f":material/function: Own calculation vs. official "
        f"(largest deviation PAQ: {max_diff_paq_c7:.2f} pp, PLQ: {max_diff_plq_c7:.2f} pp "
        f"at LQ={lq_target_c7:g}%, over {len(comp_c7)} rows)"
    ):
        st.dataframe(
            comp_c7,
            hide_index=True,
            width="stretch",
            column_config={
                "Sample size": st.column_config.NumberColumn(pinned=True),
                "Check PAQ": st.column_config.TextColumn(),
                "Check PLQ": st.column_config.TextColumn(),
                "Δ PAQ (pp)": st.column_config.NumberColumn(format="%.2f"),
                "Δ PLQ (pp)": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        st.caption(
            "Same method as for Table C.2/C.3, now with AQL fixed at 7%. "
            "For many rows, the official PLQ rounds to 0.0% -- there, "
            "Δ PLQ will almost always be small and negative once LQ is large enough; "
            "that's expected behavior, not an error."
        )

with tab_uitleg:
    st.markdown(
        "*Why a fixed inspection percentage isn't quality control, what "
        "manufacturing industry already figured out a hundred years ago, and "
        "why you should mainly use this to reward -- not to punish.*"
    )

    st.markdown("### Saturday afternoon, changeover day")
    st.markdown(
        "A mobile home passes the cleaning check. The cleaner is satisfied, "
        "the unit looks good. Two hours later the guest walks in -- and "
        "complains anyway.\n\n"
        "What happened? Somewhere between what the cleaner saw, what the guest "
        "expected, and what was actually on the surfaces, a gap opened up. "
        "And that gap has a shape worth understanding: "
        "**technically clean** (measurably free of dirt or germs), **looks clean** "
        "(fresh to a guest's eyes and nose), and **maintenance condition** "
        "(how worn the unit is) are three different circles. A "
        "spotless faucet in a discolored, damaged sink can be genuinely "
        "clean and still *look* dirty. Manage only one circle, and the "
        "other two will eventually bite you anyway."
    )
    st.markdown(
        "This tab explains why a measurable standard closes that gap -- "
        "and why a system that mainly *rewards* rather than punishes "
        "demonstrably works better than a vague seasonal threat."
    )

    st.markdown("---")
    st.markdown("### The problem with \"we inspect 25%\"")
    st.markdown(
        "Many cleaning contracts in hospitality run on a fixed rule of thumb: "
        "inspect a quarter of the units, and if it falls short \"something happens\". "
        "Nobody can say exactly what. Maybe a phone call. Maybe the contract "
        "quietly isn't renewed next year. Maybe nothing. "
        "That vagueness -- not the cleaning itself -- is the real failure factor."
    )
    st.markdown(
        "A fixed percentage is also statistically flawed: statistical "
        "reliability depends on the **absolute** number of units inspected, "
        "not on the share of the total. A fixed percentage therefore "
        "over-inspects large parks and under-inspects small ones -- exactly "
        "backwards from where the risk of a non-representative sample is "
        "greatest."
    )
    ex_col1, ex_col2 = st.columns(2)
    with ex_col1:
        st.metric("20 units, 25% inspection", "5 checks", help="Margin of error around ±40% -- this tells you almost nothing.")
    with ex_col2:
        st.metric("400 units, 25% inspection", "100 checks", help="Far more confidence than needed -- expensive overkill.")
    st.caption(
        "Same rule, two completely different outcomes. That's not a "
        "coincidence, that's the math of a fixed percentage."
    )

    st.markdown("**The alternative: BS EN 13549 Table C.1**")
    st.markdown(
        "Instead of a fixed percentage, the standard gives a "
        "sample size tuned to the lot size itself -- this is "
        "literally `TABLE_C1` at the top of the code, the table that "
        "`get_sample_size()` uses for every calculation in this "
        "tool:"
    )
    _tabel_c1_df = pd.DataFrame(
        [
            {
                "Lot size (number of cleans)": (
                    f"{lo:,}+".replace(",", ".") if hi == TABLE_C1[-1][1]
                    else f"{lo:,}\u2013{hi:,}".replace(",", ".")
                ),
                "Level I": l1,
                "Level II": l2,
                "Level III": l3,
            }
            for lo, hi, l1, l2, l3 in TABLE_C1
        ]
    )
    st.dataframe(_tabel_c1_df, hide_index=True, width="stretch")
    st.caption(
        "Compare this with the metrics above: for 20 cleans (lot size "
        "16-25) the table gives n=5 at Level II -- coincidentally the same as the 25%"
        " rule of thumb, purely because this lot is small. For 400 cleans (lot size "
        "281-500) it gives n=50, not the 100 that 25% would produce. So the "
        "sample size does grow with the lot, but far "
        "slower than the lot itself -- and that's exactly the point."
    )

    with st.container(border=True):
        st.markdown("**What is this table based on?**")
        st.markdown(
            "The lot-size boundaries and sample sizes in Table C.1 were not "
            "invented specifically for cleaning -- they are almost literally the "
            "lot-size code letters and sample sizes from **MIL-STD-105E** "
            "(the American military attribute-sampling standard from the "
            "1950s), as carried over into civilian use in **ANSI/ASQ Z1.4** and "
            "mirrored internationally in **ISO 2859-1**. BS EN 13549 adopted that "
            "generic, decades-proven industry standard "
            "for the cleaning sector, rather than inventing new math "
            "of its own."
        )
        st.markdown(
            "Two design choices are baked into those numbers:\n\n"
            "- **Sample size grows roughly with a power of lot size, well "
            "below linear -- but that's an empirical pattern in the "
            "numbers, not a textbook formula.** There's no rigorous 1/\u221an "
            "derivation being applied here; what you *can* see directly in "
            "the table's own numbers is this: take the midpoint of each "
            "lot-size bracket, take the log of that midpoint and the log of "
            "its matching sample size, and plot one against the other. The "
            "points fall close to a straight line (chart below). A straight "
            "line in log-log space means a power law, sample size \u221d lot "
            "size^k, and fitting that line gives the exponent k below -- in "
            "the same ballpark as (though not exactly) the k=0.5 of a "
            "square-root law. A fixed percentage is the opposite: sample "
            "size = 0.25 \u00d7 lot size is k=1, a straight line only on a "
            "regular (non-log) plot -- which is precisely why it produces a "
            "different, arbitrary reliability at every lot size.\n"
            "- **The sequence of sample sizes itself (2, 3, 5, 8, 13, 20, 32, 50, "
            "80, 125, 200, 315, 500, 800, 1250, 2000) grows by roughly "
            "the same factor at each step (~1.6x).** That's a classic "
            "*preferred number series* (similar to the Renard R5 series), "
            "chosen so that every lot-size threshold gives a manageable, discrete "
            "number of \"code letters\" instead of a continuous formula "
            "that spits out a different number for every lot -- exactly how "
            "MIL-STD-105E was originally meant to be looked up by hand, "
            "without a calculator."
        )

        _c1_rows_for_fit = TABLE_C1[:-1]  # drop the open-ended "500,001+" bracket
        _c1_mids = [(lo + hi) / 2 for lo, hi, l1, l2, l3 in _c1_rows_for_fit]
        _c1_n2 = [l2 for lo, hi, l1, l2, l3 in _c1_rows_for_fit]
        _c1_log_mid = [math.log10(m) for m in _c1_mids]
        _c1_log_n2 = [math.log10(v) for v in _c1_n2]
        _c1_mean_x = sum(_c1_log_mid) / len(_c1_log_mid)
        _c1_mean_y = sum(_c1_log_n2) / len(_c1_log_n2)
        _c1_cov = sum((x - _c1_mean_x) * (y - _c1_mean_y) for x, y in zip(_c1_log_mid, _c1_log_n2))
        _c1_var = sum((x - _c1_mean_x) ** 2 for x in _c1_log_mid)
        _c1_slope = _c1_cov / _c1_var
        _c1_intercept = _c1_mean_y - _c1_slope * _c1_mean_x
        _c1_ss_res = sum((y - (_c1_slope * x + _c1_intercept)) ** 2 for x, y in zip(_c1_log_mid, _c1_log_n2))
        _c1_ss_tot = sum((y - _c1_mean_y) ** 2 for y in _c1_log_n2)
        _c1_r2 = 1 - _c1_ss_res / _c1_ss_tot

        _c1_fit_df = pd.DataFrame({"Lot size (bracket midpoint)": _c1_mids, "Sample size (Level II)": _c1_n2})
        _c1_line_x = [_c1_mids[0], _c1_mids[-1]]
        _c1_line_y = [10 ** (_c1_slope * math.log10(x) + _c1_intercept) for x in _c1_line_x]
        _c1_line_df = pd.DataFrame({"Lot size (bracket midpoint)": _c1_line_x, "Sample size (Level II)": _c1_line_y})

        _c1_scatter = (
            alt.Chart(_c1_fit_df)
            .mark_circle(size=70, color="#1F4B49")
            .encode(
                x=alt.X("Lot size (bracket midpoint):Q", scale=alt.Scale(type="log"),
                        title="Lot size, bracket midpoint (log scale)"),
                y=alt.Y("Sample size (Level II):Q", scale=alt.Scale(type="log"),
                        title="Sample size n, Level II (log scale)"),
            )
        )
        _c1_fit_line = (
            alt.Chart(_c1_line_df)
            .mark_line(color="#B4863B", strokeDash=[6, 4], strokeWidth=2)
            .encode(
                x=alt.X("Lot size (bracket midpoint):Q", scale=alt.Scale(type="log")),
                y=alt.Y("Sample size (Level II):Q", scale=alt.Scale(type="log")),
            )
        )
        st.altair_chart((_c1_scatter + _c1_fit_line).properties(height=320), use_container_width=True)
        st.caption(
            "Each dot is one lot-size bracket from Table C.1 (Level II "
            "column): x = the midpoint of that bracket's lot-size range, "
            "y = the matching sample size, both on a log scale. Fitting a "
            f"straight line through log(x) vs log(y) gives slope k\u2248{_c1_slope:.2f} "
            f"(R\u00b2\u2248{_c1_r2:.2f}, computed live from `TABLE_C1` -- not "
            "hardcoded) -- in the same ballpark as, but not exactly, the "
            "k=0.5 of a pure square-root law. That's the actual, "
            "empirically-visible regularity in these numbers: not a "
            "formula the standard's authors necessarily derived from "
            "sampling-error theory, but a consistent sub-linear growth "
            "pattern baked into the code-letter table they built."
        )
        st.markdown(
            "The three levels (I, II, III) are the generic *inspection "
            "levels* from that same scheme: **Level II is the standard** "
            "(normal stringency, `DEFAULT_LEVEL = 2` in the code), **Level I** "
            "is a relaxed regime with smaller samples for situations "
            "with lower risk or a good track record, and **Level III** "
            "tightens things up with larger samples -- for example after a "
            "string of rejections. You can switch this yourself in the "
            "Calculator tab."
        )

    st.markdown("### What manufacturing industry already solved a hundred years ago")
    st.markdown(
        "Walter Shewhart laid the statistical foundation for process control "
        "in the 1920s at Bell Telephone Laboratories. Harold Dodge and Harry Romig "
        "built practical sampling tables on top of it: instead of inspecting every unit, "
        "you take a sample that is statistically matched to the "
        "*lot size*, and you accept or reject the whole lot based on a "
        "pre-agreed defect limit. Those tables became the military standard "
        "MIL-STD-105E, later adapted for civilian use as ANSI/ASQ Z1.4, and "
        "mirrored internationally as **ISO 2859** -- still the reference "
        "framework behind quality control in industry today."
    )
    st.markdown(
        "The cleaning sector has its own, sector-specific derivative "
        "of that: **BS EN 13549:2001**, a European standard that gives exact "
        "sample sizes per inspection level and lot size, with "
        "acceptance thresholds tuned to the higher natural variation of "
        "manual cleaning work compared to a factory process. The detail that "
        "matters most: sample size follows from the total lot size, "
        "not from a fixed percentage of it -- exactly what the "
        "**Calculator** tab above works out for you."
    )

    with st.container(border=True):
        st.markdown("**📋 The standards at a glance**")
        standards_overview = pd.DataFrame(
            [
                {"Standard": "ISO 22483:2020", "What it is": "Hotel service standard",
                 "What it gets you": "Requires a documented cleaning plan (public areas, occupied rooms, departure cleaning, deep cleaning, laundry)"},
                {"Standard": "EN 13549:2001", "What it is": "European base standard",
                 "What it gets you": "Defines HOW you measure cleaning quality -- sampling, assessment, follow-up"},
                {"Standard": "NEN 2075 / VSR-KMS 3", "What it is": "Dutch measurement system",
                 "What it gets you": "The practical inspection tool; scales up to EN 13549; works for both effort-based and result-based contracts"},
                {"Standard": "ISO 2859", "What it is": "Statistical sampling",
                 "What it gets you": "The math (attribute sampling) that the tables in this app are built on"},
            ]
        )
        st.dataframe(standards_overview, hide_index=True, width="stretch")

    st.markdown("### From individual defect to a single number: demerit scoring")
    st.markdown(
        "Weighing every defect equally -- or worse, stamping a unit simply "
        "\"clean\"/\"not clean\" -- throws away information that a "
        "well-designed system doesn't have to throw away. The standard "
        "technique is called a *demerit rating system*: assign each defect "
        "class a weight, sum the weighted counts into a single statistic, and "
        "compare it against a threshold. This is documented, peer-reviewed "
        "methodology for monitoring complex products with multiple "
        "simultaneous defect types (*Journal of Quality Technology*, "
        "\"Exact Properties of Demerit Control Charts\"), and similar "
        "weighted-severity schemes turn up repeatedly in patented industrial "
        "inspection systems -- a reasonable sign of how standard the "
        "underlying idea has become."
    )
    st.markdown(
        "Translated to a single accommodation: **fatal** (wrong or forgotten unit), "
        "**major** (mold, stains, a wet bathroom, leftover food), "
        "**minor** (a few spots, a light layer of dust). One rule codes "
        "several policy lines at once -- one fatal defect is enough, or "
        "several major ones, or a pile-up of minor ones -- and treats every "
        "weighted combination consistently, without a separate rule per case."
    )

    with st.container(border=True):
        st.markdown("**Scoring rule used in this tool**")
        st.markdown(
            "`score = fatal x 9 + major x 3 + minor x 1`, **threshold = 9 points** "
            "-- a score above 9 means rejection."
        )
        st.caption(
            "In other words: a lone fatal defect already scores exactly 9 "
            "points on its own, so combined with virtually anything else "
            "(even a single minor spot) the unit is rejected. A handful of "
            "major defects (4 x 3 = 12) or a large pile-up of minor ones "
            "(10 x 1 = 10) crosses the threshold on their own too, just like "
            "the underlying demerit-scoring literature above intends: no "
            "defect type is capped, but each carries a different weight."
        )
        _demerit_examples_df = pd.DataFrame(
            [
                {"Class": "Fatal", "Weight": 9, "Example": "House forgotten", "Note": ""},
                {"Class": "Fatal", "Weight": 9, "Example": "House very badly done", "Note": ""},
                {"Class": "Major", "Weight": 3, "Example": "Significant stains", "Note": "on surfaces/walls"},
                {"Class": "Major", "Weight": 3, "Example": "Significant number of crumbs", "Note": "e.g. in cutlery drawer or on surfaces"},
                {"Class": "Major", "Weight": 3, "Example": "Stains visible from any angle, larger than a 10-cent coin", "Note": ""},
                {"Class": "Major", "Weight": 3, "Example": "Clothing left behind", "Note": ""},
                {"Class": "Major", "Weight": 3, "Example": "Floor very dirty, visible stuff on floor", "Note": ""},
                {"Class": "Major", "Weight": 3, "Example": "Leftover food on kitchen equipment", "Note": ""},
                {"Class": "Minor", "Weight": 1, "Example": "Small spots/stains", "Note": ""},
                {"Class": "Minor", "Weight": 1, "Example": "A few crumbs", "Note": ""},
                {"Class": "Minor", "Weight": 1, "Example": "Stains visible from only 1 angle", "Note": ""},
                {"Class": "Minor", "Weight": 1, "Example": "Little dust", "Note": ""},
                {"Class": "Minor", "Weight": 1, "Example": "Stains on kitchen equipment", "Note": ""},
            ]
        )
        st.dataframe(_demerit_examples_df, hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("### Why you should mainly reward, not punish")
    st.markdown(
        "This is where behavioral science and the statistical method "
        "meet -- and where the real argument for management lies."
    )

    reward_col1, reward_col2 = st.columns(2)
    with reward_col1:
        with st.container(border=True):
            st.markdown("**⏱️ Why monthly, not seasonal**")
            st.markdown(
                "Consequences change behavior only if they're close in time, "
                "proportional, and predictable. This isn't intuition, it's "
                "the core of *delay-discounting* research in "
                "behavioral economics: the subjective weight of a future "
                "reward or punishment drops sharply as the delay before it "
                "increases (Ainslie, 1975; generalized by Loewenstein & "
                "Prelec, 1992). A malus that only lands next season -- if "
                "the contract even comes up for renewal -- means almost "
                "nothing to someone cleaning a unit on Tuesday morning. A "
                "monthly cycle compresses that feedback loop to a length "
                "people can actually see through."
            )
    with reward_col2:
        with st.container(border=True):
            st.markdown("**⚖️ Why rewarding mainly works**")
            st.markdown(
                "Kahneman & Tversky's *prospect theory* (1979) shows that "
                "a loss weighs psychologically heavier than an equally sized "
                "gain (*loss aversion*) -- so a malus hurts more per euro than "
                "an equally sized bonus feels good. That means you don't need "
                "harsh penalties to steer behavior: a modest but "
                "reliable malus, combined with a generous and frequent "
                "bonus, steers the same behavior with far less friction and "
                "without turning the supplier against you."
            )

    st.markdown(
        "Elinor Ostrom (Nobel laureate, *Governing the Commons*, 1990) "
        "identified **graduated sanctions** -- consequences that scale "
        "with the severity and frequency of a violation, instead of "
        "immediately applying the harshest penalty -- as one of the "
        "recurring design principles behind institutions that last. \"Keep the "
        "contract or lose it\" is the opposite of graduated: an "
        "all-or-nothing threat so disproportionate to one bad "
        "week that nobody takes it seriously. The principle works "
        "symmetrically: five levels instead of a binary pass/fail applies "
        "just as much to the bonus side as to the malus side -- and the "
        "bonus side is where management should pull hardest, because "
        "that's the side that builds the relationship with your cleaning "
        "partner instead of eroding it."
    )
    st.markdown(
        "Joseph Juran's *cost-of-quality* framework treats quality "
        "deviations as a demonstrable, quantifiable financial impact, and "
        "argues that making that cost explicit -- rather than absorbing it "
        "invisibly or litigating it after the fact -- is itself a steering "
        "tool. A financial adjustment that follows directly from a "
        "reproducible measurement is an application of exactly that idea -- "
        "and a bonus that follows directly from that same measurement is the "
        "investment you earn back in less turnover, fewer disputes, and a "
        "cleaning partner aiming for the highest level instead of "
        "merely avoiding the lowest."
    )

    st.markdown("### This tool, in practice")
    st.markdown(
        f"The five levels this entire tool uses (**Calculator** "
        f"tab) are exactly this graduated scale, with AQLs and "
        f"bonus/malus percentages that you set yourself:"
    )
    live_tiers_df = pd.DataFrame(
        {
            "Level": NIVEAUS,
            "AQL": [f"{AQL[lv]:g}%" for lv in NIVEAUS],
            "Bonus/malus": [f"{'+' if BONUS_MALUS[lv] > 0 else ''}{BONUS_MALUS[lv]}%" for lv in NIVEAUS],
        }
    )
    st.dataframe(live_tiers_df, hide_index=True, width="stretch")
    st.caption(
        "These figures come directly from the AQL and BONUS_MALUS settings "
        "at the top of the code -- change them there, and this table here "
        "changes automatically too."
    )
    st.markdown(
        "The monthly cycle is simple: **(1)** draw a correctly calculated sample "
        "from this month's cleans (Calculator/Reference table tab, not "
        "a fixed percentage), **(2)** assess every inspected unit with the "
        "weighted defect rule, **(3)** compare the month's rejection rate "
        "with the five AQL levels above, **(4)** apply the corresponding bonus or "
        "malus to *that month*, **(5)** start the next month with a clean slate. The "
        "**Official tables** tab next to it proves that the calculation itself "
        "matches the published BS EN 13549 values, so you don't have to "
        "take that on faith."
    )

    with st.container(border=True):
        st.markdown(
            "**⚠️ Why an AQL does not mean you accept that percentage of "
            "dirty houses**"
        )
        st.markdown(
            "This is the most common misconception about this whole method, so "
            "it deserves its own explanation -- using only high-school math."
        )
        st.markdown(
            "You never inspect every house, only a sample. Compare "
            "it to a blood test: you don't test all your blood, just a small "
            "vial, and draw a conclusion about the rest from that. That works "
            "fine, but it also means a sample can never say with 100% certainty "
            "\"exactly this percentage of houses is bad\" -- only "
            "\"given what I saw in this sample, this-and-this is the "
            "most likely situation.\""
        )
        st.markdown(
            "That's why this works with two probabilities instead of one hard cutoff:\n\n"
            "- **PAQ** (*Probability of Acceptance at the AQL*) -- if the "
            "actual percentage of dirty houses sits exactly at the AQL threshold, "
            "how likely is it that the sample still passes it? You deliberately "
            "set that probability **high** (95-98%), in favor of the "
            "cleaning partner -- a party that's doing just well enough "
            "shouldn't be unfairly punished over and over for bad luck with "
            "the sample.\n"
            "- **PLQ** (*Probability of Acceptance at the LQ*) -- if the "
            "actual percentage is much worse than the AQL (the "
            "**LQ**, *Limiting Quality*: the level at which you say \"this "
            "really isn't acceptable\"), how likely is it that the sample "
            "*still* passes it? You deliberately set that probability **low** "
            "(5-10%) -- only then can you confidently say you're sure."
        )
        st.markdown(
            "Between those two points lies a gradual transition, not a "
            "hard cutoff. Adjust the inputs below yourself to see how the "
            "example changes -- the options are the values this "
            "tool already uses: the sample sizes from Table C.1/C.2, "
            "the five built-in AQL levels, and the confidence slider from the "
            "Calculator tab."
        )

        _uitleg_n_opties = sorted(TABLE_C2_NORMAL.keys())
        _uitleg_aql_opties = sorted(set(AQL.values()))
        _uitleg_conf_opties = [90, 95, 98, 99]

        _uitleg_col_in1, _uitleg_col_in2, _uitleg_col_in3 = st.columns(3)
        with _uitleg_col_in1:
            _uitleg_n = st.selectbox(
                "Sample size n",
                _uitleg_n_opties,
                index=_uitleg_n_opties.index(20) if 20 in _uitleg_n_opties else 0,
                key="uitleg_n_select",
            )
        with _uitleg_col_in2:
            _uitleg_aql = st.selectbox(
                "AQL",
                _uitleg_aql_opties,
                index=_uitleg_aql_opties.index(6.5) if 6.5 in _uitleg_aql_opties else 0,
                format_func=lambda v: f"{v:g}%".replace(".", ","),
                key="uitleg_aql_select",
            )
        with _uitleg_col_in3:
            _uitleg_conf_pct = st.selectbox(
                "Confidence",
                _uitleg_conf_opties,
                index=_uitleg_conf_opties.index(98) if 98 in _uitleg_conf_opties else 0,
                format_func=lambda v: f"{v}%",
                key="uitleg_conf_select",
            )

        _uitleg_confidence = _uitleg_conf_pct / 100
        _uitleg_ac = find_acceptance_number(
            _uitleg_n, _uitleg_aql, method="binomial", confidence=_uitleg_confidence
        )
        _uitleg_aql_label = f"{_uitleg_aql:g}%".replace(".", ",")
        _uitleg_ruleofthumb = _uitleg_n * _uitleg_aql / 100

        _uitleg_metric_col1, _uitleg_metric_col2 = st.columns(2)
        with _uitleg_metric_col1:
            st.metric(f"Acceptance number Ac (n={_uitleg_n}, AQL={_uitleg_aql_label}, {_uitleg_conf_pct}%)", _uitleg_ac)
        with _uitleg_metric_col2:
            st.metric(
                f"n × AQL = {_uitleg_n} × {_uitleg_aql_label}",
                f"{_uitleg_ruleofthumb:.2f}".replace(".", ","),
            )
        _uitleg_ruleofthumb_label = f"{_uitleg_ruleofthumb:.2f}".replace(".", ",")
        st.caption(
            f"n × AQL ({_uitleg_ruleofthumb_label}) is the old rule of thumb "
            "floor(n×AQL/100) that this tool used to use (see the "
            "version history at the top of the code) -- note that this is "
            f"**not** the same number as the real acceptance number Ac={_uitleg_ac} "
            "above. Ac follows from the binomial calculation with the chosen "
            "confidence level; the rule of thumb ignores that confidence entirely and "
            "always rounds down."
        )

        _uitleg_all_counts = list(range(0, _uitleg_n + 1))
        _uitleg_all_probs = [
            probability_of_acceptance(_uitleg_n, _uitleg_ac, c / _uitleg_n * 100, "binomial") * 100
            for c in _uitleg_all_counts
        ]
        # Start at the first row where the probability just dips below 99.95%
        # (the rows before that are all ~100% and add nothing) and stop at
        # the last row where the probability is still just above 0.05% (after
        # that it's all ~0%). Ac, PAQ (n x AQL) and PLQ (3 x n x AQL) always
        # stay visible, even if they fall further into the tail than these
        # thresholds.
        _uitleg_lq_count = DEFAULT_LQ_MULTIPLIER * _uitleg_ruleofthumb  # 3 x n x AQL / 100
        _uitleg_paq_row = max(0, min(_uitleg_n, round(_uitleg_ruleofthumb)))
        _uitleg_plq_row = max(0, min(_uitleg_n, round(_uitleg_lq_count)))

        _uitleg_start_idx = next(
            (i for i, p in enumerate(_uitleg_all_probs) if p < 99.95), 0
        )
        _uitleg_stop_idx = 0
        for i in range(len(_uitleg_all_probs) - 1, -1, -1):
            if _uitleg_all_probs[i] > 0.05:
                _uitleg_stop_idx = i
                break
        _uitleg_start_idx = min(_uitleg_start_idx, _uitleg_ac, _uitleg_paq_row, _uitleg_plq_row)
        _uitleg_stop_idx = max(_uitleg_stop_idx, _uitleg_ac, _uitleg_paq_row, _uitleg_plq_row)

        _uitleg_oc_counts = _uitleg_all_counts[_uitleg_start_idx:_uitleg_stop_idx + 1]
        _uitleg_oc_probs = _uitleg_all_probs[_uitleg_start_idx:_uitleg_stop_idx + 1]

        def _uitleg_markering(c: int) -> str:
            marks = []
            if c == _uitleg_ac:
                marks.append("Ac (acceptance number)")
            if c == _uitleg_paq_row:
                marks.append("n×AQL → PAQ")
            if c == _uitleg_plq_row:
                marks.append("3×n×AQL → PLQ")
            return "← " + " + ".join(marks) if marks else ""

        uitleg_oc_curve_df = pd.DataFrame(
            {
                "Number of dirty houses": _uitleg_oc_counts,
                "Corresponding %": [f"{c / _uitleg_n * 100:.1f}%".replace(".", ",") for c in _uitleg_oc_counts],
                "Probability of acceptance": [f"{p:.1f}%".replace(".", ",") for p in _uitleg_oc_probs],
                "Marker": [_uitleg_markering(c) for c in _uitleg_oc_counts],
            }
        )
        uitleg_oc_chart_df = pd.DataFrame(
            {
                "Number of dirty houses": _uitleg_oc_counts,
                "Probability of acceptance (%)": _uitleg_oc_probs,
            }
        )

        st.markdown(
            f"Concrete example using this tool's real functions: "
            f"a sample of **{_uitleg_n} houses**, with AQL={_uitleg_aql_label} "
            f"and {_uitleg_conf_pct}% confidence gives acceptance number "
            f"**Ac={_uitleg_ac}** (from `find_acceptance_number()`). This is how the "
            "probability of acceptance drops as the number of dirty houses in the "
            "sample rises (left), and this is how that acceptance number itself came "
            "about (right):"
        )

        _uitleg_chart_col1, _uitleg_chart_col2 = st.columns(2)

        with _uitleg_chart_col1:
            st.markdown("**OC curve: probability of acceptance per number of dirty houses**")
            st.dataframe(uitleg_oc_curve_df, hide_index=True, width="stretch", height=300)
            _aql_rule = pd.DataFrame({"x": [_uitleg_ruleofthumb], "label": ["n×AQL (PAQ)"]})
            _lq_rule = pd.DataFrame({"x": [_uitleg_lq_count], "label": ["3×n×AQL (PLQ)"]})
            _ac_rule = pd.DataFrame({"x": [_uitleg_ac], "label": [f"Ac = {_uitleg_ac}"]})
            _oc_line = (
                alt.Chart(uitleg_oc_chart_df)
                .mark_line(color="#1F4B49", strokeWidth=2.5)
                .encode(
                    x=alt.X("Number of dirty houses:Q", title=f"Number of dirty houses (out of n={_uitleg_n})",
                            scale=alt.Scale(domain=[_uitleg_oc_counts[0], _uitleg_oc_counts[-1]])),
                    y=alt.Y("Probability of acceptance (%):Q", title="Probability of acceptance (%)",
                            scale=alt.Scale(domain=[0, 100])),
                )
            )
            _aql_line = (
                alt.Chart(_aql_rule)
                .mark_rule(color="#B4863B", strokeDash=[5, 4], strokeWidth=1.5)
                .encode(x="x:Q")
            )
            _aql_text = (
                alt.Chart(_aql_rule)
                .mark_text(dy=-120, dx=4, fontSize=11, color="#B4863B", align="left")
                .encode(x="x:Q", text="label:N")
            )
            _lq_line = (
                alt.Chart(_lq_rule)
                .mark_rule(color="#4B6FA8", strokeDash=[5, 4], strokeWidth=1.5)
                .encode(x="x:Q")
            )
            _lq_text = (
                alt.Chart(_lq_rule)
                .mark_text(dy=-100, dx=4, fontSize=11, color="#4B6FA8", align="left")
                .encode(x="x:Q", text="label:N")
            )
            _ac_line = (
                alt.Chart(_ac_rule)
                .mark_rule(color="#8A1F2B", strokeDash=[2, 2], strokeWidth=2)
                .encode(x="x:Q")
            )
            _ac_text = (
                alt.Chart(_ac_rule)
                .mark_text(dy=-140, dx=4, fontSize=11, color="#8A1F2B", align="left")
                .encode(x="x:Q", text="label:N")
            )
            st.altair_chart(
                (_oc_line + _aql_line + _aql_text + _lq_line + _lq_text + _ac_line + _ac_text).properties(height=300),
                use_container_width=True,
            )
            _uitleg_ruleofthumb_label2 = f"{_uitleg_ruleofthumb:.2f}".replace(".", ",")
            _uitleg_lq_count_label = f"{_uitleg_lq_count:.2f}".replace(".", ",")
            _uitleg_paq_pct_label = f"{probability_of_acceptance(_uitleg_n, _uitleg_ac, _uitleg_aql, 'binomial') * 100:.2f}%".replace(".", ",")
            _uitleg_plq_pct_label = f"{probability_of_acceptance(_uitleg_n, _uitleg_ac, _uitleg_aql * DEFAULT_LQ_MULTIPLIER, 'binomial') * 100:.2f}%".replace(".", ",")
            st.caption(
                f"Table and chart only show the range from "
                f"{_uitleg_oc_counts[0]} to {_uitleg_oc_counts[-1]} dirty "
                "houses: outside that range the probability of acceptance is practically "
                "100% (below) or practically 0% (above) and adds nothing. "
                "The brown dashed line marks n×AQL "
                f"({_uitleg_ruleofthumb_label2} dirty houses) -- the probability of "
                f"acceptance there is the PAQ ({_uitleg_paq_pct_label}). The "
                "blue dashed line marks 3×n×AQL "
                f"({_uitleg_lq_count_label} dirty houses, the LQ at the "
                f"standard multiplier {DEFAULT_LQ_MULTIPLIER:g}) -- the probability "
                f"of acceptance there is the PLQ ({_uitleg_plq_pct_label}). The "
                "red line marks the real acceptance number "
                f"Ac={_uitleg_ac}. All three are also marked in the table "
                "above. This window shows exactly the steep middle stretch "
                "of the curve -- the flat plateaus at ~100% and ~0% "
                "have been deliberately cut off, so within this range the "
                "probability of acceptance does drop almost "
                "linearly for each extra dirty house; only outside this window does the "
                "curve bend back toward the flat tails."
            )

        with _uitleg_chart_col2:
            st.markdown(f"**Why Ac={_uitleg_ac}: cumulative probability per c**")
            _uitleg_ac_range = list(range(0, _uitleg_ac + 4))
            _uitleg_ac_df = pd.DataFrame(
                {
                    "c": _uitleg_ac_range,
                    "pa_pct": [
                        probability_of_acceptance(_uitleg_n, c, _uitleg_aql, "binomial") * 100
                        for c in _uitleg_ac_range
                    ],
                }
            )
            _uitleg_ac_df["status"] = [
                f"Meets (>= {_uitleg_conf_pct}%)" if pa >= _uitleg_conf_pct else f"Not enough confidence (< {_uitleg_conf_pct}%)"
                for pa in _uitleg_ac_df["pa_pct"]
            ]
            _uitleg_ac_df["label"] = _uitleg_ac_df["pa_pct"].map(lambda v: f"{v:.2f}%".replace(".", ","))
            st.dataframe(
                _uitleg_ac_df.rename(columns={"c": "c", "pa_pct": "P(X <= c) (%)", "status": "Status"})[
                    ["c", "P(X <= c) (%)", "Status"]
                ],
                hide_index=True, width="stretch",
            )
            _uitleg_ac_bar = (
                alt.Chart(_uitleg_ac_df)
                .mark_bar()
                .encode(
                    x=alt.X("c:O", title="Acceptance number c (max. allowed rejected)"),
                    y=alt.Y("pa_pct:Q", title=f"Cumulative probability P(X <= c) at {_uitleg_aql_label} (%)",
                            scale=alt.Scale(domain=[0, 105])),
                    color=alt.Color(
                        "status:N",
                        title=None,
                        scale=alt.Scale(
                            domain=[f"Not enough confidence (< {_uitleg_conf_pct}%)", f"Meets (>= {_uitleg_conf_pct}%)"],
                            range=["#B4863B", "#1F4B49"],
                        ),
                    ),
                )
            )
            _uitleg_ac_text = (
                alt.Chart(_uitleg_ac_df)
                .mark_text(dy=-8, fontSize=11)
                .encode(x=alt.X("c:O"), y=alt.Y("pa_pct:Q"), text="label:N")
            )
            _uitleg_confidence_rule = (
                alt.Chart(pd.DataFrame({"y": [_uitleg_conf_pct]}))
                .mark_rule(color="#8A1F2B", strokeDash=[5, 4], strokeWidth=1.5)
                .encode(y="y:Q")
            )
            st.altair_chart(
                (_uitleg_ac_bar + _uitleg_ac_text + _uitleg_confidence_rule).properties(height=300),
                use_container_width=True,
            )
            st.caption(
                f"The dashed line sits at the chosen confidence level "
                f"({_uitleg_conf_pct}%). At c={_uitleg_ac - 1}, the "
                f"cumulative probability doesn't yet clear that line "
                f"({_uitleg_ac_df.loc[_uitleg_ac_df['c'] == _uitleg_ac - 1, 'label'].values[0]}). "
                f"At c={_uitleg_ac}, the probability first rises above "
                f"{_uitleg_conf_pct}% "
                f"({_uitleg_ac_df.loc[_uitleg_ac_df['c'] == _uitleg_ac, 'label'].values[0]}) "
                "-- and that's exactly the moment "
                f"`find_acceptance_number()` stops counting. Hence "
                f"Ac={_uitleg_ac}."
            )

        st.markdown(
            "The AQL is therefore not a tolerance limit, but the point at which you "
            "*deliberately* set a high probability of acceptance to protect the cleaning "
            "partner against bad luck in the sample. It's the LQ that "
            "determines from which level you want to be confident enough to say "
            "\"this doesn't meet the standard\" -- and as the table "
            "shows, there's quite a gap between the two in practice."
        )

    st.markdown("---")
    st.markdown("### The business case for management")
    with st.container(border=True):
        st.markdown("**🎯 The pitch in four points**")
        st.markdown(
            "- **Reproducible.** Two different inspectors scoring the same "
            "findings arrive at the same level and the same "
            "bonus/malus -- no more he-said-she-said over an impression.\n"
            "- **Better-spent inspection hours.** Sample size scales "
            "with volume (Table C.1), so small sites are no longer "
            "over-inspected and large ones are no longer under-inspected.\n"
            "- **A credible, timely incentive.** Monthly, "
            "graduated consequences behave the way both delay-discounting "
            "and graduated-sanctions research predict a "
            "credible incentive should behave.\n"
            "- **Less friction, more relationship.** A system that mainly rewards "
            "invites a cleaning partner to reach for the highest level "
            "instead of merely avoiding the lowest -- that's "
            "cheaper in the long run than turnover of "
            "cleaning partners or legal disputes over a vague clause."
        )
        st.markdown(
            "None of the individual pieces here is new -- acceptance sampling "
            "is a hundred years old, demerit scoring is standard quality "
            "theory, and graduated sanctions are a well-studied "
            "institutional design principle. What's missing from most cleaning "
            "contracts isn't the science -- it's that nobody has ever "
            "combined it into one monthly cycle. That's exactly what this "
            "tool does."
        )

    with st.expander(":material/school: Sources and further reading"):
        st.markdown(
            "- Shewhart, W.A. (1920s, Bell Telephone Laboratories) -- "
            "statistical foundation of process control; Dodge, H.F. & Romig, H.G. "
            "built practical acceptance-sampling tables on top of it.\n"
            "- ANSI/ASQ Z1.4, *Sampling Procedures and Tables for Inspection by "
            "Attributes* -- civilian successor to MIL-STD-105E; mirrored "
            "internationally as ISO 2859-1.\n"
            "- BS EN 13549:2001, *Cleaning services -- Basic requirements and "
            "recommendations for quality measuring systems*, Annex C.\n"
            "- Ainslie, G. (1975). *Specious reward: a behavioral theory of "
            "impulsiveness and impulse control.* Psychological Bulletin. "
            "Generalized by Loewenstein, G. & Prelec, D. (1992), "
            "*Anomalies in Intertemporal Choice*, Quarterly Journal of "
            "Economics.\n"
            "- Kahneman, D. & Tversky, A. (1979). *Prospect Theory: An Analysis "
            "of Decision under Risk.* Econometrica -- the basis of loss "
            "aversion.\n"
            "- Ostrom, E. (1990). *Governing the Commons: The Evolution of "
            "Institutions for Collective Action.* Cambridge University Press -- "
            "graduated sanctions as a design principle of durable institutions.\n"
            "- Juran, J.M. -- founder of the \"cost of quality\" framework.\n"
            "- *Journal of Quality Technology*, Vol. 31, No. 2 -- \"Exact "
            "Properties of Demerit Control Charts.\"\n"
            "- NEN 2075 / VSR-KMS 3 (Vereniging Schoonmaak Research / Stichting "
            "Schoonmaakkwaliteit, 2014); Handboek VSR-Keurmerk, SSK, version "
            "17.01 (2017).\n"
            "- ISO 22483:2020, *Tourism and related services -- Hotels -- "
            "Service requirements.*"
        )