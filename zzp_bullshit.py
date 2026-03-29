import streamlit as st
import pandas as pd
import plotly.express as px
import io

try:
    st.set_page_config(
        page_title="ZZP uurtarieven vs bullshitgehalte",
        page_icon=":material/analytics:",
        layout="wide",
    )
except:
    pass

def main():
    # ── Data ──────────────────────────────────────────────────────────────────────

    CSV = """Beroep,Specialisatie,Uurtarief (EUR),Bullshitgehalte (0-10)
    Aannemer,,55,1
    Aannemer,Bouw,55,1
    Aannemer,Duurzaam bouwen,55,1
    Aannemer,Onderhoud en reparatie,53,1
    Aannemer,Renovatie,53,1
    Accountant,,109,3
    Accountant,Administratief accountant,94,3
    Accountant,Belastingzaken,99,3
    Accountant,Controle-accountant (audit),140,3
    Administrateur,,65,3
    Administrateur,Financieel administrateur,62,3
    Administrateur,Projectadministrateur,67,3
    Administrateur,Salarisadministrateur,71,3
    Adviseur (niet belasting communicatie hypotheek),,109,7
    Adviseur (niet belasting communicatie hypotheek),Bedrijfsvoering & organisatieadvies,121,7
    Adviseur (niet belasting communicatie hypotheek),Bestuursadvies,127,7
    Adviseur (niet belasting communicatie hypotheek),Bouw- & vastgoedadvies,100,7
    Adviseur (niet belasting communicatie hypotheek),Cultuur & erfgoed,116,7
    Adviseur (niet belasting communicatie hypotheek),Duurzaamheid & energietransitie,109,7
    Adviseur (niet belasting communicatie hypotheek),Economie / retail / horeca,116,7
    Adviseur (niet belasting communicatie hypotheek),Facility & huisvesting,125,7
    Adviseur (niet belasting communicatie hypotheek),Financieel advies,119,7
    Adviseur (niet belasting communicatie hypotheek),HR / vitaliteit / inzetbaarheid,110,7
    Adviseur (niet belasting communicatie hypotheek),Inkoop & contractmanagement,112,7
    Adviseur (niet belasting communicatie hypotheek),Installaties & techniek,99,7
    Adviseur (niet belasting communicatie hypotheek),IT-advies,122,7
    Adviseur (niet belasting communicatie hypotheek),Juridisch advies,113,7
    Adviseur (niet belasting communicatie hypotheek),Learning & Development (L&O),121,7
    Adviseur (niet belasting communicatie hypotheek),Milieu & ecologie,119,7
    Adviseur (niet belasting communicatie hypotheek),Mobiliteit & verkeerskunde,111,7
    Adviseur (niet belasting communicatie hypotheek),Omgevingswet & vergunningen,117,7
    Adviseur (niet belasting communicatie hypotheek),Onderwijsadvies & onderwijshuisvesting,110,7
    Adviseur (niet belasting communicatie hypotheek),Procesmanagement & projectbeheersing,115,7
    Adviseur (niet belasting communicatie hypotheek),Sociaal domein & zorgadvies,100,7
    Adviseur (niet belasting communicatie hypotheek),Veiligheid / KAM / QHSE,106,7
    Advocaat,,241,4
    Advocaat,Familierecht,235,4
    Agrariër,,37,0
    Agrariër,Akkerbouwer,36,0
    AI-consultant / specialist,,120,8
    AI-consultant / specialist,AI implementatie bij klanten,114,8
    Architect (IT),,115,6
    Architect (IT),Applicatiearchitectuur,107,6
    Architect (IT),Cloudarchitectuur,116,6
    Architect (IT),Enterprise-architectuur,121,6
    Architect (IT),IT-strategie,112,6
    Architect (IT),Infrastructuurarchitectuur,122,6
    Architect (IT),Solutionarchitectuur,112,6
    Architect (niet IT),,90,3
    Architect (niet IT),Bouwkundig,87,3
    Architect (niet IT),Duurzaamheid,102,3
    Architect (niet IT),Interieur,97,3
    Artiest,,67,1
    Artiest,Beeldend kunstenaar,54,1
    Arts,,108,0
    Arts,Bedrijfsarts,185,0
    Arts,Huisarts,85,0
    Arts,Dierenarts,83,0
    Arts,Overige specialisten,145,0
    Assistent,,62,2
    Assistent,Administratieve ondersteuning & coördinatie,60,2
    Assistent,Office manager / managementassistent,61,2
    Assistent,Personal assistant / executive assistant,66,2
    Assistent,Projectondersteuning / PMO,65,2
    Assistent,Virtueel assistent (VA),60,2
    Audiovisueel technicus,,47,2
    Audiovisueel technicus,Live-evenementen,40,2
    Begeleider (zorg),,53,0
    Begeleider (zorg),Geestelijke gezondheidszorg (GGZ),57,0
    Begeleider (zorg),Jeugdzorg,53,0
    Begeleider (zorg),Mensen met een beperking,51,0
    Begeleider (zorg),Ouderenzorg,47,0
    Belastingadviseur,,145,5
    Belastingadviseur,Advies bij fiscale structuren,174,5
    Belastingadviseur,BTW-aangiftes,119,5
    Belastingadviseur,Fiscale optimalisatie,141,5
    Belastingadviseur,Inkomstenbelasting,125,5
    Belastingadviseur,Internationale belastingzaken,191,5
    Belastingadviseur,Vennootschapsbelasting,151,5
    Beveiliger,,37,1
    Beveiliger,Evenementenbeveiliging,35,1
    Beveiliger,Objectbeveiliging,35,1
    Beveiliger,Toezicht & handhaving,35,1
    Boekhouder,,65,3
    Boekhouder,Belastingaangiftes,67,3
    Boekhouder,Digitalisering,65,3
    Boekhouder,Financiële administratie,65,3
    Boekhouder,Herstructurering & crisisbeheer,68,3
    Boekhouder,Jaarrekeningen,69,3
    Business analist,,102,7
    Business analist,Bedrijfsprocessen modelleren,102,7
    Business analist,Data-analyse,101,7
    Business analist,Procesoptimalisatie,101,7
    Business analist,Requirementsanalyse,103,7
    Chauffeur,,41,1
    Chauffeur,Goederenvervoer,41,1
    Chauffeur,Personenvervoer,41,1
    Chef-kok,,45,1
    Chef-kok,Catering,44,1
    Chef-kok,Luxe dining,46,1
    Chef-kok,Restaurantkeuken,42,1
    Coach / trainer (niet personal trainer),,95,7
    Coach / trainer (niet personal trainer),Business/executive/teams,135,7
    Coach / trainer (niet personal trainer),Persoonlijk (life/loopbaan/gezondheid),98,7
    Coach / trainer (niet personal trainer),Soft skills (presentatie/communicatie),122,7
    Coach / trainer (niet personal trainer),Sport/beweging,57,7
    Communicatieadviseur,,92,8
    Communicatieadviseur,Bedrijfscommunicatie,94,8
    Communicatieadviseur,Crisiscommunicatie,114,8
    Communicatieadviseur,Employer branding,90,8
    Communicatieadviseur,Externe communicatie,92,8
    Communicatieadviseur,Interne communicatie,92,8
    Communicatieadviseur,Marketingcommunicatie,87,8
    Communicatieadviseur,Mediatraining & woordvoering,111,8
    Communicatieadviseur,PR & public affairs,93,8
    Communicatieadviseur,Social-media-strategie,80,8
    Communicatieadviseur,Stakeholdermanagement,100,8
    Communicatieadviseur,Strategische communicatieplanning,95,8
    Communicatieadviseur,Tekstschrijven & copywriting,85,8
    Communicatieadviseur,Verandercommunicatie,100,8
    Communicatieadviseur,Visuele communicatie (infographics video),85,8
    Consultant (IT),,108,6
    Consultant (IT),Agile/Scrum-consultancy,109,6
    Consultant (IT),Business intelligence (BI),110,6
    Consultant (IT),Cloudmigratie & -beheer,100,6
    Consultant (IT),Cybersecurity,111,6
    Consultant (IT),Data-analyse en governance,110,6
    Consultant (IT),DevOps-consultancy,110,6
    Consultant (IT),ERP/CRM-implementaties,111,6
    Consultant (IT),IT-infrastructuur & netwerken,100,6
    Consultant (IT),IT-strategie & digitalisering,110,6
    Consultant (IT),Machine learning en AI,118,6
    Consultant (IT),Projectmanagement voor IT,110,6
    Consultant (IT),Softwareontwikkeling & -optimalisatie,106,6
    Consultant (IT),Technische architectuur,105,6
    Consultant (niet IT),,121,8
    Consultant (niet IT),Arbo- & veiligheidsconsultancy,119,8
    Consultant (niet IT),Bedrijfsprocesoptimalisatie,119,8
    Consultant (niet IT),Business development & commercie,124,8
    Consultant (niet IT),Crisismanagement,138,8
    Consultant (niet IT),Duurzaamheid & energietransitie,121,8
    Consultant (niet IT),Duurzaamheids- & milieumanagement,112,8
    Consultant (niet IT),Finance / corporate finance,128,8
    Consultant (niet IT),HR- & organisatieadvies,128,8
    Consultant (niet IT),Installaties & techniek,104,8
    Consultant (niet IT),Logistiek & supply chain,123,8
    Consultant (niet IT),Marketing- en communicatieadvies,124,8
    Consultant (niet IT),Onderwijs / L&D,114,8
    Consultant (niet IT),Operations / supply-chain-management,119,8
    Consultant (niet IT),Projectmanagementadvies,122,8
    Consultant (niet IT),Public affairs / overheid,127,8
    Consultant (niet IT),QA/RA & GxP (pharma/medisch),131,8
    Consultant (niet IT),Risicomanagement & compliance,129,8
    Consultant (niet IT),Sociaal domein & zorg,106,8
    Consultant (niet IT),Strategisch advies,134,8
    Consultant (niet IT),Training & coaching,130,8
    Consultant (niet IT),Verandermanagement,130,8
    Consultant (niet IT),Zorgmanagement,132,8
    Controller,,98,4
    Controller,Financiële controle,96,4
    Controller,Kostenbeheer,97,4
    Controller,Rapportages & budgettering,99,4
    Controller,Risicoanalyse,105,4
    Controller,Strategische advisering,105,4
    Copywriter,,78,3
    Copywriter,AI-ondersteund schrijven,75,3
    Copywriter,Blogs,78,3
    Copywriter,Content voor marketingcampagnes,79,3
    Copywriter,Creatief schrijven,80,3
    Copywriter,SEO-teksten,75,3
    Copywriter,Webteksten,77,3
    Data-analist,,94,4
    Data-analist,Big data-analyses,95,4
    Data-analist,Data cleaning,92,4
    Data-analist,Visualisaties,93,4
    Designer / Grafisch vormgever,,72,2
    Designer / Grafisch vormgever,Animatie & motion graphics,81,2
    Designer / Grafisch vormgever,Digitale productdesign (apps interfaces),87,2
    Designer / Grafisch vormgever,Grafisch ontwerp,72,2
    Designer / Grafisch vormgever,Illustraties & concept art,75,2
    Designer / Grafisch vormgever,Packaging design,73,2
    Designer / Grafisch vormgever,Productontwerp,71,2
    Designer / Grafisch vormgever,Typografie en letterontwerp,72,2
    Designer / Grafisch vormgever,User Experience (UX) design,82,2
    Designer / Grafisch vormgever,User Interface (UI) design,83,2
    Designer / Grafisch vormgever,Webdesign,74,2
    Docent,,64,1
    Docent,Bijlesdocent,47,1
    Docent,Hoger onderwijs,77,1
    Docent,NT2-docent,56,1
    Docent,Primair onderwijs,63,1
    Docent,Specifieke vakgebieden,68,1
    Docent,Voortgezet onderwijs,65,1
    Elektricien,,56,0
    Elektricien,Domotica,59,0
    Elektricien,Huishoudelijke elektriciteit,54,0
    Elektricien,Industriële systemen,57,0
    Elektricien,Zonnepanelen,56,0
    Event manager,,66,5
    Event manager,Bedrijfsevents,75,5
    Event manager,Bedrijfsfeesten,72,5
    Event manager,Congressen & beurzen,81,5
    Event manager,Culturele events (concerten/festivals),62,5
    Event manager,Marketing & klant-events,71,5
    Fotograaf,,94,2
    Fotograaf,Bruiloften,96,2
    Fotograaf,Commercieel werk,98,2
    Fotograaf,Evenementen,94,2
    Fotograaf,Portretten,98,2
    Fysiotherapeut,,77,1
    Fysiotherapeut,Algemene fysiotherapie,67,1
    Hondentrimmer,,50,1
    Hondentrimmer,Rasspecifiek,51,1
    Hovenier,,48,1
    Hovenier,Aanleg & onderhoud,46,1
    Hovenier,Boomverzorging,48,1
    Hovenier,Landschapsbeheer,47,1
    Hovenier,Tuinontwerp,53,1
    HR-professional (niet recruiter),,100,6
    HR-professional (niet recruiter),Arbeidsrecht,101,6
    HR-professional (niet recruiter),Arbeidsvoorwaarden & functiehuis,100,6
    HR-professional (niet recruiter),Coaching & vertrouwenspersoon,100,6
    HR-professional (niet recruiter),Compensation & Benefits,107,6
    HR-professional (niet recruiter),Data & analytics (HR),107,6
    HR-professional (niet recruiter),HR-advies / generalist / HRBP,98,6
    HR-professional (niet recruiter),HR-processen & systemen,97,6
    HR-professional (niet recruiter),Interim HR-management,104,6
    HR-professional (niet recruiter),Learning & Development,101,6
    HR-professional (niet recruiter),Organisatieontwikkeling,107,6
    HR-professional (niet recruiter),Payroll & salarisverwerking,88,6
    HR-professional (niet recruiter),Reorganisatie,106,6
    HR-professional (niet recruiter),Talentontwikkeling,107,6
    HR-professional (niet recruiter),Verzuim & re-integratie,95,6
    Ingenieur,,90,2
    Ingenieur,Civiele techniek,97,2
    Ingenieur,Elektrotechniek,92,2
    Ingenieur,Mechanische techniek,81,2
    Inkoper,,105,4
    Inkoper,Overheidsinkoop,112,4
    Inkoper,Technische inkoop,101,4
    Installateur,,60,1
    Installateur,Airco & koeltechniek,62,1
    Installateur,Allround installatietechniek,60,1
    Installateur,Badkamerinstallaties,58,1
    Installateur,CV-installaties,60,1
    Installateur,Elektrotechniek,58,1
    Installateur,IT / audio / netwerk,61,1
    Installateur,Ventilatiesystemen,59,1
    Installateur,Verwarming / CV / warmtesystemen,60,1
    Installateur,Warmtepompen,60,1
    Installateur,Zonnepanelen & laadpalen,58,1
    Interieurdesigner,,78,4
    Interieurdesigner,Commerciële interieurs,82,4
    Interieurdesigner,Concept & styling,79,4
    Interieurdesigner,Meubel- & maatwerkontwerp,79,4
    Interieurdesigner,Particuliere interieurs,81,4
    Interieurdesigner,Ruimtelijke indeling & lichtplan,84,4
    Interim Manager (IT),,121,7
    Interim Manager (IT),Agile / Scrum-implementaties,122,7
    Interim Manager (IT),Digitalisering & transformatie,126,7
    Interim Manager (IT),ERP/CRM-implementaties,124,7
    Interim Manager (IT),IT-infrastructuur & netwerken,116,7
    Interim Manager (IT),Projectmanagement in IT,121,7
    Interim Manager (IT),Softwareontwikkelingsmanagement,123,7
    Interim Manager (IT),Teammanagement binnen DevOps,114,7
    Interim Manager (niet IT),,117,7
    Interim Manager (niet IT),Crisismanagement,128,7
    Interim Manager (niet IT),Facilitair management,106,7
    Interim Manager (niet IT),Financieel management,128,7
    Interim Manager (niet IT),HR-management,116,7
    Interim Manager (niet IT),Logistiek & supply-chain-management,122,7
    Interim Manager (niet IT),Marketing- & salesmanagement,117,7
    Interim Manager (niet IT),Onderwijs,108,7
    Interim Manager (niet IT),Operationeel management,114,7
    Interim Manager (niet IT),Overheidsmanagement,122,7
    Interim Manager (niet IT),Projectmanagement,116,7
    Interim Manager (niet IT),Strategisch management,128,7
    Interim Manager (niet IT),Verandermanagement,120,7
    Interim Manager (niet IT),Zorgmanagement,113,7
    Journalist,,66,3
    Journalist,Online media,67,3
    Journalist,Printjournalistiek,64,3
    Jurist,,135,4
    Jurist,Arbeidsrecht,144,4
    Jurist,Bestuursrecht & overheid,129,4
    Jurist,Contracten & privaatrecht,137,4
    Jurist,Omgevingsrecht / RO / milieu / energie,121,4
    Jurist,Ondernemingsrecht / M&A / corporate,137,4
    Jurist,Privacy / AVG / AI-wetgeving,133,4
    Jurist,Vastgoedrecht,129,4
    Kapper,,56,1
    Kapper,Dameskapper,56,1
    Kapper,Herenkapper,57,1
    Kapper,Kinderkapper,59,1
    Kapper,Kleur- & stylingexpert,57,1
    Klusser,,47,1
    Klusser,Allround klusjes,46,1
    Klusser,Installatiewerk,49,1
    Klusser,Onderhoud,48,1
    Klusser,Renovaties,48,1
    Kok,,38,1
    Kok,Catering,38,1
    Kok,Restaurantkeuken,38,1
    Kraanmachinist,,48,0
    Kraanmachinist,Bouw,50,0
    Kraanmachinist,Specifieke machines,48,0
    Lasser,,54,0
    Lasser,Metaalconstructies,51,0
    Lasser,MIG/TIG-lassen,53,0
    Lasser,Precisiewerk,54,0
    Loodgieter,,57,0
    Loodgieter,Duurzame oplossingen,55,0
    Loodgieter,Riolering,57,0
    Loodgieter,Verwarmingssystemen,58,0
    Loodgieter,Waterleidingen,55,0
    Marketeer,,82,8
    Marketeer,AI & marketing-automation,82,8
    Marketeer,AI-content & creatie,82,8
    Marketeer,Brand management,85,8
    Marketeer,Content marketing,74,8
    Marketeer,CRM-marketing,79,8
    Marketeer,Digital advertising (Google Ads social ads),80,8
    Marketeer,Email-marketing,75,8
    Marketeer,Event-marketing,76,8
    Marketeer,Growth-marketing,84,8
    Marketeer,Performance-marketing,84,8
    Marketeer,Product-marketing,88,8
    Marketeer,Search Engine Advertising (SEA),79,8
    Marketeer,Search Engine Optimization (SEO),77,8
    Marketeer,Social-media-marketing,72,8
    Marketeer,Strategische marketing,88,8
    Masseur,,61,2
    Masseur,Ontspanningsmassage,60,2
    Masseur,Sportmassage,59,2
    Masseur,Therapiegericht,63,2
    Meubelmaker,,52,1
    Meubelmaker,Maatwerk,51,1
    Monteur,,56,1
    Monteur,Automonteur,54,1
    Monteur,Elektromonteur,57,1
    Monteur,Installatiemonteur,53,1
    Monteur,Keukenmonteur,56,1
    Monteur,Onderhoudsmonteur,59,1
    Muzikant,,61,1
    Muzikant,Live-optredens,62,1
    Muzikant,Muziekles,60,1
    Ondernemer / eigenaar,,77,5
    Ondernemer / eigenaar,DGA,88,5
    Pedagogisch medewerker,,34,1
    Pedagogisch medewerker,Gastouder,20,1
    Pedagogisch medewerker,Kinderopvang,24,1
    Pedicure,,46,1
    Pedicure,Medisch pedicure,47,1
    Pedicure,Voetverzorging,41,1
    Personal trainer,,51,2
    Personal trainer,Afvallen / leefstijl,56,2
    Personal trainer,Kracht / conditie,53,2
    Personal trainer,PT aan huis / online,56,2
    Personal trainer,Revalidatie,61,2
    Personal trainer,Sport-specifiek,49,2
    Project Manager (IT),,112,6
    Project Manager (IT),Agile / Scrum-projecten,112,6
    Project Manager (IT),Cloudmigratie,113,6
    Project Manager (IT),Cybersecurity-projecten,116,6
    Project Manager (IT),Data & Analytics-projecten,113,6
    Project Manager (IT),DevOps-projecten,117,6
    Project Manager (IT),Digitalisering & transformatie,114,6
    Project Manager (IT),ERP/CRM-implementaties,117,6
    Project Manager (IT),Infrastructuurprojecten,109,6
    Project Manager (IT),Softwareontwikkeling en implementatie,113,6
    Project Manager (niet IT),,103,6
    Project Manager (niet IT),Bouw & infra,108,6
    Project Manager (niet IT),Commercieel / sales / tenders,100,6
    Project Manager (niet IT),Energie & klimaat,108,6
    Project Manager (niet IT),Evenementenplanning,69,6
    Project Manager (niet IT),Facility & huisvesting,103,6
    Project Manager (niet IT),HR-gerelateerde projecten,109,6
    Project Manager (niet IT),Industrie / proces / productie,107,6
    Project Manager (niet IT),Logistiek & supply-chain-projecten,113,6
    Project Manager (niet IT),Marketing & communicatiecampagnes,81,6
    Project Manager (niet IT),Non-profit & fondsenwerving,83,6
    Project Manager (niet IT),Onderwijs / cultuur / sociaal domein,91,6
    Project Manager (niet IT),Overheid / RO / gebiedsontwikkeling,113,6
    Project Manager (niet IT),Procesoptimalisatie & ketens,106,6
    Project Manager (niet IT),Productontwikkeling,109,6
    Project Manager (niet IT),Projectondersteuning,91,6
    Project Manager (niet IT),Verandermanagement & reorganisatie,115,6
    Project Manager (niet IT),Zorgprojecten,108,6
    Psycholoog,,122,2
    Psycholoog,Gezondheidspsycholoog,127,2
    Psycholoog,Kinderpsycholoog,110,2
    Psycholoog,Klinisch psycholoog,146,2
    Psycholoog,Orthopedagoog,106,2
    Psycholoog,Traumapsycholoog,116,2
    Recruiter,,88,7
    Recruiter,Corporate recruitment,88,7
    Recruiter,Freelance recruitment,85,7
    Rij-instructeur,,57,1
    Rij-instructeur,Auto-instructie,58,1
    Schilder,,47,1
    Schilder,Decoratief schilderwerk,49,1
    Schilder,Huisschilder,47,1
    Schilder,Restauratieschilder,49,1
    Schipper / kapitein,,48,1
    Schipper / kapitein,Passagiersvaart,47,1
    Schoonmaker,,29,0
    Schoonmaker,Huishoudelijk,26,0
    Schoonmaker,Kantoorschoonmaak,30,0
    Schoonmaker,Specialistisch,32,0
    Software developer,,94,3
    Software developer,Back-end development,95,3
    Software developer,Front-end development,96,3
    Software developer,Full-stack development,95,3
    Software developer,Mobile development,95,3
    Systeembeheerder,,78,3
    Systeembeheerder,Cloudbeheer,80,3
    Systeembeheerder,Netwerkbeheer,78,3
    Systeembeheerder,Windows-systemen,76,3
    Therapeut (niet fysio),,87,6
    Therapeut (niet fysio),Energetisch & spiritueel werk,78,6
    Therapeut (niet fysio),Kind & gezin,84,6
    Therapeut (niet fysio),Kindertherapeut,86,6
    Therapeut (niet fysio),Lichaamsgerichte therapie,86,6
    Therapeut (niet fysio),Orthomoleculair & natuurgeneeskunde,87,6
    Therapeut (niet fysio),Psychosociale therapie,92,6
    Therapeut (niet fysio),Systeem- & relatietherapie,91,6
    Therapeut (niet fysio),Trauma & rouw,93,6
    Timmerman,,49,0
    Timmerman,Maatwerk,49,0
    Timmerman,Meubeltimmerwerk,48,0
    Timmerman,Restauratiewerk,49,0
    Tolk / vertaler,,73,2
    Tolk / vertaler,Juridisch vertaler,83,2
    Veiligheidskundige,,98,5
    Veiligheidskundige,Arboveiligheid,99,5
    Veiligheidskundige,Milieuveiligheid,105,5
    Verpleegkundige,,62,0
    Verpleegkundige,Algemene zorg,54,0
    Verpleegkundige,Gespecialiseerd,75,0
    Verpleegkundige,Ouderenzorg,55,0
    Verzorgende,,46,0
    Verzorgende,Algemene zorg,45,0
    Verzorgende,Gehandicaptenzorg,48,0
    Verzorgende,IG-verzorgende,47,0
    Verzorgende,Terminale zorg,49,0
    Videomaker,,66,3
    Videomaker,Bedrijfsvideo's,68,3
    Videomaker,Documentaires,71,3
    Videomaker,Evenementen,68,3
    Videomaker,Reclamevideo's,64,3
    Web developer,,73,3
    Web developer,Back-end toepassingen,72,3
    Web developer,E-commerce platforms,73,3
    Web developer,Front-end toepassingen,73,3"""

    @st.cache_data
    def load_data():
        df = pd.read_csv(io.StringIO(CSV))
        df.columns = ["Beroep", "Specialisatie", "Tarief", "Bullshit"]
        df["Tarief"] = pd.to_numeric(df["Tarief"], errors="coerce")
        df["Bullshit"] = pd.to_numeric(df["Bullshit"], errors="coerce")
        df["Specialisatie"] = df["Specialisatie"].fillna("")
        return df

    df = load_data()

    # Only top-level rows (no specialisatie) for the scatter
    df_top = df[df["Specialisatie"] == ""].copy()

    # ── Sidebar filters ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header(":material/tune: Filters")

        tarief_min, tarief_max = int(df["Tarief"].min()), int(df["Tarief"].max())
        tarief_range = st.slider(
            "Uurtarief (€/uur)",
            min_value=tarief_min,
            max_value=tarief_max,
            value=(tarief_min, tarief_max),
            step=5,
        )

        bs_min, bs_max = int(df["Bullshit"].min()), int(df["Bullshit"].max())
        bs_range = st.slider(
            "Bullshitgehalte",
            min_value=bs_min,
            max_value=bs_max,
            value=(bs_min, bs_max),
            step=1,
        )

        toon_specialisaties = st.toggle("Toon specialisaties in tabel", value=False)

        st.caption("Bron: Knab ZZP Uurtarievenboekje 2026")

    # ── Apply filters ──────────────────────────────────────────────────────────────
    mask_top = (
        df_top["Tarief"].between(*tarief_range) &
        df_top["Bullshit"].between(*bs_range)
    )
    df_scatter = df_top[mask_top]

    if toon_specialisaties:
        mask_all = (
            df["Tarief"].between(*tarief_range) &
            df["Bullshit"].between(*bs_range)
        )
        df_table = df[mask_all]
    else:
        df_table = df_scatter

    # ── Color scale ────────────────────────────────────────────────────────────────
    COLOR_SCALE = [
        [0.0,  "#1D9E75"],
        [0.25, "#9FE1CB"],
        [0.5,  "#FAC775"],
        [0.75, "#D85A30"],
        [1.0,  "#A32D2D"],
    ]

    # ── Tabs ───────────────────────────────────────────────────────────────────────
    tab_scatter, tab_table = st.tabs([
        ":material/scatter_plot: Scatterplot",
        ":material/table_chart: Tabel",
    ])

    with tab_scatter:
        st.subheader("Uurtarief vs bullshitgehalte")
        st.caption(f"{len(df_scatter)} beroepen zichtbaar na filter")

        if df_scatter.empty:
            st.warning("Geen beroepen gevonden met deze filters.", icon=":material/search_off:")
        else:
            # Metric summary row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Gemiddeld tarief", f"€ {df_scatter['Tarief'].mean():.0f}")
            c2.metric("Hoogste tarief", f"€ {df_scatter['Tarief'].max():.0f}")
            c3.metric("Gem. bullshit", f"{df_scatter['Bullshit'].mean():.1f}")
            c4.metric("Beroepen", len(df_scatter))

            fig = px.scatter(
                df_scatter,
                x="Bullshit",
                y="Tarief",
                text="Beroep",
                color="Bullshit",
                color_continuous_scale=COLOR_SCALE,
                range_color=[0, 8],
                size_max=14,
                labels={
                    "Bullshit": "Bullshitgehalte (0 = nul BS, 8 = maximum)",
                    "Tarief": "Uurtarief (€/uur)",
                    "Beroep": "Beroep",
                },
                hover_data={"Beroep": True, "Tarief": True, "Bullshit": True},
                template="plotly_white",
            )

            fig.update_traces(
                marker=dict(size=12, line=dict(width=1, color="white")),
                textposition="top center",
                textfont=dict(size=10),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Tarief: €%{y}/uur<br>"
                    "Bullshit: %{x}<extra></extra>"
                ),
            )

            fig.update_layout(
                height=600,
                coloraxis_showscale=False,
                xaxis=dict(
                    tickmode="linear",
                    dtick=1,
                    range=[-0.5, 8.5],
                    gridcolor="rgba(0,0,0,0.06)",
                    zeroline=False,
                ),
                yaxis=dict(
                    gridcolor="rgba(0,0,0,0.06)",
                    zeroline=False,
                    tickprefix="€ ",
                ),
                font=dict(size=12),
                margin=dict(t=30, b=40, l=60, r=20),
                plot_bgcolor="white",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            # Add jitter annotation
            st.plotly_chart(fig, width='stretch')

            # Correlation callout
            if len(df_scatter) >= 5:
                corr = df_scatter[["Tarief", "Bullshit"]].corr().iloc[0, 1]
                direction = "positief" if corr > 0 else "negatief"
                st.caption(
                    f":material/query_stats: Correlatie tarief × bullshit: **{corr:.2f}** ({direction}) — "
                    f"{'hoe meer bullshit, hoe hoger het tarief' if corr > 0 else 'geen duidelijk verband'}."
                )

    with tab_table:
        st.subheader("Alle beroepen & specialisaties")
        st.caption(f"{len(df_table)} rijen na filter")

        if df_table.empty:
            st.warning("Geen resultaten gevonden.", icon=":material/search_off:")
        else:
            display_df = df_table[["Beroep", "Specialisatie", "Tarief", "Bullshit"]].copy()
            display_df = display_df.sort_values(["Beroep", "Specialisatie"]).reset_index(drop=True)

            st.dataframe(
                display_df,
                width='stretch',
                hide_index=True,
                column_config={
                    "Beroep": st.column_config.TextColumn("Beroep", width="medium", pinned=True),
                    "Specialisatie": st.column_config.TextColumn("Specialisatie", width="large"),
                    "Tarief": st.column_config.NumberColumn(
                        "Uurtarief (€/uur)",
                        format="€ %d",
                        width="small",
                        help="Gemiddeld uurtarief excl. btw (Knab 2026)",
                    ),
                    "Bullshit": st.column_config.ProgressColumn(
                        "Bullshitgehalte",
                        min_value=0,
                        max_value=8,
                        format="%d / 8",
                        width="medium",
                    ),
                },
                height=600,
            )

            # Download button
            csv_out = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=":material/download: Download gefilterde CSV",
                data=csv_out,
                file_name="zzp_uurtarieven_gefilterd.csv",
                mime="text/csv",
            )
    st.sidebar.info("Bron: KNAB ZZP tarieven 2025")

if __name__ == "__main__":
    main()