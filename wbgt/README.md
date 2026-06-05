# WBGT Toolbox — Hittestress-calculator (Liljegren)

**Versie:** zie individuele bestanden (`current_version = "yyyymmdd-hhmmss"`)  
**Auteur:** René Smit — [rcsmit.streamlit.app](https://rcsmit.streamlit.app) | [rene-smit.com](https://rene-smit.com)  
**Licentie:** zie afzonderlijke bestanden

---

## Overzicht

Deze toolbox berekent de **Wet Bulb Globe Temperature (WBGT)** via de Liljegren-methode — de meest nauwkeurige berekening voor buitenomstandigheden. De code combineert drie onafhankelijke implementaties van hetzelfde algoritme (pure Python, C-port, Cython-vertaling), een volledige zonnestralingscalculator, KNMI-datadownload en een Streamlit-interface.

Gebaseerd op:
- Kong & Huber (2022) *Earth's Future* — de Cython-implementatie
- Liljegren et al. — de oorspronkelijke WBGT-formules
- Bird & Hulstrom (1981) / Spencer (1971) — clear-sky zonnestraling


---

## De drie WBGT-engines vergeleken

| | `wbgt_liljegren.py` | `wbgt_liljegren_c_code.py` | `wbgt_liljegren_cython_wrapper.py` |
|--|--|--|--|
| Implementatiebasis | Eigen Python-vertaling | Directe C-code port | Kong & Huber (2022) Cython |
| Afhankelijkheden | stdlib + numpy | stdlib | numpy + coszenith-kernel |
| Batchverwerking | nee (scalar) | nee (scalar) | via 3D array |
| Gebruik in productie | primair | alternatief/validatie | voor klimaatdata (GCM) |

`wbgt_utils.py` roept standaard `wbgt_liljegren_from_station_cython()` aan (Cython-variant) en valt terug op `wbgt_liljegren_from_station_opus()` (C-code port).

De drie scripts hebben een hoge correlatie, ook met de KNMI-waardes. Echter in de ochtend (5-7am en avond (7-9pm) zijn er verschillen, door de lage zonnehoek en de manier waarop de scripts er mee omgaan.
---

## Bestandsstructuur

```
wbgt_knmi.py                        ← Streamlit entry point (tab-layout)
│
├── wbgt_select_time_place.py        ← UI: locatie- en tijdselectie
├── wbgt_utils.py                    ← UI + business logic: tabellen, grafieken, KPI's
├── wbgt_solar_app.py                ← UI: solar radiation viewer
├── wbgt_replicate_knmi.py           ← UI: historische data 1991–2025
├── wbgt_vergelijk_script_met_knmi.py ← UI: vergelijk eigen berekening met KNMI
│
├── wbgt_solar_radiation.py          ← Zonneposition + clear-sky (standalone)
│
├── wbgt_liljegren.py                ← WBGT engine · pure Python (primair)
├── wbgt_liljegren_c_code.py         ← WBGT engine · C-code port (opus)
├── wbgt_liljegren_cython_wrapper.py ← WBGT engine · wrapper voor Cython-versie
│
├── wbgt_liljegren_from_cython.py    ← Numerieke kernel: WBGT_Liljegren / fdir / Tg / Tnwb
├── wbgt_coszenith_from_cython.py    ← Numerieke kernel: cosz / cosza / coszda
│
├── wbgt_download_fulldataset_knmi.py ← KNMI-dataset download (async, standalone script)
└── utils.py                         ← Generieke hulpfuncties (heat index, LOESS, KNMI-stations)
```

> **Deployment-pad:** Streamlit Cloud verwacht de bestanden in `show_knmi_functions/`. Elke module probeert eerst een directe import; bij falen valt hij terug op `from show_knmi_functions.xxx import ...`.

---

## Lagenarchitectuur

```
┌──────────────────────────────────────────┐
│  Entry point:  wbgt_knmi.py             │  → streamlit run wbgt_knmi.py
└──────────────┬───────────────────────────┘
               │ importeert
┌──────────────▼───────────────────────────┐
│  Streamlit UI-laag                       │
│  select_time_place · wbgt_utils          │
│  wbgt_solar_app · wbgt_replicate_knmi    │
│  wbgt_vergelijk_script_met_knmi          │
│  wbgt_download_fulldataset_knmi          │
└──────────────┬───────────────────────────┘
               │ importeert
┌──────────────▼───────────────────────────┐
│  Business logic / shared                 │
│  wbgt_utils · wbgt_solar_radiation       │
│  utils                                   │
└──────────────┬───────────────────────────┘
               │ importeert
┌──────────────▼───────────────────────────┐
│  WBGT engines (drie parallelle versies)  │
│  wbgt_liljegren (pure Python)            │
│  wbgt_liljegren_c_code (C-port)          │
│  wbgt_liljegren_cython_wrapper           │
└──────────────┬───────────────────────────┘
               │ importeert
┌──────────────▼───────────────────────────┐
│  Numerieke kernels (Cython → Python)     │
│  wbgt_liljegren_from_cython              │
│  wbgt_coszenith_from_cython              │
└──────────────────────────────────────────┘
```

---

## Bestandsdocumentatie

### `wbgt_knmi.py` — Entry point

Bouwt de Streamlit-applicatie op als zeven tabs. Haalt locatie/tijd op via `select_time_place()` in de sidebar en geeft deze door aan alle tab-functies.

| Tab | Functie |
|-----|---------|
| Main | `main_()` — actuele WBGT + grafieken |
| Tabel | `referentie_tabel()` — WBGT per T/RH-combinatie |
| Calculator | `feels_like_calculator()` — gevoelstemperatuur |
| Solarinfo | `solar_wrapper()` — zonnestraling |
| 1991–2025 | `show_historical_data()` — historische KNMI-data |
| Script vs KNMI | `vergelijk_script_met_knmi_download()` — validatie |
| INFO | `show_info()` / `info()` — uitleg |

---

### `wbgt_select_time_place.py` — Locatie- en tijdwidget

```python
lat, lon, utc_dt, loc_name, selected_date, selected_time, tz, LOCATIONS = select_time_place()
```

Toont twee kolommen: locatiedropdown (24 steden + custom) en datum/tijd-invoer. Converteert lokale tijd naar naive UTC `datetime`. Geeft ook de volledige `LOCATIONS`-lijst terug zodat andere tabs er locatievergelijkingen mee kunnen doen.

**Tijdzone-afhandeling:** gebruikt `zoneinfo.ZoneInfo` (stdlib, Python 3.9+); stopt de app bij een onbekende tijdzonecode.

---

### `wbgt_utils.py` — Business logic + Streamlit-functies

Het zwaarste bestand (~1300 regels). Bevat zowel pure berekeningen als Streamlit-rendercode.

**Berekeningen:**

| Functie | Beschrijving |
|---------|-------------|
| `wbgt_buiten(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt)` | WBGT buiten via Liljegren |
| `wbgt_schaduw(temp_c, rh_pct)` | Vereenvoudigde WBGT in schaduw |
| `wbgt_bernard(temp_c, rh_pct)` | Bernard-formule |
| `wbgt_risico(wbgt)` → `(label, kleur)` | Risicozone op basis van KNMI-drempelwaarden |
| `wbgt_bereken_df(df, stn)` | Bereken WBGT voor een DataFrame met KNMI-uurdata |
| `feels_like_all(...)` → dict | Alle gevoelstemperatuurmethoden tegelijk |
| `wind_2m(u_hm, h_m, stability, setting)` | Windsnelheid omrekenen naar 2 m hoogte |

**Visualisatie (Streamlit/Plotly):**

| Functie | Beschrijving |
|---------|-------------|
| `maak_wbgt_figuur(df, toon_temp)` | Tijdreeksgrafiek WBGT + temp |
| `maak_wbgt_maand_barchart(df)` | Maandelijkse barchart |
| `maak_wbgt_barchart(df, datum)` | Dagoverzicht barchart |
| `render_wbgt_chart(df, only_dagmax)` | Volledige Streamlit-renderpijplijn |
| `referentie_tabel(lat, lon, dt_ref)` | Heatmap T×RH → WBGT |
| `feels_like_calculator(lat, lon, utc_dt)` | Interactieve gevoelscalculator |
| `main_()` | Hoofd-Streamlit-weergave (actuele WBGT) |

**Constanten:** `KNMI_DREMPELWAARDEN`, `BADGE_KLEUREN_KNMI`, `ZONE_KLEUREN_WBGT`, `RISICO_ZONES_WBGT` — gedeeld met `wbgt_replicate_knmi`.

---

### `wbgt_solar_radiation.py` — Zonnestraling (standalone)

Geen externe dependencies. Geeft een dict terug met alle zonnegrootheden.

```python
result = solar_radiation(dt, lat, lon, pressure_hpa=1013.25, temperature_c=15.0, turbidity=2.5)
```

**Retourneert:** `solar_elevation_deg`, `solar_azimuth_deg`, `zenith_deg`, `extraterrestrial_irradiance`, `clear_sky_ghi`, `clear_sky_dni`, `clear_sky_dhi`, `equation_of_time_min`, `solar_noon_utc`, `sunrise_utc`, `sunset_utc`.

**Algoritmen:** Spencer (1971) voor declinatie, Bennett (1982) voor refractiecorrectie, Bird & Hulstrom (1981) voor clear-sky GHI/DNI/DHI.

---

### `wbgt_solar_app.py` — Solar radiation Streamlit-tab

Wrapper die `solar_radiation()` aanroept en het resultaat visualiseert.

```python
solar_wrapper(lat, lon, utc_dt, loc_name, selected_date, selected_time, tz, LOCATIONS)
```

Toont KPI-metrics (elevatie, GHI, zonsopkomst/-ondergang), een dagcurve per kwartier, een detailtabel en een vergelijking over alle locaties.

**Hulpfuncties:**
- `utc_decimal_to_local_str(utc_decimal, tz, ref_date)` — decimaal UTC-uur → HH:MM lokaal
- `compute_day_curve(lat, lon, selected_date, tz)` → DataFrame met GHI/DNI/DHI per kwartier

---

### `wbgt_liljegren.py` — WBGT engine · pure Python

Volledige Liljegren-implementatie in pure Python, zonder Cython-afhankelijkheden. Meest leesbaar; geschikt voor testen en verificatie.

**Publieke API:**

```python
wbgt = wbgt_liljegren(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa=1013.25)
wbgt = wbgt_liljegren_from_station(temp_c, rh_pct, wind_ms, q_wm2, stn=260, dt=..., pressure_hpa=...)
```

**Interne functies (niet publiek):**

| Functie | Beschrijving |
|---------|-------------|
| `_globe_temp_liljegren(...)` | Boltemperatuur Tg via Newton-iteratie |
| `_nat_bol_temp_liljegren(...)` | Nat-boltemperatuur Tnwb via Newton-iteratie |
| `_coszda(dt_end, lat, lon, interval_h)` | Gemiddelde cosinus-zenithoek over daglicht-interval |
| `_fdir(S, theta, d_AU)` | Fractie directe straling |
| `solar_zenith_angle(dt, lat_deg, lon_deg)` | Directe zenithoek (°) |

---

### `wbgt_liljegren_c_code.py` — WBGT engine · C-code port

Nauwkeurige vertaling van de originele Liljegren C-code naar Python. Bevat alle tussenliggende functies (`esat`, `viscosity`, `thermal_cond`, `h_sphere_in_air`, `Tglobe`, `Twb`, `stab_srdt`, `est_wind_speed`).

**Publieke API:**

```python
wbgt = wbgt_liljegren_opus(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa=1013.25)
wbgt = wbgt_liljegren_from_station_opus(temp_c, rh_pct, wind_ms, q_wm2, stn=260, dt=..., pressure_hpa=...)
```

Intern wordt `calc_solar_parameters()` + `calc_wbgt()` aangeroepen.

---

### `wbgt_liljegren_cython_wrapper.py` — WBGT engine · Cython-wrapper

Dun wrapper-laagje dat de Kong & Huber (2022) Cython-implementatie aanroept via de Python-vertalingen in `wbgt_liljegren_from_cython` en `wbgt_coszenith_from_cython`.

**Publieke API:**

```python
wbgt = wbgt_liljegren_from_cython(temp_c, rh_pct, wind_ms, q_wm2, lat, lon, dt, pressure_hpa=1013.25)
wbgt = wbgt_liljegren_from_station_cython(temp_c, rh_pct, wind_ms, q_wm2, stn=260, dt=..., pressure_hpa=...)
```

**Interne stappen:**
1. Bereken `cosz` (instantaan) via `wbgt_coszenith_from_cython.cosz()`
2. Bereken `cosza` en `coszda` (intervalgemiddelden)
3. Bereken `fdir` via `wbgt_liljegren_from_cython.fdir()`
4. Roep `WBGT_Liljegren()` aan
5. Converteer K → °C

**Bevat ook:** `KNMI_STATIONS` — dict met stationnummer → (lat, lon, hoogte_m) voor alle 35 KNMI-stations.

---

### `wbgt_liljegren_from_cython.py` — Numerieke kernel (Kong & Huber)

Python-vertaling van de originele Cython-code door Qinqin Kong (2021). Werkt met 3D NumPy-arrays `(T, Y, X)`.

**Publieke functies:**

| Functie | Invoer | Uitvoer |
|---------|--------|---------|
| `WBGT_Liljegren(tas, hurs, ps, sfcwind, rsds, fdir_arr, cosz_arr, is2mwind)` | 3D arrays | WBGT in K (3D) |
| `WBGT_GCM(tas, hurs, ps, sfcwind, rsds, rsus, rlds, rlus, fdir_arr, cosz_arr, is2mwind)` | 3D arrays | WBGT in K (GCM-variant) |
| `Tg_Liljegren(...)` | 3D arrays | Boltemperatuur Tg (K) |
| `Tnwb_Liljegren(...)` | 3D arrays | Nat-boltemperatuur Tnwb (K) |
| `fdir(cza_arr, czda_arr, rsds, date)` | 3D arrays | Fractie directe straling |

**Eenheden:** temperatuur in K, druk in Pa, wind in m/s, straling in W/m².

---

### `wbgt_coszenith_from_cython.py` — Numerieke kernel: cosinus-zenithoek

Python-vertaling van de Cython coszenith-routines. Werkt met `datetime64`-arrays en 2D lat/lon-arrays in radialen.

**Publieke functies:**

| Functie | Beschrijving |
|---------|-------------|
| `cosz(date, lat, lon)` | Instantane cosinus-zenithoek |
| `cosza(date, lat, lon, interval)` | Gemiddelde cosinus-zenithoek over interval |
| `coszda(date, lat, lon, interval)` | Gemiddelde cosinus-zenithoek alleen over daglicht-deel van interval |

**Invoerformaat:**
```python
date  = np.array([np.datetime64('2026-07-25T12:00')], dtype='datetime64[ns]')  # shape (T,)
lat2d = np.array([[math.radians(52.0)]])   # shape (Y, X), in radialen
lon2d = np.array([[math.radians( 5.2)]])   # shape (Y, X), in radialen
```

---

### `wbgt_replicate_knmi.py` — Historische data (1991–2025)

Laadt een CSV met preprocessed KNMI WBGT-data en toont statistieken, histogrammen en boxplots per maand.

**Publieke API:**

```python
show_historical_data()   # Streamlit-tab
```

**Interne pipeline:**
1. `prepare_data()` — laad + filter CSV
2. `get_data()` — cach het DataFrame in `st.session_state`
3. `referentie_tabel_based_on_history(df)` — heatmap op basis van historische data
4. Diverse plot-functies: `histogram_risico`, `histogram_wbgt`, `boxplot_wbgt_per_maand`, `toon_verdeling_waardes`, `scatterplots`

---

### `wbgt_vergelijk_script_met_knmi.py` — Validatie

Vergelijkt de eigen WBGT-berekening met de officiële KNMI WBGT-dataset door een steekproef te downloaden en te mergen.

**Publieke API:**

```python
vergelijk_script_met_knmi_download()   # Streamlit-tab
```

**Interne pipeline:**
1. `download_voor_vergelijking()` — haal KNMI-data op via API
2. `laad_en_merge(pad_knmi)` — merge met eigen berekening via `wbgt_bereken_df()`
3. `scatter_plot()` / `regressie_plot()` / `in_de_tijd_plot()` — visualisaties

---

### `wbgt_download_fulldataset_knmi.py` — KNMI bulk download

Standalone script (geen Streamlit). Downloadt het volledige KNMI WBGT-dataset asynchroon en bouwt er een CSV van.

```bash
python wbgt_download_fulldataset_knmi.py
```

**Functies:**

| Functie | Beschrijving |
|---------|-------------|
| `main_download()` | Async download van alle CSV-bestanden via KNMI Open Data API |
| `make_dataframe()` | Samenvoegen van losse bestanden tot één DataFrame per station |
| `list_dataset_files(...)` | Paginering over KNMI API-bestandenlijst |
| `download_dataset_file(...)` | Download van één bestand met overwrite-check |

**Let op:** vervang de `api_key` door je eigen KNMI-sleutel. Pas `download_directory` en `STATION` aan.

---

### `utils.py` — Generieke hulpfuncties

Gedeelde utilities voor het bredere KNMI-dashboard.

**Relevante functies voor WBGT:**

| Functie | Beschrijving |
|---------|-------------|
| `calculate_heat_index(T, RH)` | Rothfusz heat index (°C) |
| `calculate_wind_chill(T, V)` | Windchill (°C) |
| `rh2q(rh, t, p)` | Relatieve vochtigheid → specifieke vochtigheid |
| `getdata_wrapper(stn, fromx, until)` | KNMI uurdata ophalen |
| `get_weerstations()` | Lijst van KNMI-stations |

---

## Installatie en gebruik

```bash
pip install streamlit pandas numpy plotly scikit-learn requests
streamlit run wbgt_knmi.py
```

Voor de Cython-kernels (`wbgt_liljegren_from_cython`, `wbgt_coszenith_from_cython`): deze zijn vertaald naar pure Python en hebben **geen** Cython-compilatie nodig.

---


## Gegevensstroom: één WBGT-berekening

```
Gebruiker geeft in: T=32°C, RH=60%, wind=2 m/s, lat=52°, lon=5°, dt=2026-07-25 14:00 UTC
          │
          ▼
wbgt_utils.wbgt_buiten()
          │  roept aan
          ▼
wbgt_liljegren_cython_wrapper.wbgt_liljegren_from_cython()
          │
          ├── wbgt_coszenith_from_cython.cosz()   → cos(θ) instantaan
          ├── wbgt_coszenith_from_cython.cosza()  → gemiddelde over interval
          ├── wbgt_coszenith_from_cython.coszda()  → gemiddelde over daglichtdeel
          │
          ├── wbgt_liljegren_from_cython.fdir()   → fractie directe straling
          │
          └── wbgt_liljegren_from_cython.WBGT_Liljegren()
                    │
                    ├── _Tg_Liljegren_core()   → boltemperatuur Tg
                    └── _Tnwb_Liljegren_core() → nat-boltemperatuur Tnwb
                    │
                    └── WBGT = 0.7·Tnwb + 0.2·Tg + 0.1·T_lucht
                    
          ▼
     WBGT = 31.4 °C  (voorbeeld)
```

---

## Referenties

- Kong, Q. & Huber, M. (2022). *Explicit Calculations of Wet Bulb Globe Temperature Compared with Approximations and Why It Matters for Labor Productivity.* Earth's Future. https://doi.org/10.1029/2021EF002334  
- Liljegren, J.C. et al. — originele WBGT C-implementatie  
- Spencer, J.W. (1971). Fourier series representation of the position of the sun. *Search*, 2(5), 172.  
- Bird, R.E. & Hulstrom, R.L. (1981). *A simplified clear sky model for direct and diffuse insolation on horizontal surfaces.* SERI/TR-642-761.  
- Bennett, A.H. (1982). Refraction correction formulas.

- Pereira Marghidan, C., Mokkenstorm, L., (2026) Hittewaarschuwingen: Doorontwikkeling Nationaal Hitteplan RIVM en verdere integratie met waarschuwingssystematiek, KNMI De Bilt,| Wetenschappelijk rapport; WR-26-02 https://cdn.knmi.nl/system/data_center_publications/files/000/072/492/original/WR26-02.pdf?1775760740

- Pereira Marghidan C. et al. (2026) Van Wet Bulb Globe Temperature (WBGT) naar hittekracht
KNMI number: TR-26-04, https://cdn.knmi.nl/system/data_center_publications/files/000/072/495/original/TR-26-04.pdf?1779885454
