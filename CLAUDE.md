# CLAUDE.md — AI Assistant Guide for streamlit_scripts

## Repository Overview

A personal collection of 70+ Streamlit applications by René Smit (@rcsmit). The apps cover financial tools, Dutch weather/climate analysis, demographics, utilities, and miscellaneous generators. Most data sources and variable names are in Dutch.

**Entry point**: `menu_streamlit.py` — a single Streamlit app that dynamically imports and runs all other scripts via a sidebar radio menu.

---

## Architecture

### How the Menu Works

`menu_streamlit.py` maintains a hardcoded list of `[menu_label, module_name, description]` triples. When a user selects an option, it calls `importlib.import_module(module_name)` and then `module.main()`.

**Adding a new script to the menu:**
1. Create `your_script.py` in the root with a `def main():` function.
2. Add an entry to the `options` list in `menu_streamlit.py`:
   ```python
   ["[73] Your Script Title", "your_script", "Optional tooltip text"],
   ```
3. The index number in the label (e.g. `[73]`) must be unique and sequential.

### Every Script Must Have `main()`

All scripts in the root are loaded as modules. They **must** define a `def main():` function — this is what `menu_streamlit.py` calls. Do not use `if __name__ == "__main__":` as the sole entry point.

---

## Directory Structure

```
streamlit_scripts/
├── menu_streamlit.py          # Main entry point — dynamic menu
├── welcome.py                 # Landing page (menu item [0])
├── helpers.py                 # CSS heatmap helper, string utils (left/right/mid)
├── utils.py                   # Yahoo Finance data retrieval
├── requirements.txt           # Root-level dependencies
├── Dockerfile                 # Containerization (port 7860, for handstand_analyzer)
├── *.py                       # ~106 individual app scripts
│
├── show_knmi_functions/       # KNMI weather analysis subpackage
│   └── utils.py               # ~750 lines: data fetching, plotting helpers
│
├── st_maptoposter/            # Map poster generator subproject
│   ├── st_create_map_poster.py
│   ├── create_map_poster.py
│   ├── utils.py
│   ├── requirements.txt
│   ├── themes/                # 20+ JSON theme files
│   └── README.md
│
├── input/                     # Data files (CSV, XLSX, GeoJSON, GPX) — ~269 MB
│   └── meat_consumption/
├── not_in_menu/               # Experimental/unused scripts (~23 files)
├── cache/                     # Application cache
└── printbak/                  # Print/export artifacts
```

**`not_in_menu/`** — Scripts that are not wired into the menu. Treat as experimental or legacy. Do not add these to the menu without reviewing them.

---

## Common Patterns and Conventions

### Platform Detection (Local vs Cloud)

Many scripts detect whether they're running locally (Windows) or on Streamlit Cloud:

```python
import platform

if platform.processor() != "":
    # Running locally on Windows
    local_path = "C:\\Users\\rcxsm\\Documents\\python_scripts\\streamlit_scripts\\"
else:
    # Running on Streamlit Cloud
    pass
```

Use this pattern when file paths differ between environments.

### Caching

Use `@st.cache_data()` for expensive data-fetching functions (API calls, file reads, computations):

```python
@st.cache_data()
def get_data(...):
    ...
```

### Page Config

Scripts loaded via the menu should wrap `st.set_page_config()` in a try/except because the menu already sets it:

```python
try:
    st.set_page_config(layout='wide')
except:
    pass
```

### Error Handling with `st.stop()`

When data is missing or invalid, show an error and halt execution:

```python
if len(df) == 0:
    st.error(f"No data or wrong input - {choice}")
    st.stop()
```

### DataFrame Conventions

- Use `pandas` as the primary DataFrame library; `polars` is used in some newer scripts.
- Reset index after operations: `df.reset_index(inplace=True)`
- Dutch column names are common (e.g. `"Koers"` for price, `"Datum"` for date).

---

## Utility Modules

| Module | Purpose |
|--------|---------|
| `helpers.py` | `cell_background_helper()` for DataFrame heatmaps; `left()`, `right()`, `mid()` string helpers |
| `utils.py` | `get_data_yfinance()` — Yahoo Finance data with SMA calculation |
| `knmi_utils.py` | KNMI Dutch weather data fetching and caching |
| `inkomstenbelasting_helpers.py` | Dutch income tax calculation helpers |
| `show_knmi_functions/utils.py` | Weather station data, KNMI API calls, visualization helpers |
| `st_maptoposter/utils.py` | Map styling, SVG handling, poster generation utilities |

---

## Key Dependencies

Pinned or notable packages in `requirements.txt`:

- `numpy==1.26.4` — **pinned** to prevent binary incompatibility with compiled extensions
- `streamlit` — core UI framework
- `pandas`, `polars` — data manipulation
- `plotly`, `matplotlib`, `seaborn` — visualization
- `scipy`, `scikit-learn`, `statsmodels` — statistics and ML
- `yfinance` — Yahoo Finance data
- `openai` — Claude/GPT-powered features
- `cbsodata` — Dutch Central Bureau of Statistics API
- `geopandas`, `folium`, `streamlit_folium` — geographic data
- `PyMuPDF` — PDF reading
- `gpxpy`, `geopy` — GPS track analysis
- `astral` — sunrise/sunset calculations
- `deep_translator` — translation utilities

---

## Development Workflow

### Running Locally

```bash
streamlit run menu_streamlit.py
```

To run a single script directly:

```bash
streamlit run your_script.py
```

### Branch Convention

- Default remote branch: `main`
- Feature branches: `claude/<description>-<ID>`
- Never push directly to `main`

### Installing Dependencies

```bash
pip install -r requirements.txt
```

For the map poster subproject:
```bash
pip install -r st_maptoposter/requirements.txt
```

### Sensitive Files

The following are excluded from git (see `.gitignore`):
- `keys.py` — API keys and secrets; **never commit this file**
- `.streamlit/` — local Streamlit config (may contain secrets)
- `prullenbak/` — trash/scratch directory
- `gitnore/` and `input/gitnore/` — ignored data

---

## Code Style and Conventions

- **Language**: Mostly English for code; Dutch for domain-specific labels, variable names (e.g. `"temperatuur"`, `"neerslag"`), and UI text.
- **Function naming**: `snake_case` throughout.
- **Imports**: Standard library first, then third-party, then local modules. No enforced formatter (no black/ruff config present).
- **Comments**: Inline comments in both English and Dutch.
- **Legacy code**: Old versions are often kept with suffixes like `_old.py`, `_Copy.py`, or `2025`/`2026` date suffixes. Do not delete without confirming they are truly unused.
- **Bare excepts**: Common in this codebase (`except: pass`) — do not flag these as errors when modifying files.

---

## Testing

- `test_scripts_streamlit.py` — basic import/run tests for scripts
- `not_in_menu/test_bewerk_df.py` — DataFrame utility tests
- No pytest configuration; no CI/CD pipeline detected.

When modifying a script, manually verify by running `streamlit run menu_streamlit.py` and selecting the relevant menu item.

---

## Subprojects

### `show_knmi_functions/`
A package of modules for KNMI (Dutch meteorological institute) data analysis. Each visualization type is a separate module (e.g. `polar_plot.py`, `show_warmingstripes.py`). Import from `show_knmi_functions.<module>`.

### `st_maptoposter/`
A standalone Streamlit app for generating custom map posters. Has its own `requirements.txt` and `README.md`. Run with `streamlit run st_maptoposter/st_create_map_poster.py`.

---

## Important Notes for AI Assistants

1. **Always implement `def main()`** — every script in the root must expose this function.
2. **Do not rename Dutch variables** without understanding the domain context; many names map directly to Dutch official data field names.
3. **Respect the numpy pin** — do not upgrade `numpy` beyond `1.26.4` without testing; it prevents a known binary incompatibility.
4. **`keys.py` is secret** — if a script imports from `keys.py`, do not create or modify that file; ask the user to provide credentials separately.
5. **`not_in_menu/` is experimental** — scripts there may be incomplete or broken; treat with caution.
6. **Platform detection is intentional** — the `platform.processor() != ""` checks are the established way to detect local vs cloud; do not remove them.
7. **Commented-out code is often intentional** — legacy fallback imports and disabled features are kept for reference; do not delete them unless asked.
