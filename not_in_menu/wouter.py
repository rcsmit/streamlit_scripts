import streamlit as st
import math

st.set_page_config(page_title="Nederlandse Melkvee Meststromen", layout="wide")

# ---------- Data ----------
items = [
    {"icon": "🐄", "label": "Melkkoeien",        "value": "1,580,000"},
    {"icon": "🐄", "label": "Jongvee Vlees",     "value": "85,000"},
    {"icon": "🐄", "label": "Jongvee Melk",      "value": "950,000"},
    {"icon": "🐄", "label": "Kalveren Vlees",    "value": "650,000"},
    {"icon": "🐄", "label": "Kalveren Melk",     "value": "180,000"},
    {"icon": "🐂", "label": "Vleesvee",          "value": "120,000"},
    {"icon": "🌱", "label": "Grasland",          "value": "1,000,000 ha"},
    {"icon": "🌾", "label": "Voedergewassen",    "value": "212,000 ha"},
]

# ---------- Instelling ----------
cols_per_row = 3  # aantal label+value paren per rij (bijv. 2, 3, 4)

# ---------- Styles ----------
st.markdown(f"""
<style>
.banner {{
  background: #2f9e44;
  color: white;
  padding: 28px 34px;
  border-radius: 14px;
  font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}}
.banner h1 {{
  margin: 0;
  font-size: 40px;
  line-height: 1.1;
  font-weight: 800;
}}
.banner p {{
  margin: 8px 0 0 0;
  opacity: .95;
  font-size: 16px;
}}
.card {{
  background: #f7f7f7;
  border: 1px solid #e9ecef;
  border-left: 6px solid #2f9e44;
  border-radius: 12px;
  padding: 16px 18px;
}}
.card table {{
  width: 100%;
  border-collapse: collapse;
}}
.card td {{
  padding: 8px 10px;
  vertical-align: top;
  font-size: 16px;
  border: none; /* geen cel-randen */
  width: {100 / (cols_per_row * 2)}%;
}}
.label {{
  font-weight: 700;
}}
.value {{
  font-weight: 600;
  color: #495057;
}}
</style>
""", unsafe_allow_html=True)

# ---------- Banner ----------
st.markdown("""
<div class="banner">
  <h1>🐐 Nederlandse Melkvee Meststromen V3.3</h1>
  <p>GEFIXTE N-BALANS  Geen Dubbeltelling Kalvermest Verwerking  Perfect Sluitende Massa Balans</p>
</div>
""", unsafe_allow_html=True)

st.write("")  # spacer

# ---------- Dynamisch tabel HTML bouwen ----------
rows_html = ""
for i in range(0, len(items), cols_per_row):
    row_cells = ""
    for j in range(cols_per_row):
        idx = i + j
        if idx < len(items):
            row_cells += f'<td class="label">{items[idx]["icon"]} {items[idx]["label"]}</td>'
            row_cells += f'<td class="value">{items[idx]["value"]}</td>'
        else:
            row_cells += "<td></td><td></td>"
    rows_html += f"<tr>{row_cells}</tr>"

# ---------- Kaart met tabel ----------
st.markdown(f"""
<div class="card">
  <table>
    {rows_html}
  </table>
</div>
""", unsafe_allow_html=True)
