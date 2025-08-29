import streamlit as st
def main():
    st.set_page_config(page_title="Demo Cards Layout", layout="wide")

    # ---------- Data (Lorem Ipsum) ----------
    items = [
        {"icon": "📘", "label": "Lorem",     "value": "123,456"},
        {"icon": "📗", "label": "Ipsum",     "value": "78,900"},
        {"icon": "📙", "label": "Dolor",     "value": "42,000"},
        {"icon": "📕", "label": "Sit",       "value": "11,111"},
        {"icon": "📓", "label": "Amet",      "value": "9,876"},
        {"icon": "📔", "label": "Consect",   "value": "55,432"},
        {"icon": "📒", "label": "Adipiscing","value": "7,654"},
        {"icon": "📚", "label": "Elit",      "value": "3,210"},
    ]

    # Stel aantal label+value paren per rij in
    cols_per_row = 3

    # ---------- Styles ----------
    st.markdown(f"""
    <style>
    .banner {{
      background: #1e88e5; /* blauw */
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
      border-left: 6px solid #1e88e5; /* blauwe balk */
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
      border: none;
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
      <h1>📊 Lorem Ipsum Demo Cards</h1>
      <p>Dolor sit amet — consectetur adipiscing elit. Integer nec odio. Praesent libero.</p>
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

    st.info("Inspired by Wouter de Heij @deheij (https://twitter.com/deheij)")
    st.info("Code : https://www.github.com/rcxsm/streamlit_scripts/blob/main/streamlit_layouts.py")

if __name__ == "__main__":
    main()
    