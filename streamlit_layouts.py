import streamlit as st
import re
import plotly.express as px
import pandas as pd


def is_valid_hex_color(value: str) -> bool:
    """Check if string is a valid hex color (#RGB, #RRGGBB, #RRGGBBAA)."""
    return bool(re.match(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", value))


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

    col1,col2,col3= st.columns(3)
    with col1:
        spot_color = st.color_picker("Spot color", "#1e88e5")
    with col2:
        second_color = st.color_picker("Second color", "#764ba2")
    with col3:
        card_background_color = st.color_picker("Card background color", "#f9fafb")  # lichte achtergrond voor kaarten
    # one if-statement for both
    if not (is_valid_hex_color(spot_color) and is_valid_hex_color(second_color)):
        st.error("One or both color codes are invalid. Use #RGB, #RRGGBB, or #RRGGBBAA.")

    # ---------- Styles ----------
    st.markdown(f"""
    <style>
    .banner {{
      background: {spot_color};
      color: white;
      padding: 28px 34px;
      border-radius: 14px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }}
    .category-header {{ background: linear-gradient(90deg, {spot_color} 0%, {second_color} 100%); 
                  color: white; 
                  padding: 5px 10px; 
                  border-radius: 10px; 
                  margin: 10px 0 5px 0;}}
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
      background: {card_background_color};
      border: 1px solid #e9ecef;
      border-left: 6px solid {spot_color}; /* Accent left border */
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .card table {{
      width: 100%;
      border-collapse: collapse;
    }}
    .card td {{
      padding: 3px 3px;
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

    # ---------- Kaart met tabel ----------
    st.markdown(f"""
    <div class='category-header'>
      This is a category header
    </div>
    """, unsafe_allow_html=True)

    st.info("Inspired by Wouter de Heij @deheij (https://twitter.com/deheij)")
    st.info("Code : https://www.github.com/rcxsm/streamlit_scripts/blob/main/streamlit_layouts.py")

    st.write("")  # spacer  
    st.header("Another example with cards and plotly in columns")
    st.subheader("1) KPI-tegel met sparkline - Kleine KPI’s met een mini-grafiek.")

    kpi = pd.DataFrame({"x": list(range(20)), "y": [3,4,5,6,8,7,9,12,11,13,12,14,16,15,17,18,17,19,20,22]})
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="card">
        <div class="label">Revenue</div>
        <div class="value">€ 123k</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
        <div class="label">Users</div>
        <div class="value">8,942</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        fig = px.line(kpi, x="x", y="y")
        fig.update_layout(height=90, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("3) Statbadges")
    st.markdown(f"""
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <span style="background:{spot_color};color:white;padding:6px 10px;border-radius:999px;font-weight:600;">Active</span>
        <span style="background:{second_color};color:white;padding:6px 10px;border-radius:999px;font-weight:600;">Beta</span>
        <span style="background:#e9ecef;color:#111;padding:6px 10px;border-radius:999px;font-weight:600;">Archived</span>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("5) Progress-balken in tabel - Gebruik st.data_editor met kolomconfig.")
    import numpy as np
    df_prog = pd.DataFrame({
        "Task": [f"T{i}" for i in range(1,7)],
        "Progress": np.random.randint(10,100,6),
        "Owner": ["A","B","C","A","B","C"]
    })
    st.data_editor(
        df_prog,
        column_config={
            "Progress": st.column_config.ProgressColumn(
                "Progress", min_value=0, max_value=100, format="%d%%"
            )
        },
        hide_index=True
    )
    st.subheader("6 ) Cards-grid met auto-wrap - Responsief zonder vaste kolomtelling")
   


        
    
    # HTML cards opbouwen
    cards_html = "".join([
        f"""<div class="mini-card">
        <div class="mini-title">{it['icon']} {it['label']}</div>
        <div class="mini-val">{it['value']}</div>
        </div>""" for it in items
    ])

    # CSS met escaped braces in f-string
    st.markdown(f"""
    <style>
    .mini-wrap {{ display:flex; flex-wrap:wrap; gap:12px; }}
    .mini-card {{
                flex: 1 1 220px;
                background:#fff;
                border:1px solid #eee;
                border-left:6px solid {second_color};
                border-radius:12px;
                padding:12px 14px;
                transition: box-shadow .15s, transform .15s;
                }}
    .mini-card:hover {{ box-shadow:0 6px 20px rgba(0,0,0,.08); transform: translateY(-1px); }}
    .mini-title {{ font-weight:700; margin-bottom:4px; }}
    .mini-val {{ font-weight:600; color:#495057; }}
    </style>
    <div class='mini-wrap'>{cards_html}</div>
    """, unsafe_allow_html=True)
   

    st.subheader("7) Timeline component -    Voor gebeurtenissen of releases.")

    
    timeline = [
        {"date":"2025-01-10","title":"Kickoff","desc":"Scope en doelen"},
        {"date":"2025-02-02","title":"MVP","desc":"Eerste release"},
        {"date":"2025-03-15","title":"v1.0","desc":"Go live"},
    ]
    tl_html = "".join([
        f"""<div class='tl-item'>
                <div class='tl-dot' style='background:{spot_color};'></div>
                <div class='tl-content'><b>{t['date']} • {t['title']}</b><br/>{t['desc']}</div>
            </div>""" for t in timeline
    ])
    st.markdown(f"""
        <style>
        .tl-item{{position:relative;padding-left:24px;margin:10px 0;}}
        .tl-item:before{{content:'';position:absolute;left:7px;top:0;bottom:-10px;width:2px;background:{second_color};}}
        .tl-dot{{position:absolute;left:2px;top:4px;width:12px;height:12px;border-radius:999px;}}
        .tl-content{{background:{card_background_color};border:1px solid #eef0f2;border-radius:10px;padding:8px 10px;}}
        </style>
    """+f"<div>{tl_html}</div>", unsafe_allow_html=True)

    st.subheader("7a) Timeline component – Animated & modern")

    tl_html = "".join([
        f"""<div class="tl-item2" style="--d:{i*0.08:.2f}s">
    <div class="tl-dot2"></div>
    <div class="tl-content2">
        <b>{t['date']} • {t['title']}</b><br/>{t['desc']}
    </div>
    </div>""" for i, t in enumerate(timeline)
    ])

    # Calculate animation duration for the line (should finish before last item appears)
    line_duration = len(timeline) * 0.08 + 0.3 if timeline else 0.5

    st.markdown(f"""
    <style>
    :root {{
    --spot: {spot_color};
    --second: {second_color};
    --line-x: 24px;         /* x-positie van de lijn */
    --dot-size: 14px;       /* diameter van de bolletjes */
    }}

    /* Container */
    .timeline2 {{
    position: relative;
    margin: 8px 0;
    padding-left: calc(var(--line-x) + var(--dot-size) + 20px);
    }}

    /* Verticale lijn met groei-animatie */
    .timeline2:before {{
    content: "";
    position: absolute;
    left: var(--line-x);
    top: 0;
    width: 2px;
    background: var(--second);
    height: 0;
    animation: growLine {line_duration}s ease-out forwards;
    }}

    @keyframes growLine {{
    to {{ height: 100%; }}
    }}

    /* Items */
    .tl-item2 {{
    position: relative;
    margin: 20px 0;
    opacity: 0;
    transform: translateY(8px);
    animation: slideUp .45s ease-out forwards;
    animation-delay: var(--d, 0s);
    }}

    @keyframes slideUp {{
    to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Dot - gecentreerd op de lijn */
    .tl-dot2 {{
    position: absolute;
    top: 4px;  /* uitgelijnd met de eerste regel tekst */
    left: calc(-1 * (var(--line-x) + var(--dot-size) + 20px) + var(--line-x) - var(--dot-size)/2);
    width: var(--dot-size);
    height: var(--dot-size);
    border-radius: 50%;
    background: var(--spot);
    border: 2px solid white;
    box-shadow: 0 2px 8px rgba(0,0,0,.12);
    z-index: 2;
    opacity: 0;
    animation: fadeInDot .3s ease-out forwards;
    animation-delay: var(--d, 0s);
    }}

    @keyframes fadeInDot {{
    to {{ opacity: 1; }}
    }}

    /* Pulse effect op de dot */
    .tl-dot2:after {{
    content: "";
    position: absolute;
    top: -4px;
    left: -4px;
    right: -4px;
    bottom: -4px;
    border-radius: 50%;
    border: 2px solid var(--spot);
    opacity: 0;
    animation: pulse 2s ease-out infinite;
    animation-delay: calc(var(--d, 0s) + .3s);
    }}

    @keyframes pulse {{
    0% {{ transform: scale(0.8); opacity: 0.8; }}
    70% {{ transform: scale(1.5); opacity: 0; }}
    100% {{ transform: scale(1.5); opacity: 0; }}
    }}

    /* Content-blok */
    .tl-content2 {{
    background: #f9fafb;
    border: 1px solid #eef0f2;
    border-radius: 10px;
    padding: 12px 14px;
    transition: all .2s ease;
    }}

    .tl-item2:hover .tl-content2 {{
    transform: translateX(4px);
    box-shadow: 0 4px 12px rgba(0,0,0,.08);
    border-color: var(--second);
    }}

    .tl-content2 b {{
    color: #1f2937;
    }}
    </style>

    <div class="timeline2">
    {tl_html}
    </div>
    """, unsafe_allow_html=True)
    st.subheader("8) Metric-row met iconen     Helder en compact.")

    
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Visitors","23.4k","+4.2%")
    mc2.metric("Conversion","3.1%","+0.3%")
    mc3.metric("AOV","€ 42.10","-1.2%")
    mc4.metric("Churn","2.4%","+0.1%")
    st.subheader("9) Compacte compare-table - Goed voor A vs B.")

    
    compare = pd.DataFrame({
        "Metric":["Speed","Cost","Uptime","Satisfaction"],
        "A":[9.1, "€€", "99.9%", 4.6],
        "B":[8.7, "€",   "99.5%", 4.2],
    })
    st.markdown("<div class='card'><b>Compare</b></div>", unsafe_allow_html=True)
    st.dataframe(compare, hide_index=True, use_container_width=True)

    st.subheader("Zachte hover op cards")

    st.markdown(f"""<style>
    /* Zachte hover op cards */
    .card {{
        transition: all .25s ease;
        border-radius: 10px;
        padding: 16px;
        background: white;
        border: 1px solid #e5e7eb;
    }}

    .card:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,.08);
        transform: translateY(-2px);
        border-color: {spot_color};
    }}

    /* Headings consistent */
    h1, h2, h3 {{
        font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    }}

    /* Sticky header - werkt in Streamlit */
    .sticky-container {{
        position: relative;
        width: 100%;
        margin: 20px 0;
    }}

    .sticky-head {{
        position: sticky;
        top: 100px;  /* ruimte voor Streamlit header */
        z-index: 999;
        background: linear-gradient(90deg, {spot_color} 0%, {second_color} 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,.12);
        font-weight: 600;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
        overflow: visible
    }}

    /* Alternatief: Fixed header (blijft altijd zichtbaar) */
    .fixed-head {{
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        z-index: 999;
        background: linear-gradient(90deg, {spot_color} 0%, {second_color} 100%);
        color: white;
        padding: 12px 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,.15);
        font-weight: 600;
        margin: 0 auto;
        max-width: 1200px;  /* pas aan naar je layout breedte */
    }}

    /* Demo cards */
    .demo-card {{
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        margin: 12px 0;
        transition: all .25s ease;
    }}

    .demo-card:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,.08);
        transform: translateY(-2px);
        border-color: {spot_color};
    }}
    </style>
    """, unsafe_allow_html=True)

    # Optie 1: Sticky header (beweegt mee tot top bereikt)
    st.markdown("""<div class='sticky-container'>
        <div class='sticky-head'>📊 Sticky Summary Bar - Scrollt mee</div>
    </div>
    """, unsafe_allow_html=True)

    # Demo content om scrollen te tonen
    st.markdown("""<div class='demo-card'>
        <h3>Card 1</h3>
        <p>Deze cards hebben een zachte hover effect. Probeer erover te hoveren!</p>
    </div>
    <div class='demo-card'>
        <h3>Card 2</h3>
        <p>De sticky header blijft bovenaan staan terwijl je scrollt.</p>
    </div>
    <div class='demo-card'>
        <h3>Card 3</h3>
        <p>Scroll verder om het effect te zien...</p>
    </div>
    <div class='demo-card'>
        <h3>Card 4</h3>
        <p>De hover animatie is subtiel maar effectief.</p>
    </div>
    <div class='demo-card'>
        <h3>Card 5</h3>
        <p>Blijf scrollen...</p>
    </div>
    <div class='demo-card'>
        <h3>Card 6</h3>
        <p>De sticky header zou nu bovenaan moeten blijven.</p>
    </div>
    """, unsafe_allow_html=True)

    # Optie 2: Fixed header (alternatief - altijd zichtbaar)
    # Uncomment als je een always-visible header wilt:
    # st.markdown("""<div class='fixed-head'>🔒 Fixed Summary Bar - Altijd zichtbaar</div>
    # """, unsafe_allow_html=True)

    st.subheader("Popover voor snelle hulpmenu’s")

    try:
        with st.popover("Quick actions"):
            st.button("Export CSV")
            st.button("Refresh data")
            st.toggle("Auto update", value=True)
    except Exception:
        pass
    
if __name__ == "__main__":
    main()
    