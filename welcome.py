import streamlit as st
from catalogue import options, CATEGORIES


# ---------------------------------------------------------------------------
# welcome.py  —  Landing page for rcsmit.streamlit.app
# options + CATEGORIES are imported from catalogue.py (single source of truth)
# ---------------------------------------------------------------------------


def main():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #f0f4ff 0%, #e8f4f8 100%);
        border: 1px solid #d1dce8;
        border-radius: 16px;
        padding: 2.6rem 2.4rem 2rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        bottom: -50px; right: -50px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(78,205,196,0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #1a2340;
        margin: 0 0 0.2rem;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #5a6a82;
        margin: 0 0 1.2rem;
        font-weight: 300;
    }
    .hero-tag {
        display: inline-block;
        background: #ffffff;
        border: 1px solid #c7d4e8;
        color: #3b4f72;
        border-radius: 20px;
        padding: 3px 11px;
        font-size: 0.73rem;
        font-family: 'Space Mono', monospace;
        margin: 0 4px 5px 0;
    }
    .stats-row {
        display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1.4rem;
    }
    .stat-pill {
        background: #ffffff;
        border: 1px solid #d1dce8;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        text-align: center;
        min-width: 80px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stat-number {
        font-family: 'Space Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #2563eb;
    }
    .stat-label {
        font-size: 0.65rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* category cards */
    .cat-card {
        background: #ffffff;
        border: 1px solid #e4eaf2;
        border-radius: 12px;
        padding: 1.15rem 1.2rem 0.7rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .cat-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid currentColor;
        opacity: 1;
    }
    .script-row {
        display: flex;
        align-items: flex-start;
        gap: 0.55rem;
        padding: 0.32rem 0;
        border-bottom: 1px solid #f1f5f9;
    }
    .script-row:last-child { border-bottom: none; }
    .script-dot {
        width: 6px; height: 6px;
        border-radius: 50%;
        margin-top: 6px;
        flex-shrink: 0;
    }
    .script-num {
        font-family: 'Space Mono', monospace;
        font-size: 0.65rem;
        color: #b0bec5;
        margin-top: 2px;
        flex-shrink: 0;
        width: 20px;
    }
    .script-name {
        font-size: 0.845rem;
        font-weight: 600;
        color: #1e293b;
        line-height: 1.3;
    }
    .script-desc {
        font-size: 0.74rem;
        color: #94a3b8;
        line-height: 1.35;
    }

    .about-box {
        background: #f8faff;
        border: 1px solid #dce5f5;
        border-radius: 12px;
        padding: 1.6rem 1.8rem;
        margin-top: 2rem;
    }
    .about-box h3 {
        font-family: 'Space Mono', monospace;
        color: #2563eb;
        margin-top: 0;
        font-size: 0.95rem;
    }
    .about-box p { color: #475569; line-height: 1.7; font-size: 0.875rem; margin: 0 0 0.6rem; }
    .about-box a { color: #2563eb; text-decoration: none; font-weight: 500; }
    .about-box a:hover { text-decoration: underline; }

    footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero ─────────────────────────────────────────────────────────────
    tags = ["Python", "Streamlit", "Plotly", "CBS data", "KNMI", "Finance", "Open source"]
    tag_html = "".join(f'<span class="hero-tag">{t}</span>' for t in tags)

    n_scripts = len(options) - 1  # exclude welcome itself
    n_cats    = len(CATEGORIES) - 1
    stats_html = "".join(
        f'<div class="stat-pill"><div class="stat-number">{v}</div><div class="stat-label">{k}</div></div>'
        for k, v in [("Scripts", str(n_scripts)), ("Categories", str(n_cats)), ("Years active", "5+")]
    )

    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">René Smit</div>
        <div class="hero-sub">Data analyst · Streamlit builder · Nomadic freelancer</div>
        {tag_html}
        <div class="stats-row">{stats_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("👈 **Use the sidebar** to open any script, or browse the full catalogue below.")
    st.markdown("---")

    # ── Category grid — driven entirely by CATEGORIES + options ──────────
    cols = st.columns(2, gap="medium")
    col_idx = 0
    readme_txt = ""
    for letter, cat_name, cat_indices, color in CATEGORIES:
        if cat_indices == [0]:   # skip Home meta-entry
            continue

        scripts_html = ""
        readme_txt_cat = ""
        for idx in cat_indices:
            label = options[idx][0]
            desc  = options[idx][2]
            num   = label.split("]")[0].replace("[", "").strip()
            name  = label.split("] ", 1)[-1]
            scripts_html += f"""
<div class='script-row'>
<div class='script-dot' style='background:{color}'></div>
<div class='script-num'>{num}</div>
<div>
<div class='script-name'>{name}</div>
<div class='script-desc'>{desc}</div>
</div>
</div>"""
            readme_txt_cat += f"""
| {num} | [{name}](https://rcsmit.streamlit.app/?choice={num}) | {desc} |"""

        cols[col_idx % 2].markdown(f"""
<div class='cat-card'>
<div class='cat-header' style='color:{color}; border-color:{color}'>
{cat_name}
</div>
{scripts_html}
</div>
        """, unsafe_allow_html=True)

        readme_txt += f"""
### {cat_name}
| # | Script | Description |
|---|--------|-------------|{readme_txt_cat}
"""
        col_idx += 1

    # ── About ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-box">
        <h3>👋 About</h3>
        <p>
            I'm René — a Dutch multidisciplinary freelancer with a nomadic lifestyle across
            Da Nang, Chiang Mai, Bali, and the Netherlands.
            This portfolio is a living collection of Streamlit apps built over the years:
            data explorations, financial tools, trading backtests, Dutch open data, maps, and the occasional bit of fun.
        </p>
        <p>
            Most scripts pull from Dutch open data (CBS, KNMI, Eurostat) or public APIs and are written
            in Python with Streamlit, Plotly, and pandas. Everything is open source.
        </p>
        <p>
            These scripts were developed over several years and most are not actively maintained. 
            External data sources and Python dependencies change over time, which may affect 
            the functionality of individual tools. If you encounter an issue or have a specific 
            use case in mind, please feel free to reach out.
        </p>
        <p>
            🔗 <a href="https://github.com/rcsmit/streamlit_scripts" target="_blank">GitHub</a> &nbsp;·&nbsp;
            🌐 <a href="https://rene-smit.com" target="_blank">rene-smit.com</a> &nbsp;·&nbsp;
            📊 <a href="https://rcsmit.streamlit.app" target="_blank">rcsmit.streamlit.app</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Contents for readme.MD"):
        st.code(readme_txt)


if __name__ == "__main__":
    main()
