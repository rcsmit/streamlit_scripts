import re
import xml.etree.ElementTree as ET
from collections import Counter

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(
    page_title="Blog viewer",
    page_icon=":material/article:",
    layout="centered",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #f8f6f2; }
    [data-testid="stSidebar"]          { background-color: #f0ede8; }
    .article-meta { color: #888; font-size: 0.85rem; margin-bottom: 0.25rem; }
    </style>
    <script>
    (function suppressWebSocketError() {
        const _error = console.error.bind(console);
        console.error = (...args) => {
            const msg = String(args[0] ?? '').toLowerCase();
            if (msg.includes('websocket')) return;
            _error(...args);
        };
        const observer = new MutationObserver(() => {
            document.querySelectorAll('[data-testid="stNotificationContentError"], [data-testid="stNotificationContentWarning"]').forEach(el => {
                const text = (el.innerText || '').toLowerCase();
                if (text.includes('websocket') || text.includes('connection')) {
                    const toast = el.closest('[data-testid="toastContainer"], .stToast, [class*="Toast"]');
                    if (toast) toast.style.display = 'none';
                }
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def read() -> pd.DataFrame:
    sheet_name = "gegevens"
    sheet_id = "1R5YDxVqpT1brUHz1P-Zjoyz0iHgLBJUIsSrbH9IfW5c"
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}"
        f"/gviz/tq?tqx=out:csv&sheet={sheet_name}"
    )
    df = pd.read_csv(
        url, delimiter=",", header=0, usecols=list(range(10)),
        names=["id","id_entry_original","titel","kopfoto","artikel",
               "datum","afbeelding","link","blog","categorie"],
    )
    df=df[df["blog"]!="CrazyWaiter"].copy()  # Exclude CW entries from sheet (we'll fetch them via RSS)
    df=df[df["blog"]!="YepYoga"].copy()  # Exclude CW entries from sheet (we'll fetch them via RSS)
    
    try:
        df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")
    except ValueError:
        df["datum"] = pd.to_datetime(df["datum"], format="mixed")
    df["jaar"]   = df["datum"].dt.strftime("%Y")
    df["maand"]  = df["datum"].dt.strftime("%m")
    df["maand_"] = df["datum"].dt.strftime("%Y-%m")
    return df


@st.cache_data(ttl=3600)
def fetch_rss_as_df(url: str, blog_name: str) -> pd.DataFrame:
    NS = {
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc":      "http://purl.org/dc/elements/1.1/",
    }
    empty = pd.DataFrame(columns=[
        "id","id_entry_original","titel","kopfoto","artikel",
        "datum","afbeelding","link","blog","categorie","jaar","maand","maand_",
    ])
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        st.sidebar.warning(f"RSS feed unavailable: {e}")
        return empty

    rows = []
    for i, item in enumerate(root.findall(".//item")):
        title    = item.findtext("title", "").strip()
        link     = item.findtext("link", "").strip()
        pub_date = item.findtext("pubDate", "").strip()
        content  = item.findtext(f"{{{NS['content']}}}encoded", "").strip()
        if not content:
            content = item.findtext("description", "").strip()

        image_url = "_"
        mc = item.find("{http://search.yahoo.com/mrss/}content")
        if mc is not None:
            image_url = mc.get("url", "_")
        if image_url == "_":
            enc = item.find("enclosure")
            if enc is not None:
                image_url = enc.get("url", "_")
        if image_url == "_" and content:
            m = re.search(r'<img[^>]+src=["\'](https?://[^"\']+)["\'][^>]*>', content)
            if m:
                image_url = m.group(1)

        try:
            datum = pd.to_datetime(pub_date, format="%a, %d %b %Y %H:%M:%S %z", utc=True).tz_localize(None)
        except Exception:
            try:
                datum = pd.to_datetime(pub_date, utc=True).tz_localize(None)
            except Exception:
                datum = pd.NaT

        rows.append({
            "id": f"rss_{i}", "id_entry_original": "_",
            "titel": title, "kopfoto": "_", "artikel": content,
            "datum": datum, "afbeelding": image_url, "link": link,
            "blog": blog_name,
            "categorie": ", ".join(c.text.strip() for c in item.findall("category") if c.text) or "_",
        })

    if not rows:
        return empty
    df = pd.DataFrame(rows)
    df["jaar"]   = df["datum"].dt.strftime("%Y")
    df["maand"]  = df["datum"].dt.strftime("%m")
    df["maand_"] = df["datum"].dt.strftime("%Y-%m")
    return df


def clean_artikel(series: pd.Series) -> pd.Series:
    series = series.astype(str)
    series = series.str.replace("_x000D_", "", regex=False)
    series = series.str.replace(
        r"http://www\.yepcheck\.com/printbak/",
        "https://github.com/rcsmit/streamlit_scripts/tree/main/printbak/",
        regex=True,
    )
    series = series.str.replace(r"<P>", "", regex=True)
    series = series.str.replace(r"</P>", "\n", regex=True)
    return series


def show_statistics(df_full: pd.DataFrame) -> None:
    """Statistics tab — always uses the complete unfiltered dataset."""
    TEAL = "Teal"
    BG   = "rgba(0,0,0,0)"

    df_s = df_full[df_full["datum"].notna()].copy()
    df_s["jaar_int"]  = df_s["datum"].dt.year
    df_s["jaar"]      = df_s["jaar_int"].astype(str)
    df_s["maand_num"] = df_s["datum"].dt.month
    df_s["weekday"]   = df_s["datum"].dt.day_name()
    df_s["word_count"] = df_s["artikel"].apply(
        lambda x: len(re.sub(r"<[^>]+>", " ", str(x)).split())
    )
    all_cats = [c for cats in df_s["categorie_list"] for c in cats]
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.subheader("Overview")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total posts",       len(df_s))
    k2.metric("Blogs / sources",   df_s["blog"].nunique())
    k3.metric("Unique categories", len(set(all_cats)))
    k4.metric("Years active",      df_s["jaar_int"].nunique())
    k5.metric("Avg words / post",  int(df_s["word_count"].mean()))
    st.divider()

    # ── Posts per year, stacked by blog ───────────────────────────────────
    posts_year = (
        df_s.groupby(["jaar", "blog"]).size()
        .reset_index(name="count").sort_values("jaar")
    )
    fig_year = px.bar(
        posts_year, x="jaar", y="count", color="blog",
        title="Posts per year, per blog",
        labels={"jaar": "Year", "count": "Posts", "blog": "Blog"},
        barmode="stack",
    )
    fig_year.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    st.plotly_chart(fig_year, use_container_width=True)

    col_l, col_r = st.columns(2)

    # ── Posts per blog (donut) ────────────────────────────────────────────
    with col_l:
        posts_blog = df_s.groupby("blog").size().reset_index(name="count")
        fig_blog = px.pie(
            posts_blog, names="blog", values="count",
            title="Posts per blog", hole=0.45,
        )
        fig_blog.update_traces(textposition="inside", textinfo="percent+label")
        fig_blog.update_layout(paper_bgcolor=BG)
        st.plotly_chart(fig_blog, use_container_width=True)

    # ── Seasonal posting pattern ──────────────────────────────────────────
    with col_r:
        posts_month = df_s.groupby("maand_num").size().reset_index(name="count")
        posts_month["maand_naam"] = posts_month["maand_num"].apply(lambda m: month_names[m-1])
        fig_month = px.bar_polar(
            posts_month, r="count", theta="maand_naam",
            title="Seasonal posting pattern",
            color="count", color_continuous_scale=TEAL,
            category_orders={"maand_naam": month_names},
        )
        fig_month.update_layout(coloraxis_showscale=False, paper_bgcolor=BG)
        st.plotly_chart(fig_month, use_container_width=True)

    # ── Cumulative posts over time ────────────────────────────────────────
    df_cum = df_s.sort_values("datum")[["datum"]].assign(n=1).reset_index(drop=True)
    df_cum["cumsum"] = df_cum["n"].cumsum()
    fig_cum = px.area(
        df_cum, x="datum", y="cumsum",
        title="Cumulative posts over time",
        labels={"datum": "Date", "cumsum": "Total posts"},
        color_discrete_sequence=["#2a9d8f"],
    )
    fig_cum.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    st.plotly_chart(fig_cum, use_container_width=True)

    col_a, col_b = st.columns(2)

    # ── Top 15 categories ────────────────────────────────────────────────
    with col_a:
        cat_counts = Counter(all_cats)
        top_cats = pd.DataFrame(cat_counts.most_common(15), columns=["category","count"])
        fig_cats = px.bar(
            top_cats.sort_values("count"), x="count", y="category",
            orientation="h", title="Top 15 categories",
            labels={"count": "Posts", "category": ""},
            color="count", color_continuous_scale=TEAL,
        )
        fig_cats.update_layout(coloraxis_showscale=False, plot_bgcolor=BG, paper_bgcolor=BG)
        st.plotly_chart(fig_cats, use_container_width=True)

    # ── Posts by day of week ──────────────────────────────────────────────
    with col_b:
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        posts_day = (
            df_s.groupby("weekday").size()
            .reindex(day_order, fill_value=0)
            .reset_index()
        )
        posts_day.columns = ["day","count"]
        fig_day = px.bar(
            posts_day, x="day", y="count",
            title="Posts by day of week",
            labels={"day": "", "count": "Posts"},
            color="count", color_continuous_scale=TEAL,
        )
        fig_day.update_layout(coloraxis_showscale=False, plot_bgcolor=BG, paper_bgcolor=BG)
        st.plotly_chart(fig_day, use_container_width=True)

    # ── Word count distribution ───────────────────────────────────────────
    wc_cap = df_s["word_count"].quantile(0.95)
    fig_words = px.histogram(
        df_s[df_s["word_count"] < wc_cap],
        x="word_count", color="blog", nbins=40,
        title="Post length distribution (word count, excl. top 5%)",
        labels={"word_count": "Words", "count": "Posts"},
        barmode="overlay", opacity=0.7,
    )
    fig_words.update_layout(plot_bgcolor=BG, paper_bgcolor=BG)
    st.plotly_chart(fig_words, use_container_width=True)

    # ── Heatmap: posts per year × month ──────────────────────────────────
    heat = (
        df_s.groupby(["jaar_int", "maand_num"]).size()
        .reset_index(name="count")
    )
    heat_pivot = (
        heat.pivot(index="maand_num", columns="jaar_int", values="count")
        .fillna(0)
    )
    heat_pivot.index = [month_names[i-1] for i in heat_pivot.index]
    fig_heat = px.imshow(
        heat_pivot,
        title="Posts heatmap — month × year",
        labels={"x": "Year", "y": "Month", "color": "Posts"},
        color_continuous_scale=TEAL, aspect="auto",
    )
    fig_heat.update_layout(paper_bgcolor=BG)
    st.plotly_chart(fig_heat, use_container_width=True)


def main():
    st.title(":material/article: Blog viewer")

    # ── Load all data ─────────────────────────────────────────────────────
    df_full = read()
    df_full = df_full.fillna("_")
    df_full["artikel"] = clean_artikel(df_full["artikel"])

    feeds = [
        ["https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/crazywaiter.xml", "crazywaiter"],
        ["https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/yepyoga.xml",    "yepyoga"],
        ["https://rene-smit.com/feed/?posts_per_page=100",                                                   "rene-smit.com"],
    ]
    for url, name in feeds:
        rss_df = fetch_rss_as_df(url, blog_name=name)
        if not rss_df.empty:
            rss_df["artikel"] = clean_artikel(rss_df["artikel"])
            df_full = pd.concat([df_full, rss_df], ignore_index=True)

    df_full = df_full.sort_values("datum", ascending=False, na_position="last").reset_index(drop=True)
    df_full["categorie_list"] = df_full["categorie"].apply(
        lambda x: [c.strip() for c in x.split(",") if c.strip() and c.strip() != "_"] if x != "_" else []
    )

    # ── Sidebar filters ───────────────────────────────────────────────────
    options = sorted(df_full["blog"].unique().tolist())
    selected_blogs = st.sidebar.multiselect("Select blog", options, options)

    if not selected_blogs:
        st.error("Select one or more blogs.", icon=":material/error:")
        st.stop()

    df = df_full[df_full["blog"].isin(selected_blogs)].copy()

    if len(selected_blogs) == 1 and selected_blogs[0] in ("CrazyWaiter", "YepYoga", "rene-smit.com"):
        all_categories = sorted({c for cats in df["categorie_list"] for c in cats})
        selected_categories = st.sidebar.multiselect("Select categories", all_categories, all_categories)
        if selected_categories:
            df = df[df["categorie_list"].apply(lambda cats: any(c in selected_categories for c in cats))]
        if df.empty:
            st.error("Choose a category.", icon=":material/filter_list:")
            st.stop()

    st.sidebar.markdown("---")
    zoekterm = st.sidebar.text_input("Search", placeholder="Zoeken...", label_visibility="visible")
    if zoekterm:
        mask = (
            df["titel"].str.contains(zoekterm, case=False, na=False)
            | df["artikel"].str.contains(zoekterm, case=False, na=False)
            | df["categorie"].str.contains(zoekterm, case=False, na=False)
        )
        df = df[mask]

    total = len(df)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### Pagination :gray-badge[{total}]")
    posts_per_page = st.sidebar.selectbox("Posts per page", [5, 10, 25, 50], index=1)
    n_pages = max(1, -(-total // posts_per_page))
    page    = st.sidebar.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
    start   = (page - 1) * posts_per_page
    df_page = df.iloc[start : start + posts_per_page]

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_articles, tab_stats = st.tabs([
        ":material/article: Articles",
        ":material/bar_chart: Statistics",
    ])

    with tab_articles:
        st.caption(f"Page {page} of {n_pages} · {total} article(s) in filter")

        for row in df_page.itertuples(index=False):
            with st.container(border=True):
                st.subheader(row.titel)
                meta = f':material/calendar_today: {row.datum.strftime("%d %B %Y")}'
                if row.categorie != "_":
                    meta += f" &nbsp;·&nbsp; :material/label: {row.categorie}"
                st.markdown(meta)

                if row.kopfoto != "_":
                    thumb = (
                        "https://raw.githubusercontent.com/rcsmit/"
                        f"streamlit_scripts/main/printbak/thumbnails/{row.kopfoto}"
                    )
                    st.image(thumb, use_container_width=False, width=480)
                elif row.afbeelding != "_" and row.afbeelding.startswith("http"):
                    st.image(row.afbeelding, use_container_width=False, width=480)

                st.markdown(row.artikel, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                if row.afbeelding != "_":
                    with col1:
                        st.link_button(":material/image: Afbeelding", row.afbeelding)
                if row.link != "_":
                    with col2:
                        st.link_button(":material/open_in_new: Link", row.link)

    with tab_stats:
        # Statistics always work on the complete df_full (all blogs, no filters)
        show_statistics(df_full)


if __name__ == "__main__":
    main()