import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Blog viewer",
    page_icon=":material/article:",
    layout="centered",
)

# Light background + card styling + suppress WebSocket error toast
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
        url,
        delimiter=",",
        header=0,
        usecols=list(range(10)),
        names=["id", "id_entry_original", "titel", "kopfoto", "artikel",
               "datum", "afbeelding", "link", "blog", "categorie"],
    )
    df=df[df["blog"]!="CrazyWaiter"]
    df=df[df["blog"]!="YepYoga"]
    try:
        df["datum"] = pd.to_datetime(df["datum"], format="%d-%m-%Y")
    except ValueError:
        df["datum"] = pd.to_datetime(df["datum"], format="mixed")

    df["jaar"] = df["datum"].dt.strftime("%Y")
    df["maand"] = df["datum"].dt.strftime("%m")
    df["maand_"] = df["datum"].dt.strftime("%Y-%m")
    return df


@st.cache_data(ttl=3600)
def fetch_rss_as_df(url: str, blog_name: str) -> pd.DataFrame:
    """Fetch a WordPress RSS feed and return rows shaped like the main dataframe."""
    NS = {
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc":      "http://purl.org/dc/elements/1.1/",
    }
    empty = pd.DataFrame(columns=[
        "id", "id_entry_original", "titel", "kopfoto", "artikel",
        "datum", "afbeelding", "link", "blog", "categorie",
        "jaar", "maand", "maand_",
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

        # Featured image: try multiple sources in order of preference
        image_url = "_"
        media_content = item.find("{http://search.yahoo.com/mrss/}content")
        if media_content is not None:
            image_url = media_content.get("url", "_")
        if image_url == "_":
            enclosure = item.find("enclosure")
            if enclosure is not None:
                image_url = enclosure.get("url", "_")
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
            "id":                f"rss_{i}",
            "id_entry_original": "_",
            "titel":             title,
            "kopfoto":           "_",
            "artikel":           content,
            "datum":             datum,
            "afbeelding":        image_url,
            "link":              link,
            "blog":              blog_name,
            "categorie":         ", ".join(c.text.strip() for c in item.findall("category") if c.text) or "_",
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


def main():
    st.title(":material/article: Blog viewer")

    df = read()
    df = df.fillna("_")
    df["artikel"] = clean_artikel(df["artikel"])

    feeds = [
        ["https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/crazywaiter.xml", "crazywaiter"],
        ["https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/yepyoga.xml", "yepyoga"],
        ["https://rene-smit.com/feed/?posts_per_page=100", "rene-smit.com"],
    ]
    for url, name in feeds:
        rss_df = fetch_rss_as_df(url, blog_name=name)
        if not rss_df.empty:
            rss_df["artikel"] = clean_artikel(rss_df["artikel"])
            df = pd.concat([df, rss_df], ignore_index=True)

    # Sort all entries newest-first
    df = df.sort_values("datum", ascending=False, na_position="last").reset_index(drop=True)

    # Split comma-separated categories into a list for accurate filtering
    df["categorie_list"] = df["categorie"].apply(
        lambda x: [c.strip() for c in x.split(",") if c.strip() and c.strip() != "_"] if x != "_" else []
    )

    # --- Sidebar: blog filter ---
    options = sorted(df["blog"].unique().tolist())
    selected_blogs = st.sidebar.multiselect("Select blog", options, options)

    if not selected_blogs:
        st.error("Select one or more blogs.", icon=":material/error:")
        st.stop()

    df = df[df["blog"].isin(selected_blogs)]

    # --- Sidebar: category filter ---
    if len(selected_blogs) == 1 and selected_blogs[0] in ("CrazyWaiter", "YepYoga", "rene-smit.com"):
        all_categories = sorted({c for cats in df["categorie_list"] for c in cats})
        selected_categories = st.sidebar.multiselect("Select categories", all_categories, all_categories)
        if selected_categories:
            df = df[df["categorie_list"].apply(lambda cats: any(c in selected_categories for c in cats))]
        if df.empty:
            st.error("Choose a category.", icon=":material/filter_list:")
            st.stop()

    # --- Sidebar: search ---
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

    # --- Sidebar: pagination ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"#### Pagination :gray-badge[{total}]")
    posts_per_page = st.sidebar.selectbox(
        "Posts per page", [5, 10, 25, 50], index=1
    )
    n_pages = max(1, -(-total // posts_per_page))  # ceiling division
    page = st.sidebar.number_input(
        "Page", min_value=1, max_value=n_pages, value=1, step=1
    )
    start = (page - 1) * posts_per_page
    end   = start + posts_per_page
    df_page = df.iloc[start:end]

    st.caption(f"Page {page} of {n_pages} · {total} article(s) total")

    # --- Article cards ---
    for row in df_page.itertuples(index=False):
        with st.container(border=True):
            st.subheader(row.titel)
            meta = f':material/calendar_today: {row.datum.strftime("%d %B %Y")}'
            if row.categorie != "_":
                meta += f' &nbsp;·&nbsp; :material/label: {row.categorie}'
            meta +=f'&nbsp;·&nbsp; :material/article: {row.blog}'
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


if __name__ == "__main__":
    main()