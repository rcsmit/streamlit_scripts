import streamlit as st
import feedparser
import xml.etree.ElementTree as ET
from datetime import datetime
import requests
from urllib.parse import urlparse
import time


import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
import time
import requests
import feedparser
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

# --------- Config ---------
INPUT_OPML_PATH  = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\feedly2020.opml"
OUTPUT_OPML_PATH = "feed2025.opml"
DAYS_BACK = 365
TIMEOUT_SEC = 12
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; OPMLPruner/1.0)"}
def parse_opml_checkfunction(opml_text: str) -> List[Dict]:
    """Return flat list of feeds: [{category, title, xmlUrl, htmlUrl}] preserving nested category path."""
    root = ET.fromstring(opml_text)
    body = root.find("body")
    feeds: List[Dict] = []

    def walk(node, path: List[str]):
        for o in node.findall("outline"):
            label = (o.get("title") or o.get("text") or "").strip()
            xml = (o.get("xmlUrl") or "").strip()
            html = (o.get("htmlUrl") or "").strip()
            if xml:
                feeds.append({
                    "category": "/".join(path) if path else "Uncategorized",
                    "title": label or xml,
                    "xmlUrl": xml,
                    "htmlUrl": html,
                })
            # Recurse if it has children
            if list(o.findall("outline")):
                walk(o, path + [label or ""])
    if body is not None:
        walk(body, [])
    # Deduplicate on xmlUrl, keep first occurrence/category
    seen = set()
    uniq = []
    for f in feeds:
        if f["xmlUrl"] in seen:
            continue
        seen.add(f["xmlUrl"])
        uniq.append(f)
    return uniq

def latest_entry_dt(feed) -> Optional[datetime]:
    """Extract latest entry datetime in UTC if available."""
    best = None
    for e in feed.entries:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = e.published_parsed
        elif getattr(e, "updated_parsed", None):
            ts = e.updated_parsed
        if ts:
            dt = datetime.fromtimestamp(time.mktime(ts), tz=timezone.utc)
            if best is None or dt > best:
                best = dt
    return best

def feed_has_recent_post(xml_url: str, cutoff: datetime) -> bool:
    """True if feed exists and has any entry newer than cutoff."""
    try:
        resp = requests.get(xml_url, headers=HEADERS, timeout=TIMEOUT_SEC, allow_redirects=True)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)
        if parsed.bozo and not parsed.entries:
            return False
        dt = latest_entry_dt(parsed)
        return bool(dt and dt >= cutoff)
    except Exception:
        return False

def prune_feeds(opml_path: str, out_path: str, days_back: int = DAYS_BACK) -> Tuple[int, int]:
    opml_text = open(opml_path, encoding="utf-8", errors="ignore").read()
    feeds = parse_opml_checkfunction(opml_text)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    # Check each feed
    kept: List[Dict] = []
    for f in feeds:
        print (f)
        if feed_has_recent_post(f["xmlUrl"], cutoff):
        #if feed_has_recent_post(f, cutoff):
            kept.append(f)

    # Rebuild OPML grouped by category
    root = ET.Element("opml", {"version": "1.0"})
    head = ET.SubElement(root, "head")
    title = ET.SubElement(head, "title")
    title.text = "Filtered Subscriptions (active within last year)"
    body = ET.SubElement(root, "body")

    # group by category
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for f in kept:
        groups[f["category"]].append(f)

    # Build a simple flat category structure (no nested rebuild; category path is a single outline)
    for cat in sorted(groups.keys()):
        cat_outline = ET.SubElement(body, "outline", {"text": cat, "title": cat})
        for f in sorted(groups[cat], key=lambda x: x["title"].lower()):
            attrs = {
                "type": "rss",
                "text": f["title"],
                "title": f["title"],
                "xmlUrl": f["xmlUrl"],
            }
            if f["htmlUrl"]:
                attrs["htmlUrl"] = f["htmlUrl"]
            ET.SubElement(cat_outline, "outline", attrs)

    # Write pretty-ish XML
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            for child in elem:
                indent(child, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    indent(root)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    with open(out_path, "wb") as f:
        f.write(xml_bytes)

    return len(feeds), len(kept)

def check_feeds():

    # --------------------------

    total, kept = prune_feeds(INPUT_OPML_PATH, OUTPUT_OPML_PATH, DAYS_BACK)
    print(f"Checked {total} feeds, kept {kept}. Saved to {OUTPUT_OPML_PATH}.")


# Page config
st.set_page_config(
    page_title="RSS Newsreader",
    page_icon="📰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .feed-item {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
    }
    .feed-title {
        font-size: 18px;
        font-weight: bold;
        color: #333;
    }
    .feed-meta {
        color: #666;
        font-size: 14px;
        margin: 5px 0;
    }
    .feed-summary {
        color: #555;
        line-height: 1.6;
    }
    .category-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 10px;
        border-radius: 10px;
        margin: 10px 0 5px 0;
    }
</style>
""", unsafe_allow_html=True)
@st.cache_data(ttl=600)
def parse_opml(opml_content):
    """Parse OPML without getparent. Builds {category: [feeds]} using a recursive walk."""
    if isinstance(opml_content, bytes):
        opml_text = opml_content.decode("utf-8", errors="ignore")
    else:
        opml_text = opml_content

    root = ET.fromstring(opml_text)
    categories = {}

    def walk(node, path):
        for o in node.findall("outline"):
            title = o.get("title") or o.get("text") or "Uncategorized"
            xml = o.get("xmlUrl")
            if xml:
                cat = "/".join(path) if path else "Uncategorized"
                categories.setdefault(cat, []).append({
                    "title": title,
                    "xmlUrl": xml,
                    "htmlUrl": o.get("htmlUrl", "")
                })
            # Recurse into children
            if list(o.findall("outline")):
                walk(o, path + [title])

    body = root.find("body")
    if body is not None:
        walk(body, [])

    return categories

@st.cache_data(ttl=300)
def fetch_feed(url, timeout=10):
    """Fetch and parse RSS feed with cache-safe return."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)

        # Make cache-safe: remove or stringify unpicklable fields
        if "bozo_exception" in parsed:
            parsed["bozo_exception"] = str(parsed["bozo_exception"])

        if parsed.bozo and not parsed.entries:
            return None
        return parsed
    except Exception:
        return None


def format_date(date_struct):
    """Format date from struct_time to readable string"""
    try:
        if date_struct:
            dt = datetime.fromtimestamp(time.mktime(date_struct))
            return dt.strftime('%B %d, %Y at %I:%M %p')
    except:
        pass
    return 'Date not available'

def clean_summary(summary, max_length=300):
    """Clean and truncate summary text"""
    if not summary:
        return "No summary available"
    
    # Remove HTML tags
    import re
    clean = re.compile('<.*?>')
    summary = re.sub(clean, '', summary)
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def main_claude():
    # Main app
    st.title("📰 RSS Newsreader")
    st.markdown("Browse your RSS feeds organized by category")

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Settings")
        opml_file=r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\input\feedly2025.opml"
    
        # Upload OPML file
        # uploaded_file = st.file_uploader("Upload OPML file", type=['opml', 'xml'])
        
        # if uploaded_file is not None:
        #     opml_content = uploaded_file.read()
        #     st.session_state['opml_content'] = opml_content
        #     st.success("OPML file loaded successfully!")
        
        opml_content =   open(opml_file, encoding="utf-8", errors="ignore").read()
        st.session_state['opml_content'] = opml_content
        # Number of items to display per feed
        items_per_feed = st.slider("Items per feed", 1, 1000, 10)
        
        # Filter options
        # st.header("🔍 Filter")
        # search_term = st.text_input("Search in titles")
        
        # Refresh button
        if st.button("🔄 Refresh All Feeds", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # Main content
    if 'opml_content' in st.session_state:
        categories = parse_opml(st.session_state['opml_content'])
        
        # Category selection
        all_categories = list(categories.keys())
        selected_categories = st.multiselect(
            "Select categories to display",
            all_categories,
            default=all_categories [:5] if len(all_categories) > 5 else all_categories
        )
        
        # Display feeds by category
        for category in selected_categories:
            if category in categories:
                st.markdown(f'<div class="category-header"><h2>{category}</h2></div>', 
                        unsafe_allow_html=True)
                
                # Create columns for feeds in this category
                feeds = categories[category]
                
                # Limit number of feeds to display (for performance)
                max_feeds = min(len(feeds), 50)
                
                for feed_info in feeds[:max_feeds]:
                    feed_url = feed_info['xmlUrl']
                        
                    # Fetch feed
                    with st.spinner(f"Loading {feed_info['title']}..."):
                        feed = fetch_feed(feed_url)
                    if feed and feed.entries:
                        
                    # Display feed info
                        # col1, col2 = st.columns([3, 1])
                        if hasattr(feed.feed, 'title'):
                            title = (f"**{feed.feed.title}**")
                    
                        if feed_info['htmlUrl']:
                            url = (f"{feed_info['htmlUrl']}")
                            
                        with st.expander(f"📡 {feed_info['title']} | {url}", expanded=False):
                        
                        
                            # # Display feed info
                            # col1, col2 = st.columns([3, 1])
                            # with col1:
                            #     if hasattr(feed.feed, 'title'):
                            #         st.markdown(f"**{feed.feed.title}**")
                            # with col2:
                            #     if feed_info['htmlUrl']:
                            #         st.markdown(f"[Visit Website]({feed_info['htmlUrl']})")
                            
                            # Display entries
                            for entry in feed.entries[:items_per_feed]:
                                # Filter by search term
                                # if search_term and search_term.lower() not in entry.title.lower():
                                #     continue
                                
                                # st.markdown("---")
                                
                                # Entry title with link
                                title = entry.title if hasattr(entry, 'title') else 'Untitled'
                                link = entry.link if hasattr(entry, 'link') else '#'
                                
                                # Publication date
                                if hasattr(entry, 'published_parsed'):
                                    date_str = format_date(entry.published_parsed)
                                    # st.caption(f"📅 {date_str}")
                                else:
                                    date_str= ""
                                
                                st.markdown(f"[{title}]({link}) | 📅 {date_str}")
                                
                                    # Summary
                                if hasattr(entry, 'summary'):
                                    summary = clean_summary(entry.summary)
                                    if summary !="No summary available":
                                        with st.expander("Summary"):
                                        
                                            st.markdown(f"<p class='feed-summary'>{summary}</p>", 
                                                    unsafe_allow_html=True)
                                            
                                            # Author
                                            if hasattr(entry, 'author'):
                                                st.caption(f"✍️ By {entry.author}")
                    else:
                        print(f"Could not load feed: {feed_info['title']}")
                
                if len(feeds) > max_feeds:
                    st.info(f"Showing {max_feeds} of {len(feeds)} feeds in this category")

    else:
        # Instructions if no OPML file is loaded
        st.info("👈 Please upload your OPML file in the sidebar to get started")
        
        st.markdown("""
        ### How to use this newsreader:
        
        1. **Upload your OPML file** using the file uploader in the sidebar
        2. **Select categories** you want to view
        3. **Browse articles** from your RSS feeds
        4. **Use the search** to filter articles by title
        5. **Adjust settings** like number of items per feed
        
        ### Features:
        - 📱 Responsive design
        - 🔄 Refresh feeds
        - 🔍 Search functionality
        - 📊 Organized by categories
        - ⚡ Cached for performance
        """)
        
        # Sample OPML for testing
        if st.button("Load Sample OPML for Testing"):
            sample_opml = """<?xml version="1.0" encoding="UTF-8"?>
            <opml version="1.0">
                <head><title>Sample Feeds</title></head>
                <body>
                    <outline text="Tech News" title="Tech News">
                        <outline type="rss" text="Hacker News" title="Hacker News" 
                                xmlUrl="https://news.ycombinator.com/rss" 
                                htmlUrl="https://news.ycombinator.com"/>
                        <outline type="rss" text="TechCrunch" title="TechCrunch" 
                                xmlUrl="https://techcrunch.com/feed/" 
                                htmlUrl="https://techcrunch.com"/>
                    </outline>
                </body>
            </opml>"""
            st.session_state['opml_content'] = sample_opml.encode()
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit 🚀 | RSS Newsreader")


if __name__ == '__main__':
    #main_chatgpt()
    main_claude()

    print("----------------------------")
    #check_feeds()
