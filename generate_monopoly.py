
import json
import io
import re
from collections import defaultdict
from urllib.parse import quote

import requests
import streamlit as st

# -------------------- PAGE ----------------------
try:
    st.set_page_config(page_title="SVG + JSON Merger", layout="wide")
except:
    pass

st.title("SVG Placeholder Editor and Merger")

# -------------------- HELPERS -------------------
def prepare_monopoly(svg_original): 
    """ Read SVG, find all text elements, replace with placeholders {text_i}. Save new SVG and JSON mapping. 
        Args: 
            svg_original: filename of original 
        Returns: 
            placeholders dict : {text_i: original_text, ...} 
            new_svg_content : str with SVG content with placeholders """

    # Load the SVG file
    
    # with open(file_path, "r", encoding="utf-8") as f:
    #     svg_content = f.read()
    svg_content = fetch_text(svg_original)
    # Find all text elements in SVG (between > and </text>, or inside <text ...>...</text>)
    texts = re.findall(r'>([^<>]+)<', svg_content)

    # Create placeholder mapping
    placeholders = {f"text_{i+1}": text.strip() for i, text in enumerate(texts) if text.strip()}

    # Replace text in SVG with placeholders {text_i}
    new_svg_content = svg_content
    for i, (key, value) in enumerate(placeholders.items(), start=1):
        new_svg_content = new_svg_content.replace(value, f"{{{key}}}")

    # # Save new SVG with placeholders
    # placeholder_svg_path = "/mnt/data/Seminopoly_placeholders.svg"
    # with open(placeholder_svg_path, "w", encoding="utf-8") as f:
    #     f.write(new_svg_content)

    # # Save JSON mapping
    # json_path = "/mnt/data/Seminopoly_placeholders.json"
    # with open(json_path, "w", encoding="utf-8") as f:
    #     json.dump(placeholders, f, indent=4, ensure_ascii=False)

    return new_svg_content, placeholders

@st.cache_data
def fetch_json(url: str) -> dict:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

@st.cache_data
def fetch_text(url: str) -> str:
    r = requests.get(url, timeout=20)

    r.raise_for_status()
 
    return r.text

def merge_svg(svg_text: str, mapping: dict) -> tuple[str, set, set]:
    svg_keys = set(re.findall(r"\{([A-Za-z0-9_:-]+)\}", svg_text))
    map_keys = set(mapping.keys())
    missing = svg_keys - map_keys
    unused = map_keys - svg_keys

    def repl(m):
        k = m.group(1)
        return str(mapping.get(k, m.group(0)))
    merged = re.sub(r"\{([A-Za-z0-9_:-]+)\}", repl, svg_text)
    return merged, missing, unused

def merge_svg(svg_text: str, mapping: dict) -> tuple[str, set, set]: 
    """ Replace placeholders {key} with mapping[key]. Returns: merged_svg, missing_keys, unused_keys """ 
    # Collect all placeholders present in the SVG 
    #template 
    # 
    svg_keys = set(re.findall(r"\{([A-Za-z0-9_:-]+)\}", svg_text)) 
    map_keys = set(mapping.keys()) 
    missing = svg_keys - map_keys 
    unused = map_keys - svg_keys 
    # Replace using a safe function so we only touch {key} patterns 
    def repl(match): 
        k = match.group(1) 
        return str(mapping.get(k, match.group(0))) # keep {key} if missing 
    merged = re.sub(r"\{([A-Za-z0-9_:-]+)\}", repl, svg_text) 
    return merged, missing, unused

def data_uri_svg(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)

def main():
    
    # -------------------- CONFIG --------------------
    JSON_URL = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/Seminopoly_placeholders.json"
    SVG_URL  = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/Seminopoly_placeholders.svg"
    PER_ROW = 5

    # -------------------- LOAD ----------------------
    try:
        data = fetch_json(JSON_URL)  # {placeholder: text}
    except Exception as e:
        st.error(f"Could not read JSON. {e}")
        st.stop()

    if not isinstance(data, dict):
        st.error("JSON must be an object like {placeholder: text}")
        st.stop()

    # try:
    #     svg_template = fetch_text(SVG_URL)  # SVG with {placeholders}
    # except Exception as e:
    #     st.error(f"Could not read SVG. {e}")
    #     st.stop()
    new_svg_content, data = prepare_monopoly(SVG_URL)
    # -------------------- GROUP VALUES ----------------
    rev = defaultdict(list)
    for k, v in data.items():
        v_str = "" if v is None else str(v)
        rev[v_str].append(k)

    unique_values = sorted(rev.keys(), key=lambda x: (x.lower(), -len(rev[x])))

    st.subheader("Edit unique values")
    st.caption("One input per unique value. All placeholders that share it update together.")

    # Global fields
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        currency_symbol = st.text_input("Currency symbol", "€")
    with col_g2:
        price_text = st.text_input("Price text", "PRIJS")

    edited_values = {}

    for start in range(0, len(unique_values), PER_ROW):
        cols = st.columns(PER_ROW)
        row_vals = unique_values[start:start+PER_ROW]
        for j, val in enumerate(row_vals):
            i = start + j + 1
            keys_sample = ", ".join(rev[val][:5])
            more = "" if len(rev[val]) <= 5 else f" … +{len(rev[val]) - 5} more"
            label = f"Value {i}: “{val}”"
            help_txt = f"Shared by keys: {keys_sample}{more}"
            with cols[j]:
                if val and val[0] == "#":
                    edited_values[val] = st.text_input(label, value=val, help=help_txt, key=f"val::{i}")
                else:
                    # Keep value as is, but show small context
                    # st.caption(label)
                    edited_values[val] = val

    st.divider()

    # -------------------- GENERATE UPDATED JSON ----------------
    left, mid, right = st.columns([1,1,2])

    with left:
        if st.button("Generate JSON"):
            new_map = {}
            for k, v in data.items():
                v_str = "" if v is None else str(v)
                new_map[k] = edited_values.get(v_str, v_str)

            st.session_state["new_map"] = new_map
            st.success("Updated JSON ready")
            st.json(new_map)

            buf = io.BytesIO(json.dumps(new_map, ensure_ascii=False, indent=2).encode("utf-8"))
            st.session_state["json_buf"] = buf

    with mid:
        st.info(
            f"Unique value groups: {len(unique_values)}\n"
            f"Total placeholders: {len(data)}"
        )

    with right:
        # Allow fallback to original map if user did not click Generate yet
        use_generated = st.checkbox("Use updated JSON from this session", value=True)

        if st.button("Merge into SVG"):
            active_map = None
            if use_generated and "new_map" in st.session_state:
                st.write(st.session_state["new_map"])
                active_map = st.session_state["new_map"]
            else:
                # build a pass-through map from current JSON
                active_map = data
                st.info("No new input")

            merged_svg, missing_keys, unused_keys = merge_svg(new_svg_content, active_map)
            merged_svg = merged_svg.replace("{CURRENCY_SYMBOL}", currency_symbol)
            merged_svg = merged_svg.replace("{PRICE}", price_text)

            if missing_keys:
                st.warning(f"Missing placeholders in JSON: {sorted(missing_keys)}")
            if unused_keys:
                st.info(f"JSON keys not used in template: {sorted(list(unused_keys))[:10]}{' …' if len(unused_keys) > 10 else ''}")

            st.success("Merged SVG created")
            st.code(merged_svg[:2000] + ("\n...\n" if len(merged_svg) > 2000 else ""), language="xml")

            # Inline preview
            st.markdown(f'<img src="{data_uri_svg(merged_svg)}" style="max-width:100%;">', unsafe_allow_html=True)

            # Downloads
            if "json_buf" in st.session_state:
                st.download_button(
                    "Download updated JSON",
                    data=st.session_state["json_buf"],
                    file_name="placeholders_updated.json",
                    mime="application/json",
                )

            svg_buf = io.BytesIO(merged_svg.encode("utf-8"))
            st.download_button(
                "Download merged SVG",
                data=svg_buf,
                file_name="merged.svg",
                mime="image/svg+xml",
            )

if __name__ == "__main__":
    main()  
