
import json
import io
import re
from collections import defaultdict
from urllib.parse import quote
from html import unescape
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
    placeholders = {}
    n = 1  # counter for non-skipped items

    for s in texts:
        t = s.strip()
        if not t:
            continue

        # Skip CSS selectors or hex colors
        if t.startswith("#"):
            placeholders[f"{t}"] = t
        else:
            placeholders[f"text_{n:03}"] = t
            n += 1


   
    # Replace text in SVG with placeholders {text_i}
    new_svg_content = svg_content
    # for i, (key, value) in enumerate(placeholders.items(), start=1):
    #     new_svg_content = new_svg_content.replace(value, f"{{{key}}}")


        # Replace text in SVG with placeholders {key}
    # 1) sort by length (desc) to avoid prefix collisions
    items = sorted(placeholders.items(), key=lambda kv: len(kv[1]), reverse=True)

    for key, value in items:
        # 2) replace only when value is the *whole* text node content
        pattern = r'>' + re.escape(value) + r'<'
        replacement = f'>{{{key}}}<'
        new_svg_content = re.sub(pattern, replacement, new_svg_content)


        #st.write(f"Replaced '{value}' with '{{{key}}}'")
    new_svg_content = new_svg_content.replace("$", "{CURRENCY_SYMBOL}")  # Example replacement
    new_svg_content = new_svg_content.replace("PRICE", "{PRICE}")  #
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

def merge_svg_(svg_text: str, mapping: dict) -> tuple[str, set, set]:

    # Matches {anything_without_spaces_or_braces}
    PLACEHOLDER_RE = re.compile(r"\{([^{}\s]+)\}")
    #PLACEHOLDER_RE = re.compile(r"\{([#A-Za-z0-9_:-]+)\}")
    # If your SVG has &#123;key&#125; this makes them real braces
    svg_text = unescape(svg_text)

    # Normalize mapping keys and values
    mapping = {str(k).strip(): str(v) for k, v in mapping.items()}

    svg_keys = set(PLACEHOLDER_RE.findall(svg_text))
    map_keys = set(mapping.keys())
    missing = svg_keys - map_keys
    unused = map_keys - svg_keys

    def repl(m: re.Match) -> str:
        k = m.group(1)
        return mapping.get(k, m.group(0))  # keep as-is if not mapped

    merged = PLACEHOLDER_RE.sub(repl, svg_text)
    return merged, missing, unused

def merge_svg(svg_text: str, mapping: dict) -> tuple[str, set, set]:
    #svg_keys = set(re.findall(r"\{([A-Za-z0-9_:-]+)\}", svg_text))
    svg_keys = set(re.findall(r"\{([#A-Za-z0-9_:-]+)\}", svg_text))

    map_keys = set(mapping.keys())
    missing = svg_keys - map_keys
    unused = map_keys - svg_keys

    def repl(m):
        k = m.group(1)
        return str(mapping.get(k, m.group(0)))
    merged = re.sub(r"\{([#A-Za-z0-9_:-]+)\}", repl, svg_text)
    return merged, missing, unused


def data_uri_svg(svg: str) -> str:
    return "data:image/svg+xml;utf8," + quote(svg)

def main():
    
    # -------------------- CONFIG --------------------
    SVG_URL  = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/main/input/Seminopoly_placeholders.svg"
    PER_ROW = 5

    # -------------------- LOAD ----------------------
    # try:
    #     data = fetch_json(JSON_URL)  # {placeholder: text}
    # except Exception as e:
    #     st.error(f"Could not read JSON. {e}")
    #     st.stop()

    # if not isinstance(data, dict):
    #     st.error("JSON must be an object like {placeholder: text}")
    #     st.stop()

    # try:
    #     svg_template = fetch_text(SVG_URL)  # SVG with {placeholders}
    # except Exception as e:
    #     st.error(f"Could not read SVG. {e}")
    #     st.stop()

    # new_svg_content, data = prepare_monopoly(SVG_URL)

    # -------------------- LOAD ----------------------
    # Build SVG-derived defaults
    new_svg_content, auto_map = prepare_monopoly(SVG_URL)

    st.subheader("Load a previous mapping")
    uploaded_json = st.file_uploader("Upload JSON mapping file", type=["json"])

    data = None
    if uploaded_json is not None:
        try:
            # Read the uploaded JSON as mapping {placeholder: value}
            data = json.load(uploaded_json)
            if not isinstance(data, dict):
                st.error("Uploaded JSON must be an object like {placeholder: text}")
                st.stop()
            st.success("Uploaded JSON loaded")
        except Exception as e:
            st.error(f"Could not parse uploaded JSON. {e}")
            st.stop()
    else:
        # Fallback to the mapping auto-extracted from the SVG
        data = auto_map
        st.info("No JSON uploaded. Using mapping auto-extracted from SVG.")
    # -------------------- GROUP VALUES ----------------
    rev = defaultdict(list)
    for k, v in data.items():
        v_str = "" if v is None else str(v)
        rev[v_str].append(k)

    unique_values = sorted(rev.keys(), key=lambda x: (x.lower(), -len(rev[x])))

    st.subheader("Edit unique values")
    st.caption("One input per unique value. All placeholders that share it update together.")

    # Global fields
    col_g1, col_g2,col3 = st.columns(3)
    with col_g1:
        currency_symbol = st.text_input("Currency symbol", "€")
    with col_g2:
        price_text = st.text_input("Price text", "PRIJS")
    with col3:
        only_streets = st.checkbox("Only streets", True)

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
                    if only_streets == False:
                        edited_values[val] = st.text_input(label, value=val, help=help_txt, key=f"val::{i}")
                    else:
                        # Keep value as is, but show small context
                        # st.caption(label)
                        edited_values[val] = val

    st.divider()

    # -------------------- GENERATE UPDATED JSON ----------------
    #left, mid, right = st.columns([1,1,2])
    buf = None
    if 1==1:
        if st.button("Generate JSON + SVG"):
            new_map = {}
            for k, v in data.items():
                v_str = "" if v is None else str(v)
                new_map[k] = edited_values.get(v_str, v_str)

            st.session_state["new_map"] = new_map
            # st.success("Updated JSON ready")
            

            buf = io.BytesIO(json.dumps(new_map, ensure_ascii=False, indent=2).encode("utf-8"))
            st.session_state["json_buf"] = buf
            

   
            st.info(
                f"Unique value groups: {len(unique_values)}\n"
                f"Total placeholders: {len(data)}"
            )
            

            # Allow fallback to original map if user did not click Generate yet
            use_generated = True # st.checkbox("Use updated JSON from this session", value=True)

        
            active_map = None
            if use_generated and "new_map" in st.session_state:
                #st.write(st.session_state["new_map"])
                # st.info("New map used")
                active_map = st.session_state["new_map"]
            else:
                # build a pass-through map from current JSON
                active_map = data
                # st.info("No new input")
            
            merged_svg, missing_keys, unused_keys = merge_svg(new_svg_content, active_map)
            merged_svg = merged_svg.replace("{CURRENCY_SYMBOL}", currency_symbol)
            merged_svg = merged_svg.replace("{PRICE}", price_text)
            merged_svg = merged_svg.replace("$", currency_symbol)
            merged_svg = merged_svg.replace("PRICE", price_text)
            merged_svg = merged_svg.replace("XXX", "")
            
            @st.fragment
            def fragment_function_download_json():
                st.download_button(
                    "Download updated JSON to reuse later",
                    data=st.session_state["json_buf"],
                    file_name="placeholders_updated.json",
                    mime="application/json",
                )

            fragment_function_download_json()
            
            # Downloads
            #if "json_buf" in st.session_state:
            @st.fragment
            def fragment_function_download_svg():
                svg_buf = io.BytesIO(merged_svg.encode("utf-8"))
                st.download_button(
                    "Download merged SVG",
                    data=svg_buf,
                    file_name="merged.svg",
                    mime="image/svg+xml",
                )
            fragment_function_download_svg()
            if 1==2:
                if missing_keys:
                    st.warning(f"Missing placeholders in JSON: {sorted(missing_keys)}")
                if unused_keys:
                    st.info(f"JSON keys not used in template: {sorted(list(unused_keys))}")
                    for l in sorted(unused_keys):
                        st.write(f"- `{l}` : `{data[l]}`")

                
                st.json(new_map)
            st.success("Merged SVG created")
            
            # Inline preview
            st.markdown(f'<img src="{data_uri_svg(merged_svg)}">', unsafe_allow_html=True)
            st.code(merged_svg[:2000] + ("\n...\n" if len(merged_svg) > 2000 else ""), language="xml")


    st.info("Board design by jeffgeerling.com, 2007.  CC BY-SA 3.0 US")
    st.info(f"SVG Template - {SVG_URL}")
if __name__ == "__main__":
    main()  
