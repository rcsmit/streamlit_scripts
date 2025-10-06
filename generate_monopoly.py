import re
import json
import json
import io
import re
from collections import defaultdict
import streamlit as st
import requests

try:
    st.set_page_config(page_title="SVG + JSON Merger", layout="wide")
    
except:
    pass
def prepare_monopoly(json_original, svg_original,  svg_updated):
    """
    Read SVG, find all text elements, replace with placeholders {text_i}.
    Save new SVG and JSON mapping.
    Args:
        json_original: filename to save JSON mapping
        svg_original: filename of original SVG
        svg_updated: filename to save updated SVG with placeholders
    Returns: 
        placeholders dict : {text_i: original_text, ...}
        new_svg_content : str with SVG content with placeholders

    """


    # Load the SVG file
   
    with open(svg_original, "r", encoding="utf-8") as f:
        svg_content = f.read()

    svg_content = svg_content.replace("PRICE","{PRICE}")
    
    svg_content = svg_content.replace("$","{CURRENCY_SYMBOL}")
    svg_content = svg_content.replace("XXX","")
    # Find all text elements in SVG (between > and </text>, or inside <text ...>...</text>)
    texts = re.findall(r'>([^<>]+)<', svg_content)

    # Create placeholder mapping
    placeholders = {f"text_{i+1}": text.strip() for i, text in enumerate(texts) if text.strip()}

    # Replace text in SVG with placeholders {text_i}
    new_svg_content = svg_content
    for i, (key, value) in enumerate(placeholders.items(), start=1):
        new_svg_content = new_svg_content.replace(value, f"{{{key}}}")
   
    
    # Save new SVG with placeholders
   
    with open(svg_updated, "w", encoding="utf-8") as f:
        f.write(new_svg_content)
    
    # Save JSON mapping
    
    with open(json_original, "w", encoding="utf-8") as f:
        json.dump(placeholders, f, indent=4, ensure_ascii=False)

    print("done")
    return placeholders, new_svg_content

def merge_svg(svg_text: str, mapping: dict) -> tuple[str, set, set]:
    """
    Replace placeholders {key} with mapping[key].
    Returns: merged_svg, missing_keys, unused_keys
    """
    # Collect all placeholders present in the SVG template
    svg_keys = set(re.findall(r"\{([A-Za-z0-9_:-]+)\}", svg_text))
    map_keys = set(mapping.keys())

    missing = svg_keys - map_keys
    unused = map_keys - svg_keys

    # Replace using a safe function so we only touch {key} patterns
    def repl(match):
        k = match.group(1)
        return str(mapping.get(k, match.group(0)))  # keep {key} if missing

    merged = re.sub(r"\{([A-Za-z0-9_:-]+)\}", repl, svg_text)
    return merged, missing, unused

def input_with_default(json_original, svg_updated): # placeholders, new_svg_content):
    """ Streamlit app to edit JSON values and merge into SVG.
    Args:
        json_original: filename to load original JSON mapping
       
        svg_updated: filename of SVG template with placeholders
    """

    st.title("SVG Placeholder Editor and Merger")
    #data = placeholders
    # --- Load JSON from filename ---
    try:
        response = requests.get(json_original)
        data = response.json()  # directly parses JSON
    except Exception as e:
        st.error(f"Could not read JSON from filename. {e}")
        st.stop()

    if not isinstance(data, dict):
        st.error("The JSON must be an object of the form {placeholder: text, ...}")
        st.stop()

    # --- Group by identical value: value -> [keys] ---
    rev = defaultdict(list)
    for k, v in data.items():
        v_str = "" if v is None else str(v)
        rev[v_str].append(k)

    st.subheader("Edit unique values")
    st.caption("One input per unique text value. All placeholders sharing that value update together.")

    unique_values = sorted(rev.keys(), key=lambda x: (x.lower(), -len(rev[x])))
    edited_values = {}
    CURRENCY_SYMBOL_CHOSEN = st.text_input("Currency symbol","€")
    PRICE_TXT_CHOSEN = st.text_input("Price text","PRIJS")  
    
     
    # for i, val in enumerate(unique_values, start=1):
    #     keys_sample = ", ".join(rev[val][:5])
    #     more = "" if len(rev[val]) <= 5 else f" … +{len(rev[val]) - 5} more"
    #     label = f"Value {i}: “{val}”"
    #     help_txt = f"Shared by keys: {keys_sample}{more}"
    #     if val[0] == "#":
    #         edited_values[val] = st.text_input(label, value=val, help=help_txt, key=f"val::{i}")
    #     else:
    #         edited_values[val] = val

    edited_values = {}
    per_row = 5

    for start in range(0, len(unique_values), per_row):
        cols = st.columns(per_row)
        row_vals = unique_values[start:start+per_row]

        for j, val in enumerate(row_vals):
            i = start + j + 1  # numbering continues across rows
            keys_sample = ", ".join(rev[val][:5])
            more = "" if len(rev[val]) <= 5 else f" … +{len(rev[val]) - 5} more"
            label = f"Value {i}: “{val}”"
            help_txt = f"Shared by keys: {keys_sample}{more}"

            with cols[j]:
                if val and val[0] == "#":
                    edited_values[val] = st.text_input(
                        label,
                        value=val,
                        help=help_txt,
                        key=f"val::{i}"
                    )
                else:
                    edited_values[val] = val

    
    colA, colB,colC,colD = st.columns([1, 1,1,1])

    # Cache original -> edited mapping to reuse for SVG merge
    new_map_holder = st.session_state.setdefault("new_map_holder", None)

    with colA:
        st.subheader("Generate new JSON")
        if st.button("Generate JSON"):
            new_map = {}
            for k, v in data.items():
                v_str = "" if v is None else str(v)
                new_map[k] = edited_values.get(v_str, v_str)

            st.session_state["new_map_holder"] = new_map
            st.success("Generated updated JSON")
            #st.json(new_map)

            buf = io.BytesIO(json.dumps(new_map, ensure_ascii=False, indent=2).encode("utf-8"))
            st.download_button(
                "Download updated JSON",
                data=buf,
                file_name="placeholders_updated.json",
                mime="application/json",
            )

    
        st.info(
            # f"File: {filename}\n"
            f"Unique value groups: {len(unique_values)}\n"
            f"Total placeholders: {len(data)}"
        )
    with colB:
        st.subheader("Merge into SVG")
        
        # Load SVG template
        try:
            # with open(svg_updated, "r", encoding="utf-8") as f:
            #     svg_template = f.read()
            # url =   # make sure this is the full URL string
            response = requests.get(svg_updated)
            response.raise_for_status()  # will error if not found
            svg_template = response.text
        except Exception as e:
            st.error(f"Could not read SVG template. {e}")
            st.stop()
        # svg_template=new_svg_content
        st.caption(f"Template: {svg_updated}")

        # Choose which JSON to merge
        use_generated = st.checkbox("Use the updated JSON from this session", value=True)
        if not use_generated:
            st.caption("Using the original JSON loaded from file")
    
        merge_btn = st.button("Merge into SVG")

        

        if merge_btn:
            active_map = st.session_state["new_map_holder"] if use_generated and st.session_state["new_map_holder"] else data
            merged_svg, missing_keys, unused_keys = merge_svg(svg_template, active_map)
            merged_svg = merged_svg.replace("{CURRENCY_SYMBOL}",CURRENCY_SYMBOL_CHOSEN)
            merged_svg = merged_svg.replace("{PRICE}",PRICE_TXT_CHOSEN)
            if missing_keys:
                st.warning(f"Missing placeholders in JSON: {sorted(missing_keys)}")
            if unused_keys:
                st.info(f"JSON keys not used in template: {sorted(list(unused_keys))[:10]}{' …' if len(unused_keys) > 10 else ''}")

            # Preview and download
            st.success("Merged SVG created")
            st.code(merged_svg[:2000] + ("\n...\n" if len(merged_svg) > 2000 else ""), language="xml")

            # Try to render inline
            try:
                st.image(merged_svg.encode("utf-8"), caption="SVG Preview", output_format="SVG")
            except Exception:
                st.caption("Preview not available. Download to view.")

            svg_buf = io.BytesIO(merged_svg.encode("utf-8"))
            st.download_button(
                "Download merged SVG",
                data=svg_buf,
                file_name="merged.svg",
                mime="image/svg+xml",
            )
def main():
     # Assume you already set this variable elsewhere in your code
    # json_original = r"C:\Users\rcxsm\Downloads\Seminopoly_placeholders.json"
    # svg_original = r"C:\Users\rcxsm\Downloads\Seminopoly_placeholders.svg"
    # json_updated = r"C:\Users\rcxsm\Downloads\Seminopoly_placeholders.json"
    # svg_updated = r"C:\Users\rcxsm\Downloads\Seminopoly_placeholders_edited.svg"
    json_original ="https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/Seminopoly_placeholders.json"
    svg_updated = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/input/Seminopoly_placeholders.svg"

    #placeholders, new_svg_content = prepare_monopoly(json_original, svg_original, json_updated, svg_updated)
    input_with_default(json_original,  svg_updated) #placeholders, new_svg_content)
if __name__ == "__main__":
    main()
