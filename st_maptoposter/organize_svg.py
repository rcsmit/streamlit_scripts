import xml.etree.ElementTree as ET
from collections import defaultdict
import json
import os
import urllib.request
import streamlit as st


# ORGANIZE SVG ELEMENTS BY COLOR THEME

# Generate a map with maptoposter with the colortheme all_colors.json

# Then use this script to reorganize the SVG elements into groups by their stroke and fill colors
# and name the groups according to the color names in the theme.

# Example usage: python organize_svg.py input.svg output.svg
# theme file is in http://www.github.com/rcsmit/streamlit_scripts/st_maptoposter/themes/all_colors.json

def load_color_theme(theme_path):
    """Load color theme from JSON file or URL."""
    # Check if it's a URL
    if theme_path.startswith('http://') or theme_path.startswith('https://'):
        with urllib.request.urlopen(theme_path) as response:
            theme = json.loads(response.read().decode('utf-8'))
    else:
        # Local file
        with open(theme_path, 'r') as f:
            theme = json.load(f)
    
    # Create a reverse mapping from color hex to name
    color_map = {}
    for key, value in theme.items():
        if key not in ['name', 'description'] and isinstance(value, str) and value.startswith('#'):
            # Normalize to lowercase for comparison
            color_map[value.lower()] = key 
    return color_map

def parse_style(style_str):
    """Parse inline style attribute into a dictionary."""
    if not style_str:
        return {}
    styles = {}
    for item in style_str.split(';'):
        item = item.strip()
        if ':' in item:
            key, value = item.split(':', 1)
            styles[key.strip()] = value.strip()
    return styles

def get_color_key(element):
    """Extract stroke and fill colors from an element."""
    # Check inline style attribute
    style = element.get('style', '')
    style_dict = parse_style(style)
    
    # Get stroke and fill from style or direct attributes
    stroke = style_dict.get('stroke') or element.get('stroke', 'none')
    fill = style_dict.get('fill') or element.get('fill', 'none')
    
    return f"stroke:{stroke}_fill:{fill}"

def get_color_name(color_hex, color_map):
    """Map hex color to descriptive name using the loaded color map."""
    # Normalize color to lowercase
    color_lower = color_hex.lower() if color_hex else 'none'
    return color_map.get(color_lower, color_lower)

def organize_svg_by_color(input_file, output_file, color_map):
    """Organize SVG elements into groups by stroke and fill color."""
    
    # Parse the SVG
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Define namespaces
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink'
    }
    
    # Register namespaces to preserve them
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)
    
    # Find all groups that contain paths or other drawable elements
    figure = root.find('.//{http://www.w3.org/2000/svg}g[@id="figure_1"]')
    if figure is None:
        print("No figure_1 group found!")
        return
    
    axes = figure.find('.//{http://www.w3.org/2000/svg}g[@id="axes_1"]')
    if axes is None:
        print("No axes_1 group found!")
        return
    
    # Collect all elements to reorganize
    color_groups = defaultdict(list)
    elements_to_remove = []
    
    # Process all groups in axes_1
    for group in list(axes):
        if group.tag.endswith('}g') or group.tag == 'g':
            # Process all children in this group
            for element in list(group):
                color_key = get_color_key(element)
                color_groups[color_key].append(element)
                elements_to_remove.append((group, element))
    
    # Remove original elements
    for parent, element in elements_to_remove:
        parent.remove(element)
    
    # Create new organized groups
    for idx, (color_key, elements) in enumerate(sorted(color_groups.items())):
        # Parse the color key to get stroke and fill
        stroke_part, fill_part = color_key.split('_fill:')
        stroke = stroke_part.replace('stroke:', '')
        fill = fill_part
        
        # Get descriptive names
        stroke_name = get_color_name(stroke, color_map)
        fill_name = get_color_name(fill, color_map)
        
        # Create a descriptive group ID
        if fill_name != 'none':
            group_id = f'layer_{fill_name}'
        elif stroke_name != 'none':
            group_id = f'layer_{stroke_name}'
        else:
            group_id = f'layer_{idx}'
        
        # Create a new group for this color combination
        new_group = ET.SubElement(axes, '{http://www.w3.org/2000/svg}g')
        new_group.set('id', group_id)
        
        # Add a comment describing the group
        comment_text = f" {fill_name if fill_name != 'none' else stroke_name} ({len(elements)} elements) "
        comment = ET.Comment(comment_text)
        axes.insert(list(axes).index(new_group), comment)
        
        # Add all elements with this color to the new group
        for element in elements:
            if group_id != "layer_background":  # Skip background layer
                new_group.append(element)
   
    # Write the modified SVG
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    
    # Print summary
    print(f"Organized {len(color_groups)} color groups:")
    for color_key, elements in sorted(color_groups.items()):
        stroke_part, fill_part = color_key.split('_fill:')
        stroke = stroke_part.replace('stroke:', '')
        fill = fill_part
        stroke_name = get_color_name(stroke, color_map)
        fill_name = get_color_name(fill, color_map)
        layer_name = fill_name if fill_name != 'none' else stroke_name
        print(f"  {layer_name}: {len(elements)} elements (stroke={stroke}, fill={fill})")


def main(input_file,output_file):
    # Path to the color theme JSON file (can be local path or URL)
    #theme_path = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\st_maptoposter\themes\all_colors.json"
    
    theme_path = "https://raw.githubusercontent.com/rcsmit/streamlit_scripts/refs/heads/main/st_maptoposter/themes/all_colors.json"
    
    # Or use a local file:
    # theme_path = r"C:\Users\rcxsm\Documents\python_scripts\streamlit_scripts\st_maptoposter\themes\all_colors.json"
    
    # Check if it's a local file and doesn't exist
    if not theme_path.startswith('http') and not os.path.exists(theme_path):
        print(f"Error: Theme file not found at {theme_path}")
        print("Please update the theme_path variable to point to your all_colors.json file")
        exit(1)
    
    # Load color theme
    print(f"Loading color theme from: {theme_path}")
    try:
        color_map = load_color_theme(theme_path)
        print(f"Loaded {len(color_map)} colors from theme\n")
    except Exception as e:
        print(f"Error loading theme: {e}")
        exit(1)

    # Organize the SVG
    organize_svg_by_color(input_file, output_file, color_map)
    print(f"\nOrganized SVG saved to: {output_file}")


def organize_svg_with_theme(input_svg_path, output_svg_path):
    """
    Convenience function to organize SVG using a theme dictionary.
    
    Args:
        input_svg_path: Path to input SVG file
        output_svg_path: Path to save organized SVG file
        theme_dict: Dictionary containing theme colors (same format as all_colors.json)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create color map from theme dictionary
        main(input_file,output_file)
        return output_file
    except Exception as e:
        st.write(f"Error organizing SVG: {e}")
        return None
 
if __name__ == "__main__":

     # File paths
    input_file = r"C:\Users\rcxsm\Downloads\custom_location_all_colors_20260123_085324.svg"  # Change this to your input file
    output_file = r"C:\Users\rcxsm\Downloads\organized_output.svg"  # Change this to your desired output file
    
    main(input_file,output_file)