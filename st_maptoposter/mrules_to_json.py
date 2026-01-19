import re
import json
from pathlib import Path
import sys

def parse_color(color_str):
    """
    Parse Maperitive color format to hex.
    Handles formats like:
    - #RRGGBB
    - #RRGGBB black 20% (with modifiers)
    - color_name
    """
    if not color_str:
        return None
    
    # Remove whitespace and convert to lowercase
    color_str = color_str.strip()
    
    # If it starts with #, extract just the hex part
    if color_str.startswith('#'):
        # Split by space to remove modifiers like "black 20%"
        hex_color = color_str.split()[0]
        return hex_color.upper()
    
    # Handle named colors (basic mapping)
    color_map = {
        'white': '#FFFFFF',
        'black': '#000000',
        'red': '#FF0000',
        'blue': '#0000FF',
        'green': '#00FF00',
        'yellow': '#FFFF00',
        'orange': '#FFA500',
        'purple': '#800080',
        'cyan': '#00FFFF',
        'gray': '#808080',
        'grey': '#808080',
        'lightblue': '#ADD8E6',
    }
    
    return color_map.get(color_str.lower(), None)

def find_section_colors(content, target_pattern, color_property='line-color'):
    """
    Find colors within a specific target section using regex.
    Returns the first color found for the specified property.
    """
    # Build pattern to find target section and extract color
    pattern = rf'target\s*:\s*{target_pattern}.*?{color_property}\s*:\s*([#\w\s%]+?)(?:\n|;|$)'
    
    matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        # Return first match, parsed
        return parse_color(matches[0])
    return None

def find_area_colors(content, condition_pattern, color_property='fill-color'):
    """
    Find colors in area definitions (within $featuretype(area) blocks).
    """
    # Find the entire area block
    area_block_pattern = r'target\s*:\s*\$featuretype\(area\)(.*?)(?=target\s*:|$)'
    area_blocks = re.findall(area_block_pattern, content, re.DOTALL)
    
    if not area_blocks:
        return None
    
    area_content = area_blocks[0]
    
    # Look for specific condition (if/elseif blocks)
    condition_pattern_full = rf'(?:if|elseif)\s*:\s*{condition_pattern}.*?{color_property}\s*:\s*([#\w\s%]+?)(?:\n|define|if|elseif|else|draw|for)'
    
    matches = re.findall(condition_pattern_full, area_content, re.DOTALL | re.IGNORECASE)
    
    if matches:
        return parse_color(matches[0])
    return None

def extract_colors_from_mrules(mrules_file, verbose=False):
    """
    Extract colors from Maperitive .mrules file.
    Returns a dictionary with theme colors.
    """
    
    colors = {
        'bg': '#FFFFFF',
        'text': '#000000',
        'gradient_color': '#FFFFFF',
        'water': None,
        'parks': None,
        'road_motorway': None,
        'road_primary': None,
        'road_secondary': None,
        'road_tertiary': None,
        'road_residential': None,
        'road_default': None
    }
    
    # Read the file
    with open(mrules_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if verbose:
        print("\n🔍 Analyzing .mrules file...\n")
    
    # Extract map background color from properties section
    bg_match = re.search(r'map-background-color\s*:\s*([#\w]+)', content)
    if bg_match:
        colors['bg'] = parse_color(bg_match.group(1))
        if verbose:
            print(f"✓ Background: {colors['bg']}")
    
    # Extract gradient color (same as background)
    colors['gradient_color'] = colors['bg']
    
    # Extract water color from area definition
    water_color = find_area_colors(content, r'water', 'fill-color')
    if water_color:
        colors['water'] = water_color
        if verbose:
            print(f"✓ Water (area): {colors['water']}")
    
    # Try water line as fallback
    if not colors['water']:
        water_line_color = find_section_colors(content, r'(?:water\s+line|river|stream|canal)', 'line-color')
        if water_line_color:
            colors['water'] = water_line_color
            if verbose:
                print(f"✓ Water (line): {colors['water']}")
    
    # Extract park/green space color
    park_color = find_area_colors(content, r'park', 'fill-color')
    if park_color:
        colors['parks'] = park_color
        if verbose:
            print(f"✓ Parks: {colors['parks']}")
    
    # If no park, try forest or grass
    if not colors['parks']:
        for alt in ['forest', 'grass', 'nature\s+reserve']:
            alt_color = find_area_colors(content, alt, 'fill-color')
            if alt_color:
                colors['parks'] = alt_color
                if verbose:
                    print(f"✓ Parks (from {alt}): {colors['parks']}")
                break
    
    # Extract road colors - MOTORWAY
    motorway_color = find_section_colors(content, r'(?:highway\s+)?motorway(?!\s+link)', 'line-color')
    if motorway_color:
        colors['road_motorway'] = motorway_color
        if verbose:
            print(f"✓ Motorway: {colors['road_motorway']}")
    
    # PRIMARY
    primary_color = find_section_colors(content, r'(?:highway\s+)?primary(?!\s+link)', 'line-color')
    if primary_color:
        colors['road_primary'] = primary_color
        if verbose:
            print(f"✓ Primary: {colors['road_primary']}")
    
    # If no primary found, look in "major road" section
    if not colors['road_primary']:
        major_color = find_section_colors(content, r'major\s+road', 'line-color')
        if major_color:
            colors['road_primary'] = major_color
            if verbose:
                print(f"✓ Primary (from major road): {colors['road_primary']}")
    
    # SECONDARY
    secondary_color = find_section_colors(content, r'(?:highway\s+)?secondary', 'line-color')
    if secondary_color:
        colors['road_secondary'] = secondary_color
        if verbose:
            print(f"✓ Secondary: {colors['road_secondary']}")
    else:
        # Fallback to primary or major road color
        colors['road_secondary'] = colors['road_primary']
    
    # TERTIARY
    tertiary_color = find_section_colors(content, r'(?:highway\s+)?tertiary', 'line-color')
    if tertiary_color:
        colors['road_tertiary'] = tertiary_color
        if verbose:
            print(f"✓ Tertiary: {colors['road_tertiary']}")
    else:
        # Fallback to primary color
        colors['road_tertiary'] = colors['road_primary']
    
    # RESIDENTIAL (minor roads)
    residential_color = find_section_colors(content, r'(?:highway\s+)?residential', 'line-color')
    if not residential_color:
        residential_color = find_section_colors(content, r'minor\s+road', 'line-color')
    if residential_color:
        colors['road_residential'] = residential_color
        if verbose:
            print(f"✓ Residential: {colors['road_residential']}")
    
    # DEFAULT (path/footway)
    default_color = find_section_colors(content, r'(?:highway\s+)?path', 'line-color')
    if not default_color:
        default_color = find_section_colors(content, r'(?:highway\s+)?footway', 'line-color')
    if default_color:
        colors['road_default'] = default_color
        if verbose:
            print(f"✓ Default/Path: {colors['road_default']}")
    
    # Set fallback colors for any missing values
    fallbacks = {
        'water': '#B5D0D0',
        'parks': '#C0F6B0',
        'road_motorway': "#000000",
        'road_primary': "#000000",
        'road_secondary': "#000000",
        'road_tertiary': "#000000",
        'road_residential':"#000000",
        'road_default':"#000000",
    }
    nr_fallback = 0
    for key, fallback_value in fallbacks.items():
        if not colors[key]:
            colors[key] = fallback_value
            nr_fallback =nr_fallback+ 1
            if verbose:
                print(f"⚠ {key}: using fallback {fallback_value}")
    
    # Text color - try to extract or default to black
    colors['text'] = '#000000'
    
    return colors, nr_fallback

def create_theme_json(mrules_file, output_file=None, theme_name=None, description=None, verbose=False):
    """
    Convert a Maperitive .mrules file to a JSON theme file.
    
    Args:
        mrules_file: Path to the .mrules file
        output_file: Path for the output JSON file (optional)
        theme_name: Name for the theme (optional)
        description: Description for the theme (optional)
        verbose: Print detailed extraction info
    """
    
    mrules_path = Path(mrules_file)
    
    if not mrules_path.exists():
        raise FileNotFoundError(f"File not found: {mrules_file}")
    
    # Extract colors
    colors, nr_fallback = extract_colors_from_mrules(mrules_path, verbose=verbose)
    
    # Extract theme name from filename or use provided
    if not theme_name:
        # Try to extract theme name from comments
        with open(mrules_path, 'r', encoding='utf-8') as f:
            first_lines = ''.join(f.readlines()[:20])
            
        # Look for descriptive comments
        if 'amsterdam' in first_lines.lower():
            theme_name = "Amsterdam Classic"
        elif 'google' in first_lines.lower():
            theme_name = "Google Maps Style"
        elif 'mapnik' in first_lines.lower():
            theme_name = "OSM Mapnik"
        else:
            theme_name = mrules_path.stem.replace('_', ' ').title()
    
    # Create theme object
    theme = {
        "name": theme_name,
        "description": description or f"Theme converted from {mrules_path.name}",
        **colors
    }
    if nr_fallback <3:
        print(f"\n🎉 Successfully extracted colors from {mrules_path.name} with {nr_fallback} fallbacks.")
        # Determine output file
        if not output_file:
            #output_file = mrules_path.parent / f"{mrules_path.stem}_{nr_fallback}_fallbacks.json"
            output_file = mrules_path.parent / f"{mrules_path.stem}.json"
        
        # Write JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(theme, f, indent=2)
        
        print(f"\n✅ Created theme: {output_file}")
        print(f"📋 Theme name: {theme['name']}")
        
        if not verbose:
            print(f"\n💡 Use --verbose flag to see detailed color extraction")
    else:
        print(f"\n❌  Unsuccessfully extracted colors from {mrules_path.name} with {nr_fallback} fallbacks.")
        
    return theme

def batch_convert_mrules(input_dir, output_dir=None, verbose=False):
    """
    Convert all .mrules files in a directory to JSON themes.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    output_path.mkdir(exist_ok=True)
    
    mrules_files = list(input_path.glob("*.mrules"))
    
    if not mrules_files:
        print(f"⚠️ No .mrules files found in {input_dir}")
        return
    
    print(f"Found {len(mrules_files)} .mrules file(s)\n")
    
    for mrules_file in mrules_files:
        output_file = output_path / f"{mrules_file.stem}.json"
        try:
            create_theme_json(mrules_file, output_file, verbose=verbose)
            print()
        except Exception as e:
            print(f"❌ Error converting {mrules_file.name}: {e}\n")

# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Maperitive Rules to JSON Theme Converter")
    print("=" * 60)
    print()
    
    # Parse command line arguments
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    # Remove verbose flag from args
    args = [arg for arg in sys.argv[1:] if arg not in ['--verbose', '-v']]
    
    # Check command line arguments
    if len(args) < 1:
        print("Usage:")
        print("  python mrules_to_json.py <file.mrules> [options]")
        print("  python mrules_to_json.py <file.mrules> <output.json> [options]")
        print("  python mrules_to_json.py --batch <directory> [options]")
        print()
        print("Options:")
        print("  --verbose, -v    Show detailed color extraction")
        print()
        print("Examples:")
        print("  python mrules_to_json.py amsterdam.mrules")
        print("  python mrules_to_json.py amsterdam.mrules --verbose")
        print("  python mrules_to_json.py amsterdam.mrules themes/amsterdam.json")
        print("  python mrules_to_json.py --batch maperitive_rules/")
        print()
        
        # Try to find .mrules files in current directory
        mrules_files = list(Path('.').glob("*.mrules"))
        if mrules_files:
            print(f"Found {len(mrules_files)} .mrules file(s) in current directory:")
            for f in mrules_files:
                print(f"  - {f.name}")
            print()
            response = input("Convert all? (y/n): ").strip().lower()
            if response == 'y':
                verbose_choice = input("Use verbose mode? (y/n): ").strip().lower()
                verbose = verbose_choice == 'y'
                for mrules_file in mrules_files:
                    create_theme_json(mrules_file, verbose=verbose)
                    print()
        else:
            print("No .mrules files found in current directory.")
        
        sys.exit(0)
    
    # Batch mode
    if args[0] == '--batch':
        if len(args) < 2:
            print("❌ Error: Please specify a directory")
            print("Usage: python mrules_to_json.py --batch <directory>")
            sys.exit(1)
        
        batch_convert_mrules(args[1], verbose=verbose)
    
    # Single file mode
    else:
        input_file = args[0]
        output_file = args[1] if len(args) > 1 else None
        
        create_theme_json(input_file, output_file, verbose=verbose)