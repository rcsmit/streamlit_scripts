import re
import json
from pathlib import Path

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
        'gray': '#808080',
        'grey': '#808080',
    }
    
    return color_map.get(color_str.lower(), None)

def extract_colors_from_mrules(mrules_file):
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
    
    # Extract map background color
    bg_match = re.search(r'map-background-color\s*:\s*([#\w]+)', content)
    if bg_match:
        colors['bg'] = parse_color(bg_match.group(1))
    
    # Extract water color
    # Look for water area fill-color
    water_pattern = r'if\s*:\s*water.*?fill-color\s*:\s*([#\w\s%]+)'
    water_match = re.search(water_pattern, content, re.DOTALL)
    if water_match:
        colors['water'] = parse_color(water_match.group(1))
    
    # Also check for water line color as fallback
    if not colors['water']:
        water_line_pattern = r'target\s*:\s*water\s+line.*?line-color\s*:\s*([#\w\s%]+)'
        water_line_match = re.search(water_line_pattern, content, re.DOTALL)
        if water_line_match:
            colors['water'] = parse_color(water_line_match.group(1))
    
    # Extract park/green space color
    park_pattern = r'elseif\s*:\s*park.*?fill-color\s*:\s*([#\w\s%]+)'
    park_match = re.search(park_pattern, content, re.DOTALL)
    if park_match:
        colors['parks'] = parse_color(park_match.group(1))
    
    # Extract road colors
    # Motorway
    motorway_pattern = r'target\s*:\s*motorway\s*\n.*?line-color\s*:\s*([#\w\s%]+)'
    motorway_match = re.search(motorway_pattern, content, re.DOTALL)
    if motorway_match:
        colors['road_motorway'] = parse_color(motorway_match.group(1))
    
    # Major road (primary/secondary/tertiary)
    major_road_pattern = r'target\s*:\s*major\s+road.*?line-color\s*:\s*([#\w\s%]+)'
    major_road_match = re.search(major_road_pattern, content, re.DOTALL)
    if major_road_match:
        major_color = parse_color(major_road_match.group(1))
        # Use same color for primary, secondary, tertiary if not specified
        colors['road_primary'] = major_color
        colors['road_secondary'] = major_color
        colors['road_tertiary'] = major_color
    
    # Minor road (residential)
    minor_road_pattern = r'target\s*:\s*minor\s+road.*?line-color\s*:\s*([#\w\s%]+)'
    minor_road_match = re.search(minor_road_pattern, content, re.DOTALL)
    if minor_road_match:
        colors['road_residential'] = parse_color(minor_road_match.group(1))
    
    # Path as default
    path_pattern = r'target\s*:\s*path.*?line-color\s*:\s*([#\w\s%]+)'
    path_match = re.search(path_pattern, content, re.DOTALL)
    if path_match:
        colors['road_default'] = parse_color(path_match.group(1))
    
    # Set fallback colors if not found
    if not colors['water']:
        colors['water'] = '#8EC1E9'  # Light blue from the file
    if not colors['parks']:
        colors['parks'] = '#8CC98D'  # Light green from the file
    if not colors['road_motorway']:
        colors['road_motorway'] = '#FC9066'  # Orange from the file
    if not colors['road_primary']:
        colors['road_primary'] = '#F9F177'  # Yellow from the file
    if not colors['road_secondary']:
        colors['road_secondary'] = '#F9F177'
    if not colors['road_tertiary']:
        colors['road_tertiary'] = '#F9F177'
    if not colors['road_residential']:
        colors['road_residential'] = '#FEFEFE'  # White from the file
    if not colors['road_default']:
        colors['road_default'] = '#FEFEFE'
    
    # Text color - try to extract or default to black
    colors['text'] = '#000000'
    
    # Gradient color - same as background
    colors['gradient_color'] = colors['bg']
    
    return colors

def create_theme_json(mrules_file, output_file=None, theme_name=None, description=None):
    """
    Convert a Maperitive .mrules file to a JSON theme file.
    
    Args:
        mrules_file: Path to the .mrules file
        output_file: Path for the output JSON file (optional)
        theme_name: Name for the theme (optional)
        description: Description for the theme (optional)
    """
    
    mrules_path = Path(mrules_file)
    
    if not mrules_path.exists():
        raise FileNotFoundError(f"File not found: {mrules_file}")
    
    # Extract colors
    colors = extract_colors_from_mrules(mrules_path)
    
    # Create theme object
    theme = {
        "name": theme_name or mrules_path.stem.replace('_', ' ').title(),
        "description": description or f"Theme converted from {mrules_path.name}",
        **colors
    }
    
    # Determine output file
    if not output_file:
        output_file = mrules_path.parent / f"{mrules_path.stem}.json"
    
    # Write JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(theme, f, indent=2)
    
    print(f"✅ Created theme: {output_file}")
    print(f"📋 Theme name: {theme['name']}")
    print(f"\nColors extracted:")
    for key, value in colors.items():
        if key not in ['name', 'description']:
            print(f"  {key:20s}: {value}")
    
    return theme

def batch_convert_mrules(input_dir, output_dir=None):
    """
    Convert all .mrules files in a directory to JSON themes.
    
    Args:
        input_dir: Directory containing .mrules files
        output_dir: Directory for output JSON files (optional, defaults to input_dir)
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
            create_theme_json(mrules_file, output_file)
            print()
        except Exception as e:
            print(f"❌ Error converting {mrules_file.name}: {e}\n")

# Example usage
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Maperitive Rules to JSON Theme Converter")
    print("=" * 60)
    print()
    
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python mrules_to_json.py <file.mrules>")
        print("  python mrules_to_json.py <file.mrules> <output.json>")
        print("  python mrules_to_json.py --batch <directory>")
        print()
        print("Examples:")
        print("  python mrules_to_json.py amsterdam.mrules")
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
                for mrules_file in mrules_files:
                    create_theme_json(mrules_file)
                    print()
        else:
            print("No .mrules files found in current directory.")
        
        sys.exit(0)
    
    # Batch mode
    if sys.argv[1] == '--batch':
        if len(sys.argv) < 3:
            print("❌ Error: Please specify a directory")
            print("Usage: python mrules_to_json.py --batch <directory>")
            sys.exit(1)
        
        batch_convert_mrules(sys.argv[2])
    
    # Single file mode
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        create_theme_json(input_file, output_file)