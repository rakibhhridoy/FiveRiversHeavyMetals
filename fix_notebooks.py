#!/usr/bin/env python3
"""
Fix corrupted notebooks by restoring newlines in cells
"""

import json
from pathlib import Path
import re

def fix_notebook(nb_path):
    """Fix notebook by ensuring all lines end with newlines"""

    print(f"Fixing: {nb_path.name}")

    with open(nb_path) as f:
        nb = json.load(f)

    fixed_count = 0

    # Process each cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            # Ensure source is a list of strings
            source = cell.get('source', [])

            if isinstance(source, str):
                # Convert string to list
                cell['source'] = source.split('\n')
                fixed_count += 1
            elif isinstance(source, list):
                # Fix each line that doesn't end with newline
                fixed_lines = []
                for line in source:
                    if not line.endswith('\n') and line:  # Add newline if missing
                        fixed_lines.append(line + '\n')
                    else:
                        fixed_lines.append(line)

                if fixed_lines != source:
                    cell['source'] = fixed_lines
                    fixed_count += 1

    # Save fixed notebook
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)  # Compact output

    print(f"  ✓ Fixed {fixed_count} cells")
    return True


if __name__ == "__main__":
    # Fix all notebooks in SedimentRainyAE and SedimentWinterAE
    for season_folder in ['SedimentRainyAE', 'SedimentWinterAE']:
        folder_path = Path(f"/Users/rakibhhridoy/Five_Rivers/gis/{season_folder}")

        if not folder_path.exists():
            print(f"Skipping {season_folder} (not found)")
            continue

        print(f"\nProcessing {season_folder}...")
        notebooks = list(folder_path.glob("*.ipynb"))

        for nb_path in notebooks:
            if nb_path.name.startswith('.'):  # Skip temp files
                continue
            try:
                fix_notebook(nb_path)
            except Exception as e:
                print(f"  ✗ Error: {e}")

    print("\n✓ All notebooks fixed!")
