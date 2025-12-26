#!/usr/bin/env python3
"""
Run a single model with AlphaEarth data
"""

import json
import sys
import subprocess
from pathlib import Path
import shutil

def run_model(season, model_name, option="B"):
    """Run a single model notebook"""

    print(f"\n{'='*70}")
    print(f"Running {season.upper()} - {model_name} (Option {option})")
    print(f"{'='*70}\n")

    # Determine season folder
    if season == "rainy":
        nb_dir = Path("/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE")
    else:
        nb_dir = Path("/Users/rakibhhridoy/Five_Rivers/gis/SedimentWinterAE")

    nb_path = nb_dir / model_name

    if not nb_path.exists():
        print(f"✗ Notebook not found: {nb_path}")
        return False

    # Load and modify notebook
    with open(nb_path) as f:
        nb = json.load(f)

    # Find and modify ALPHA_EARTH_OPTION cell
    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source_str = ''.join(cell['source'])
            if 'ALPHA_EARTH_OPTION' in source_str:
                new_source = source_str.replace(
                    "ALPHA_EARTH_OPTION = 'B'",
                    f"ALPHA_EARTH_OPTION = '{option}'"
                )
                cell['source'] = new_source.split('\n')
                modified = True
                break

    if not modified:
        print(f"✗ Could not find ALPHA_EARTH_OPTION in notebook")
        return False

    # Save temporary notebook
    temp_nb_path = nb_dir / f".temp_{model_name.replace('.ipynb', '')}_opt{option}.ipynb"
    with open(temp_nb_path, 'w') as f:
        json.dump(nb, f, indent=2)

    # Execute using nbconvert
    cmd = [
        "python3", "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=14400",
        str(temp_nb_path)
    ]

    print(f"Executing: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=14400)
        success = result.returncode == 0

        if success:
            print(f"\n✓ Model completed successfully")
        else:
            print(f"\n✗ Model execution failed with code {result.returncode}")

        # Cleanup
        if temp_nb_path.exists():
            temp_nb_path.unlink()

        return success

    except subprocess.TimeoutExpired:
        print(f"✗ Model execution timed out (4 hours)")
        if temp_nb_path.exists():
            temp_nb_path.unlink()
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        if temp_nb_path.exists():
            temp_nb_path.unlink()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 run_single_model.py <season> <model_name> [option]")
        print("Example: python3 run_single_model.py rainy 'Transformer CNN GNN MLP.ipynb' B")
        sys.exit(1)

    season = sys.argv[1]
    model = sys.argv[2]
    option = sys.argv[3] if len(sys.argv) > 3 else "B"

    success = run_model(season, model, option)
    sys.exit(0 if success else 1)
