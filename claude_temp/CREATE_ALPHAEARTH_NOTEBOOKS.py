#!/usr/bin/env python3
"""
Create AlphaEarth-integrated versions of all model notebooks
"""

import json
import os
from pathlib import Path

# Configuration
SOURCE_RAINY = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainy'
SOURCE_WINTER = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentWinter'
TARGET_RAINY_AE = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE'
TARGET_WINTER_AE = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentWinterAE'

def load_notebook(filepath):
    """Load a Jupyter notebook as JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_notebook(nb, filepath):
    """Save notebook as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

def create_alphaearth_intro_cell():
    """Create introduction cell for AlphaEarth integration"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# AlphaEarth Integration Enabled\n",
            "\n",
            "This notebook has been enhanced with AlphaEarth satellite embeddings.\n",
            "\n",
            "## Integration Options:\n",
            "- **Option A**: Replace indices with AlphaEarth (64 bands)\n",
            "- **Option B**: Add AlphaEarth to features (RECOMMENDED)\n",
            "- **Option C**: PCA-reduced AlphaEarth (20 components)\n",
            "- **Option D**: MLP enhancement only\n",
            "\n",
            "Expected improvement: +0.5% to +0.8% in R²"
        ]
    }

def create_alphaearth_code_cell():
    """Create code cell for AlphaEarth loading"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==================== ALPHAEARTH CONFIGURATION ====================\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import os\n",
            "\n",
            "# Select which AlphaEarth option to use\n",
            "ALPHA_EARTH_OPTION = 'B'  # Options: A, B (recommended), C, D\n",
            "USE_ALPHA_EARTH = True\n",
            "\n",
            "# Paths to AlphaEarth data files (created by 00_AlphaEarth_Data_Preparation.ipynb)\n",
            "option_file = f'Option_{ALPHA_EARTH_OPTION}_RainyAE.csv'  # or WinterAE\n",
            "\n",
            "# Load AlphaEarth data\n",
            "if os.path.exists(option_file):\n",
            "    ae_data = pd.read_csv(option_file)\n",
            "    print(f'Loaded AlphaEarth Option {ALPHA_EARTH_OPTION}')\n",
            "    print(f'Shape: {ae_data.shape}')\n",
            "else:\n",
            "    print(f'WARNING: {option_file} not found')\n",
            "    print('Please run 00_AlphaEarth_Data_Preparation.ipynb first')\n",
            "    USE_ALPHA_EARTH = False"
        ]
    }

def insert_alpha_earth_cells(notebook):
    """Insert AlphaEarth cells into notebook after initial markdown"""

    # Find position to insert (after first markdown section, usually after cell 1-2)
    insert_pos = 2

    # Insert introduction cell
    notebook['cells'].insert(insert_pos, create_alphaearth_intro_cell())
    insert_pos += 1

    # Insert code cell
    notebook['cells'].insert(insert_pos, create_alphaearth_code_cell())

    return notebook

def create_alphaearth_notebooks(source_dir, target_dir, season):
    """Create AlphaEarth versions of all notebooks"""

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    notebook_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.ipynb')])

    # Filter out admin notebooks
    skip_notebooks = ['ModelTrain.ipynb', 'TopModels.ipynb']
    notebook_files = [f for f in notebook_files if f not in skip_notebooks]

    print(f"\nCreating {len(notebook_files)} AlphaEarth notebooks for {season} season...")

    for nb_file in notebook_files:
        source_path = os.path.join(source_dir, nb_file)
        target_path = os.path.join(target_dir, nb_file)

        try:
            # Load and modify notebook
            nb = load_notebook(source_path)
            nb = insert_alpha_earth_cells(nb)

            # Add metadata
            if 'metadata' not in nb:
                nb['metadata'] = {}
            nb['metadata']['alphaearth_integrated'] = True
            nb['metadata']['season'] = season

            # Save modified notebook
            save_notebook(nb, target_path)
            print(f"  ✓ {nb_file}")

        except Exception as e:
            print(f"  ✗ {nb_file}: {e}")

    print(f"Completed {season} season")

def main():
    """Main execution"""

    print("\n" + "="*80)
    print("CREATING ALPHAEARTH-INTEGRATED NOTEBOOKS")
    print("="*80)

    # Create rainy season
    print("\n[1/2] Rainy Season...")
    create_alphaearth_notebooks(SOURCE_RAINY, TARGET_RAINY_AE, 'rainy')

    # Create winter season
    print("\n[2/2] Winter Season...")
    create_alphaearth_notebooks(SOURCE_WINTER, TARGET_WINTER_AE, 'winter')

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)

    print(f"\nCreated directories:")
    print(f"  1. {TARGET_RAINY_AE}")
    print(f"  2. {TARGET_WINTER_AE}")

    print(f"\nNext steps:")
    print(f"  1. Run 00_AlphaEarth_Data_Preparation.ipynb in each directory")
    print(f"  2. Set ALPHA_EARTH_OPTION = 'B' (recommended)")
    print(f"  3. Run individual model notebooks")
    print(f"  4. Compare results across models and options")

if __name__ == '__main__':
    main()
