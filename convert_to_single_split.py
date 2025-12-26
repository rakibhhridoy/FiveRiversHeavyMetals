#!/usr/bin/env python3
"""
Convert all notebooks from K-Fold to single train/test split (80/20)
"""

import json
import re
from pathlib import Path

def convert_notebook_to_single_split(nb_path):
    """Convert notebook from k-fold to single split"""

    print(f"Converting: {nb_path.name}")

    with open(nb_path) as f:
        nb = json.load(f)

    converted = False

    # Process each cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            source_str = ''.join(source) if isinstance(source, list) else source

            # Check if this cell has KFold code
            if 'KFold' in source_str or 'n_splits' in source_str or 'for fold' in source_str:

                # Replace KFold import with train_test_split
                new_source = source_str.replace(
                    'from sklearn.model_selection import KFold',
                    'from sklearn.model_selection import train_test_split'
                )

                # Replace n_splits = 5 with train/test split setup
                new_source = re.sub(
                    r'n_splits = 5\s+k_folds = KFold\([^)]*\)',
                    '# Single train/test split (80/20)',
                    new_source
                )

                # Replace the fold loop with single split
                # Match the for loop structure
                pattern = r'for fold, \(train_index, val_index\) in enumerate\(k_folds\.split\(combined_data\)\):'
                replacement = 'if True:  # Single train/test split'
                new_source = re.sub(pattern, replacement, new_source)

                # Update fold counter display
                new_source = new_source.replace(
                    'print(f"\\n--- Fold {fold+1}/{n_splits} ---")',
                    'print(f"\\n--- Training and Validation Split ---")'
                )

                # Update results summary
                new_source = new_source.replace(
                    f'print(f"Overall Single Split Results ({{n_splits}} Folds)")',
                    'print(f"Training and Validation Results")'
                )
                new_source = new_source.replace(
                    'f"Fold {fold+1} R²:',
                    'f"Test R²:'
                )

                # Store results in list instead of append
                new_source = new_source.replace(
                    'fold_r2_scores.append(r2_fold)',
                    'fold_r2_scores = [r2_fold]'
                )
                new_source = new_source.replace(
                    'fold_rmse_scores.append(rmse_fold)',
                    'fold_rmse_scores = [rmse_fold]'
                )

                # Update the averaging
                new_source = new_source.replace(
                    'print(f"Average R²: {np.mean(fold_r2_scores):.4f} +/- {np.std(fold_r2_scores):.4f}")',
                    'print(f"Test R²: {fold_r2_scores[0]:.4f}")'
                )
                new_source = new_source.replace(
                    'print(f"Average RMSE: {np.mean(fold_rmse_scores):.4f} +/- {np.std(fold_rmse_scores):.4f}")',
                    'print(f"Test RMSE: {fold_rmse_scores[0]:.4f}")'
                )

                if new_source != source_str:
                    # Convert back to list format
                    if isinstance(source, list):
                        cell['source'] = new_source.split('\n')
                        # Add newlines back
                        cell['source'] = [line + '\n' if not line.endswith('\n') and line else line for line in cell['source']]
                    else:
                        cell['source'] = new_source
                    converted = True
                    print(f"  ✓ Converted cell with KFold/train loop")

    if converted:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✓ Saved converted notebook")
        return True
    else:
        print(f"  - No changes needed")
        return False


if __name__ == "__main__":
    # Convert all notebooks in SedimentRainyAE and SedimentWinterAE
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
            if 'AlphaEarth_Data' in nb_path.name:  # Skip data prep notebook
                continue
            try:
                convert_notebook_to_single_split(nb_path)
            except Exception as e:
                print(f"  ✗ Error: {e}")

    print("\n✓ Conversion complete!")
