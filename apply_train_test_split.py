#!/usr/bin/env python3
"""
Apply proper train_test_split to all notebooks
"""

import json
import re
from pathlib import Path

def apply_train_test_split(nb_path):
    """Apply train_test_split to notebooks"""

    with open(nb_path) as f:
        nb = json.load(f)

    fixed = False

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source_list = cell.get('source', [])
            source_str = ''.join(source_list) if isinstance(source_list, list) else source_list

            #Check if it still has KFold
            if 'KFold' in source_str and 'fold_r2_scores = []' in source_str:
                # Replace the entire KFold section with train_test_split
                # Find where KFold is used
                pattern = r'(n_splits = 5\s*\n\s*k_folds = KFold[^\n]*\n[^\n]*random_state=42\))'
                replacement = '''n_splits = 5
train_index, val_index = train_test_split(
    np.arange(len(y_all)), test_size=0.2, random_state=42
)'''
                new_source = re.sub(pattern, replacement, source_str)

                # Fix the fold loop
                if 'for fold, (train_index, val_index) in enumerate(k_folds.split' in new_source:
                    # Replace entire loop with simple if True
                    new_source = new_source.replace(
                        'for fold, (train_index, val_index) in enumerate(k_folds.split(combined_data)):',
                        'if True:  # Single train/test split'
                    )
                    # Update print statements
                    new_source = new_source.replace(
                        'print(f"\\n--- Fold {fold+1}/{n_splits} ---")',
                        'print(f"\\n--- Single Train/Test Split ---")'
                    )
                    new_source = new_source.replace(
                        f'print(f"Fold {{fold+1}} R²: {{r2_fold:.4f}} | RMSE: {{rmse_fold:.4f}}")',
                        'print(f"Test R²: {r2_fold:.4f} | RMSE: {rmse_fold:.4f}")'
                    )

                    # Update results storage
                    new_source = new_source.replace(
                        'fold_r2_scores.append(r2_fold)',
                        'fold_r2_scores = [r2_fold]'
                    )
                    new_source = new_source.replace(
                        'fold_rmse_scores.append(rmse_fold)',
                        'fold_rmse_scores = [rmse_fold]'
                    )

                    # Fix model save path - remove fold number
                    new_source = re.sub(
                        r"f'([^']*)\{fold\+1\}([^']*)'",
                        r"'\1_1\2'",
                        new_source
                    )

                    fixed = True

            if new_source != source_str if 'new_source' in locals() else False:
                if isinstance(source_list, list):
                    cell['source'] = new_source.split('\n')
                    cell['source'] = [line + '\n' if not line.endswith('\n') and line else line for line in cell['source']]
                else:
                    cell['source'] = new_source
                print(f"  ✓ Fixed {nb_path.name}")
                break

    if fixed:
        with open(nb_path, 'w') as f:
            json.dump(nb, f, indent=1)

# Apply to all rainy season top 5
top5_models = [
    'GNN MLP AE.ipynb',
    'CNN GNN MLP PG.ipynb',
    'GNN MLP.ipynb',
    'Stacked CNN GNN MLP.ipynb'
]

for model in top5_models:
    nb_path = Path(f"/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE/{model}")
    apply_train_test_split(nb_path)

print("✓ Done!")
