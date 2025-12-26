#!/usr/bin/env python3
"""
Feature Selection for AlphaEarth Data
Reduces 89 features to top 20 most predictive features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

def select_best_features(csv_file, season_name, n_features=20):
    """Select best features using F-statistic"""

    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION: {season_name}")
    print(f"{'='*80}")

    # Load data (handle encoding issues)
    print(f"\nLoading: {csv_file}")
    try:
        data = pd.read_csv(csv_file, encoding='utf-8')
    except:
        data = pd.read_csv(csv_file, encoding='latin-1')
    drop_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry']
    feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]

    # Clean data: remove any non-numeric characters
    X = data[feature_cols].copy()
    for col in X.columns:
        if isinstance(X[col].iloc[0], str):
            X[col] = pd.to_numeric(X[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    X = X.values.astype(float)

    y = data['RI'].copy()
    if isinstance(y.iloc[0], str):
        y = pd.to_numeric(y.astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    y = y.values.astype(float)

    print(f"✓ Loaded {len(data)} samples with {len(feature_cols)} features")

    # Feature selection
    print(f"\nRunning SelectKBest (k={n_features})...")
    selector = SelectKBest(f_regression, k=n_features)
    X_selected = selector.fit_transform(X, y)

    # Results
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'Feature': feature_cols,
        'F-Score': scores
    }).sort_values('F-Score', ascending=False)

    selected_indices = selector.get_support(indices=True)
    selected_features = np.array(feature_cols)[selected_indices]

    # Categorize selected features
    original_selected = [f for f in selected_features if f in ['hydro_dist_brick', 'num_brick_field', 'hydro_dist_ind', 'num_industry',
                                                               'CrR', 'NiR', 'CuR', 'AsR', 'CdR', 'PbR', 'MR', 'SandR', 'SiltR', 'ClayR', 'FeR']]
    ae_selected = [f for f in selected_features if f.startswith('AE_')]

    print(f"\n✓ Selection Complete")
    print(f"  Original environmental features selected: {len(original_selected)}/15")
    print(f"  AlphaEarth bands selected: {len(ae_selected)}/64")
    print(f"  Total selected: {len(selected_features)}/89")

    print(f"\nTop 20 Features (by F-Score):")
    print(feature_scores.head(20).to_string(index=False))

    print(f"\nSelected Features (by category):")
    print(f"\n  Original Environmental Features ({len(original_selected)}):")
    for feat in sorted(original_selected):
        score = feature_scores[feature_scores['Feature'] == feat]['F-Score'].values[0]
        print(f"    {feat}: {score:.2f}")

    print(f"\n  AlphaEarth Bands ({len(ae_selected)}):")
    for feat in sorted(ae_selected):
        score = feature_scores[feature_scores['Feature'] == feat]['F-Score'].values[0]
        print(f"    {feat}: {score:.2f}")

    # Save
    output_file = csv_file.replace('.csv', '_SELECTED_FEATURES.txt')
    selected_features_list = list(selected_features)
    with open(output_file, 'w') as f:
        f.write('\n'.join(selected_features_list))
    print(f"\n✓ Selected features saved to: {output_file}")

    # Create combined data with selected features
    csv_output = csv_file.replace('.csv', '_SELECTED.csv')
    keep_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry', 'RI'] + list(selected_features)
    data[keep_cols].to_csv(csv_output, index=False)
    print(f"✓ Selected feature CSV saved to: {csv_output}")

    # Plot
    feature_scores_top = feature_scores.head(20).copy()
    feature_scores_top['Category'] = feature_scores_top['Feature'].apply(
        lambda x: 'AlphaEarth' if x.startswith('AE_') else 'Original'
    )

    plt.figure(figsize=(12, 8))
    colors = ['#FF6B6B' if cat == 'AlphaEarth' else '#4ECDC4' for cat in feature_scores_top['Category']]
    plt.barh(range(len(feature_scores_top)), feature_scores_top['F-Score'].values, color=colors)
    plt.yticks(range(len(feature_scores_top)), feature_scores_top['Feature'].values, fontsize=10)
    plt.xlabel('F-Score', fontsize=12)
    plt.title(f'Top 20 Features - {season_name} (AlphaEarth Dataset)', fontsize=14, fontweight='bold')
    plt.legend(['AlphaEarth', 'Original'], loc='lower right')
    plt.tight_layout()

    plot_output = csv_file.replace('.csv', '_FEATURE_SELECTION.png')
    plt.savefig(plot_output, dpi=150)
    print(f"✓ Visualization saved to: {plot_output}")
    plt.close()

    return selected_features_list, feature_scores

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ALPHAEARTH FEATURE SELECTION")
    print("="*80)

    # Process Rainy Season
    rainy_csv = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE/Option_B_RainyAE.csv'
    rainy_features, rainy_scores = select_best_features(rainy_csv, 'RAINY SEASON', n_features=20)

    # Process Winter Season
    winter_csv = '/Users/rakibhhridoy/Five_Rivers/gis/SedimentWinterAE/Option_B_WinterAE.csv'
    winter_features, winter_scores = select_best_features(winter_csv, 'WINTER SEASON', n_features=20)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n✓ Feature selection complete for both seasons")
    print(f"✓ Selected feature lists saved")
    print(f"✓ Visualizations created")
    print(f"\nNext step: Update run_top5_models_simple.py to use selected features")
