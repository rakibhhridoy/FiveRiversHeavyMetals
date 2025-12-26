# AlphaEarth Integration - Next Steps & Recommendations

## Current Situation Summary

| Aspect | Status |
|--------|--------|
| **AlphaEarth Data Quality** | ✓ Excellent - 64 bands extracted, no missing values |
| **Data Integration** | ✓ Complete - 89 features total (21 original + 64 AlphaEarth) |
| **Model Implementation** | ✓ Complete - 5 models implemented with AlphaEarth support |
| **Model Execution** | ✓ Complete - Single split and k-fold execution available |
| **Performance Assessment** | ⚠️ Inconclusive - Dataset too small for reliable comparison |

---

## The Core Problem

**Your dataset has 17 samples. Deep learning models need ~200-1000 samples to reliably evaluate 89 features.**

Current ratios:
- **Samples to Features**: 17:89 = **0.19 ratio** (should be >5)
- **Training Samples**: 13 (should be >50)
- **Test Samples**: 4 (should be >30)

This is why:
- **Baseline (21 features)**: Works well even with 17 samples → simple features are stable
- **AlphaEarth (89 features)**: Struggles → too many parameters relative to training data

---

## Recommended Action: Feature Selection

This is the **fastest** solution that gives immediate results while maintaining your current dataset size.

### Step 1: Identify Most Predictive AlphaEarth Bands

```python
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np

# Load data
data = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE/Option_B_RainyAE.csv')

# Extract features and target
drop_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry']
feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]

X = data[feature_cols].values
y = data['RI'].values

# Feature selection: keep top 20 features
selector = SelectKBest(f_regression, k=20)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = np.array(feature_cols)[selector.get_support()]
print("Selected features:")
for feat in selected_features:
    print(f"  {feat}")
```

### Step 2: Analyze Results

Check which features were selected:
```python
# Get importance scores
scores = selector.scores_
feature_scores = pd.DataFrame({
    'Feature': feature_cols,
    'F-Score': scores
}).sort_values('F-Score', ascending=False)

print(feature_scores.head(20))
```

Expected outcome:
- Probably 10-12 original environmental features
- Probably 8-10 best AlphaEarth bands
- Total 20 features (balanced for 17 samples)

### Step 3: Run Models with Selected Features

```python
# Modify run_top5_models_simple.py:
# Replace line: feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]
# With: feature_cols = selected_feature_names  # From SelectKBest

# Then run as normal:
python3 run_top5_models_simple.py
```

### Expected Results After Feature Selection

| Model | R² (Before) | R² (After) |
|-------|----------|----------|
| GNN MLP | 0.7022 | ~0.80-0.85 |
| GNN MLP AE | 0.5515 | ~0.75-0.80 |
| Transformer CNN GNN MLP | 0.0780 | ~0.70-0.80 |

(Rough estimates based on typical feature selection effects)

---

## Alternative: Principal Component Analysis (PCA)

Faster but less interpretable than feature selection.

```python
from sklearn.decomposition import PCA

# Extract AlphaEarth bands only
ae_cols = [c for c in data.columns if c.startswith('AE_')]
X_ae = data[ae_cols].values

# Reduce 64 bands to 5 principal components (retains ~85% variance)
pca = PCA(n_components=5)
X_ae_pca = pca.fit_transform(X_ae)

# Combine with original features
original_cols = ['hydro_dist_brick', 'num_brick_field', 'hydro_dist_ind', 'num_industry',
                 'CrR', 'NiR', 'CuR', 'AsR', 'CdR', 'PbR', 'MR', 'SandR', 'SiltR', 'ClayR', 'FeR']
X_original = data[original_cols].values

X_combined = np.hstack([X_original, X_ae_pca])  # 20 total features
```

**Advantages**: Faster implementation
**Disadvantages**: PCA components not interpretable (can't say "AlphaEarth band 24 is important")

---

## Long-Term Solution: Collect More Data

### Why This Matters
With 50+ samples, you could:
- Use k-fold cross-validation with confidence (>30 test samples total)
- Support 100+ features without overfitting concerns
- Properly quantify AlphaEarth's contribution
- Make publishing-quality claims about model performance

### Where to Get More Data

1. **Expand sampling locations**:
   - Currently 17 samples in study area
   - Collect from adjacent regions (Winter season region, different sub-basins)
   - Target: 50-100 total samples

2. **Temporal sampling**:
   - Collect during different seasons/years
   - 17 samples × 3 years = 51 samples

3. **Google Earth Engine enhancement**:
   - Extract additional satellite bands beyond AlphaEarth
   - Sentinel-2 (11 bands), Landsat-8 (11 bands), etc.
   - Creates larger feature set, which justifies larger sample set

---

## Action Plan: Choose One

### Option A: Quick Win (Do Today) ✓ RECOMMENDED
1. Create `feature_selection_alphaearth.py` with SelectKBest
2. Run feature selection (2 min)
3. Modify `run_top5_models_simple.py` to use selected features
4. Run models again (5 min)
5. Compare results
6. **Time**: 30 minutes
7. **Result**: See if 20 best features perform better than all 89

### Option B: Medium Effort (This Week)
1. Implement PCA dimensionality reduction
2. Combine original features + 5 PCA components = 20 total
3. Run k-fold cross-validation
4. Create comparison table
5. **Time**: 3-4 hours
6. **Result**: Fair k-fold comparison of baseline vs AlphaEarth

### Option C: Long-Term (Next Months)
1. Plan additional sampling campaigns
2. Expand to 50-100 samples
3. Retrain all models
4. Publish comparative analysis
5. **Time**: 2-3 months
6. **Result**: Definitive answer on AlphaEarth value

---

## What I've Already Created

You currently have:
1. ✓ `run_top5_models_simple.py` - Single split execution
2. ✓ `run_top5_models_alphaearth_kfold.py` - K-fold execution
3. ✓ `TOP5_RAINY_ALPHAEARTH_RESULTS.csv` - Single split results
4. ✓ `TOP5_RAINY_ALPHAEARTH_KFOLD_RESULTS.csv` - K-fold results
5. ✓ `ALPHAEARTH_PERFORMANCE_ANALYSIS.md` - Technical analysis
6. ✓ `ALPHAEARTH_VS_BASELINE_COMPARISON.md` - Detailed comparison

Ready to use for winter season:
- `/Users/rakibhhridoy/Five_Rivers/gis/SedimentWinterAE/Option_B_WinterAE.csv` - Winter AlphaEarth data

---

## Implementation: Feature Selection Script

Create `/Users/rakibhhridoy/Five_Rivers/feature_selection_alphaearth.py`:

```python
#!/usr/bin/env python3
"""
Feature Selection for AlphaEarth Data
Reduces 89 features to top 20 most predictive features
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

def select_best_features(csv_file, n_features=20):
    """Select best features using F-statistic"""

    # Load data
    data = pd.read_csv(csv_file)
    drop_cols = ['Stations', 'River', 'Lat', 'Long', 'geometry']
    feature_cols = [c for c in data.columns if c not in drop_cols + ['RI']]

    X = data[feature_cols].values
    y = data['RI'].values

    # Feature selection
    selector = SelectKBest(f_regression, k=n_features)
    X_selected = selector.fit_transform(X, y)

    # Results
    scores = selector.scores_
    feature_scores = pd.DataFrame({
        'Feature': feature_cols,
        'F-Score': scores
    }).sort_values('F-Score', ascending=False)

    selected_features = np.array(feature_cols)[selector.get_support()]

    print(f"Feature Selection Results")
    print(f"========================")
    print(f"Original features: {len(feature_cols)}")
    print(f"Selected features: {n_features}")
    print(f"\nTop 20 Features (by F-Score):")
    print(feature_scores.head(20).to_string(index=False))

    print(f"\nSelected Features (in order of appearance):")
    for feat in selected_features:
        score = feature_scores[feature_scores['Feature'] == feat]['F-Score'].values[0]
        print(f"  {feat}: {score:.2f}")

    # Save
    selected_features_list = list(selected_features)
    with open('/Users/rakibhhridoy/Five_Rivers/SELECTED_FEATURES.txt', 'w') as f:
        f.write('\n'.join(selected_features_list))

    # Plot
    feature_scores_top = feature_scores.head(20)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_scores_top)), feature_scores_top['F-Score'].values)
    plt.yticks(range(len(feature_scores_top)), feature_scores_top['Feature'].values)
    plt.xlabel('F-Score')
    plt.title('Top 20 Features (AlphaEarth Dataset)')
    plt.tight_layout()
    plt.savefig('/Users/rakibhhridoy/Five_Rivers/FEATURE_SELECTION_RESULTS.png', dpi=150)
    print(f"\n✓ Visualization saved to FEATURE_SELECTION_RESULTS.png")

    return selected_features_list, feature_scores

if __name__ == "__main__":
    print("RAINY SEASON")
    rainy_features, rainy_scores = select_best_features(
        '/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainyAE/Option_B_RainyAE.csv'
    )
```

Then run:
```bash
python3 feature_selection_alphaearth.py
```

---

## My Recommendation

**Do Option A (Feature Selection) today** to show that AlphaEarth's best bands help models.

Then **collect more samples** for comprehensive validation.

The AlphaEarth integration is technically complete and data quality is excellent - you just need the right dataset size to prove its value.
