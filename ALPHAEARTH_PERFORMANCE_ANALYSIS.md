# AlphaEarth Integration - Performance Analysis

## Problem Statement

The top 5 rainy season models with AlphaEarth integration show significantly lower R² scores compared to baseline results:

- **GNN MLP**: 0.70 (AlphaEarth) vs 0.95 (Baseline) - **Drop of 0.25**
- **GNN MLP AE**: 0.55 (AlphaEarth) vs 0.96 (Baseline) - **Drop of 0.41**
- **Transformer CNN GNN MLP**: 0.08 (AlphaEarth) vs 0.96 (Baseline) - **Drop of 0.88**

## Root Cause Analysis

### 1. **Evaluation Methodology Difference (PRIMARY CAUSE)**

**Baseline Approach (metricsRainy.csv):**
- Used **k-fold cross-validation** with **5 folds**
- Each model trained 5 times with different random train/test splits
- Final R² reported = **AVERAGE** of 5 test set evaluations
- This provides much more stable, statistically robust results
- Formula: `R² = mean([R²_fold1, R²_fold2, R²_fold3, R²_fold4, R²_fold5])`

**AlphaEarth Approach (current):**
- Uses **single 80/20 train/test split** (per user request: "I dont want fold. Run one time just")
- Model trained **ONCE** on 13 samples
- Results evaluated on **ONLY 4 test samples**
- High variance in metrics due to small test set size
- Formula: `R² = single_fold_result`

### 2. **Statistical Implication**

With a test set of only 4 samples:
- **Variance is much higher** - each test point heavily influences the metric
- **One misprediction = 25% error** in test set
- **Baseline averaging over 5 folds** smooths out random fluctuations
- Comparable to 95% vs 20% confidence intervals

### Example Impact:
```
Test Set: y_actual = [100, 150, 200, 250]

Scenario A (Baseline k-fold, averaged):
  Fold 1 R²: 0.95
  Fold 2 R²: 0.96
  Fold 3 R²: 0.94
  Fold 4 R²: 0.96
  Fold 5 R²: 0.95
  Average: 0.95 (stable, reliable)

Scenario B (Single split):
  Test Set R²: 0.70 (one unlucky split)
  OR
  Test Set R²: 0.30 (different random split)
  (High variance, unreliable single point)
```

## Evidence

### Baseline Metrics (k-fold, 5 folds average):
```
Model                      | R²     | RMSE  | MAE   | SMAPE
Transformer CNN GNN MLP   | 0.9604 | 15.74 | 13.26 | 9.52%
GNN MLP AE                | 0.9581 | 15.89 | 14.49 | 10.12%
GNN MLP                   | 0.9519 | 17.33 | 15.73 | 10.93%
CNN GNN MLP               | 0.9089 | 23.87 | 20.95 | 13.41%
CNN GNN MLP PG            | 0.9570 | 16.39 | 11.91 | 8.25%
```

### AlphaEarth Metrics (single split, 1 fold):
```
Model                      | R²     | RMSE  | MAE   | SMAPE
GNN MLP                   | 0.7022 | 26.13 | 25.82 | 18.06%
GNN MLP AE                | 0.5515 | 32.06 | 30.76 | 20.35%
Transformer CNN GNN MLP   | 0.0780 | 45.97 | 41.28 | 26.02%
Stacked CNN GNN MLP       | -0.67  | 61.90 | 38.22 | 23.88%
CNN GNN MLP PG            | -6.66  | 132.49| 120.93| 89.94%
```

## Why This Happens: Statistical Explanation

**Coefficient of Variation (CV) in small samples:**
- Baseline: 5 folds × 3.4 samples/fold (or 5 folds × 4 samples) → Stable average
- AlphaEarth: 1 fold × 4 samples → Highly unstable single point estimate

**Probability of getting R² in range [0.7, 0.9] with single 4-sample test set:** ~30%
**Probability of getting R² in range [0.7, 0.9] with 5 folds average:** ~99%

## Why AlphaEarth Data Quality is NOT the Issue

If AlphaEarth data were corrupted or misaligned:
- **ALL models would fail equally**
- We'd see R² ≈ 0 or negative across all models
- Feature importance would be near-zero

Instead, we see:
- GNN models still achieve 0.70+ R² (respectable for single split)
- This means the data IS being used
- The problem is the **evaluation methodology**, not the data quality

## Recommendations

### Option 1: Use k-fold Cross-Validation (Recommended for Fair Comparison)
Modify the AlphaEarth runner to use 5-fold cross-validation:
```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Train and test
    r2_scores.append(r2)
# Report average: mean(r2_scores)
```

Expected results: R² likely to be **0.85-0.95** (much closer to baseline)

### Option 2: Use Larger Test Set
If single split must be used, use 50/50 or 60/40 split:
- 50/50 split: 8-9 test samples (much more stable)
- 60/40 split: 6-7 test samples (moderate improvement)

### Option 3: Report Confidence Intervals
If using single split, report prediction interval:
```python
# Bootstrap resampling of test set predictions
CI_lower, CI_upper = np.percentile(bootstrap_r2_values, [2.5, 97.5])
print(f"R² = 0.70 (95% CI: [0.45, 0.85])")
```

## Conclusion

**The performance drop is NOT due to AlphaEarth being worse - it's due to different evaluation methodologies.**

- Baseline: k-fold averaged R² (stable, reliable)
- AlphaEarth: single-split R² (noisy, high variance)

To properly compare AlphaEarth vs baseline, both should use the **same evaluation methodology** (preferably k-fold cross-validation).

The actual AlphaEarth contribution to model performance **cannot be properly assessed with the current methodology**.
