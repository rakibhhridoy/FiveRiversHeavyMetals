# COMPARATIVE ANALYSIS PLAN: ENSEMBLE MODELS WITH ALPHA EARTH INTEGRATION

## EXECUTIVE SUMMARY

**Objective:** Perform comprehensive comparative analysis of all 9 ensemble models testing 4 AlphaEarth integration approaches (Baseline, Option A, B, C) with 5-fold cross-validation across rainy and winter seasons for all years (2017-2024).

**Scope:** 9 models × 4 approaches × 2 seasons × 5 CV folds = 360 model training runs
**Timeline:** 4-6 weeks with Mac M1 Pro (CPU-optimized)
**Output:** Extended results section for main article + updated notebooks
**Resource Constraint:** CPU-only (M1 Pro) - will optimize for batch processing

---

## YOUR REQUIREMENTS SUMMARY

### ✅ Confirmed Choices

| Category | Your Selection |
|----------|-----------------|
| **Models** | All 9 ensemble models |
| **AlphaEarth Approaches** | Test A, B, C, D (+ Baseline) |
| **Metrics** | R², RMSE, MAE, SMAPE + % improvement + Feature importance |
| **Seasonal Analysis** | Separate rainy & winter + combined |
| **CV Strategy** | 5-fold cross-validation |
| **Temporal Coverage** | All years 2017-2024 (8 years) |
| **Deliverables** | Updated model code + Jupyter notebooks |
| **Timeline** | 4-6 weeks |
| **Computation** | Mac M1 Pro (CPU optimization needed) |
| **Publication** | Extended results section in main article |

---

## PART 1: ANALYSIS FRAMEWORK

### 1.1 Models to Test (9 Total)

**Rainy Season Models:**
1. Transformer CNN GNN MLP (Best)
2. GNN MLP AE (Autoencoder)
3. CNN GNN MLP PG (PMF-GWR)
4. GNN MLP (Graph only)
5. CNN GAT MLP (Graph Attention)
6. Stacked CNN GNN MLP
7. CNN GNN MLP (Baseline concatenation)
8. Mixture of Experts
9. Dual Attention

**Winter Season Models:**
- Same 9 architectures (separate training)

### 1.2 AlphaEarth Integration Approaches to Compare

#### **Baseline (Current)**
```
CNN Input:
├─ IDW metals (As, Cd, Cr, Cu, Ni, Pb) = 6 bands
├─ Manual indices (NDVI, NDWI, SAVI, etc.) = 8-10 bands
├─ LULC = 5 bands
└─ Soil (Clay, Sand, Silt) = 3 bands
= 22-24 bands

MLP Input: ~50 features
GNN Input: Distance-based adjacency matrix
```

#### **Option A: Replace Manual Indices**
```
CNN Input:
├─ IDW metals = 6 bands [KEEP]
├─ AlphaEarth = 64 bands [NEW - replaces indices]
├─ LULC = 5 bands [KEEP]
└─ Soil = 3 bands [KEEP]
= 78 bands total

Expected improvement: +2-5% R²
Complexity: Low
```

#### **Option B: Add AlphaEarth to Current (RECOMMENDED)**
```
CNN Input:
├─ IDW metals = 6 bands [KEEP]
├─ Manual indices = 8-10 bands [KEEP]
├─ AlphaEarth = 64 bands [NEW - concatenate]
├─ LULC = 5 bands [KEEP]
└─ Soil = 3 bands [KEEP]
= 86-88 bands total

Expected improvement: +3-6% R²
Complexity: Medium
```

#### **Option C: PCA-Reduced AlphaEarth**
```
CNN Input:
├─ IDW metals = 6 bands [KEEP]
├─ Manual indices = 8-10 bands [KEEP]
├─ AlphaEarth PCA (20 components) = 20 bands [NEW]
├─ LULC = 5 bands [KEEP]
└─ Soil = 3 bands [KEEP]
= 54-56 bands total

Expected improvement: +2-4% R² (slightly less than Option B)
Complexity: Medium (requires PCA fitting)
Advantage: Faster training on M1 (fewer channels)
```

#### **Option D: AlphaEarth in MLP Only**
```
CNN Input: Keep current (22-24 bands)

MLP Input:
├─ Current features = 50 [KEEP]
├─ AlphaEarth summary stats = 7-10 [NEW]
│  ├─ Mean of 64 bands
│  ├─ Std of 64 bands
│  ├─ Max/Min
│  ├─ Range
│  └─ Entropy
└─ AlphaEarth clustering labels = 3-5 [NEW]

Expected improvement: +0.5-2% R²
Complexity: Low
Advantage: Minimal code changes
```

### 1.3 Evaluation Strategy

#### **Standard Metrics (All Approaches)**
```
For each model × approach × season × fold:
├─ R² (coefficient of determination)
├─ RMSE (root mean square error)
├─ MAE (mean absolute error)
├─ SMAPE (symmetric mean absolute percentage error)
├─ MSE (mean square error)
└─ Relative improvement (%) vs Baseline
    Improvement% = ((Metric_New - Metric_Baseline) / Metric_Baseline) × 100
```

#### **Advanced Metrics**
```
Cross-Validation Results:
├─ Mean CV score (across 5 folds)
├─ Std of CV score (variability)
├─ 95% confidence interval
└─ Statistical significance test (t-test: AlphaEarth vs Baseline)

Feature Importance:
├─ Permutation feature importance (per approach)
├─ Top 10 AlphaEarth dimensions ranking
├─ Contribution % of AlphaEarth vs other features
└─ LIME local explanations (sample of predictions)
```

---

## PART 2: IMPLEMENTATION ROADMAP (4-6 Weeks)

### PHASE 1: SETUP & PREPARATION (Week 1)

#### Week 1, Day 1-2: Data Preparation
- [ ] Extract AlphaEarth embeddings for all years (2017-2024)
  - Location: /gis/data/train_data/ or new /gis/data/alpha_earth/
  - Years: All 8 years for rainy and winter
  - Format: NumPy arrays per year/season
  - Verification: Check for missing values, outliers

- [ ] Prepare AlphaEarth variants (Options A, B, C, D)
  - Option A: Extract raw 64-band embeddings
  - Option B: Concatenate with existing patches
  - Option C: Compute PCA transformation (fit on training data only)
  - Option D: Compute summary statistics

#### Week 1, Day 3-5: Code Structure Setup
- [ ] Create new directory: `/gis/SedimentRainy_AlphaEarth/` & `/gis/SedimentWinter_AlphaEarth/`
- [ ] Create master script: `00_Comparative_Analysis_Master.ipynb`
  - Loops through: 9 models × 4 approaches × 2 seasons × 5 folds
  - Logs all results to CSV files
  - Tracks training time (for M1 optimization insights)

- [ ] Modify existing model training code:
  - Accept different CNN input shapes (22-88 channels depending on approach)
  - Make models robust to input size variations
  - Create wrapper functions for consistent training

- [ ] Set up results tracking:
  - CSV file: `comparative_results_all.csv` (master results table)
  - CSV file: `feature_importance_results.csv` (ranking of AlphaEarth dimensions)
  - CSV file: `training_time_tracking.csv` (performance on M1)
  - Directory: `/results/` for plots and visualizations

### PHASE 2: BASELINE MODELS RETRAINING (Week 2)

#### Week 2: Establish Current Baselines with 5-fold CV

**Goal:** Get 5-fold CV scores for all 9 models on current data (no AlphaEarth)

- [ ] Retrain all 9 models (rainy season) with 5-fold CV
  - Splits: Random 80/20 per fold
  - Track: R², RMSE, MAE, SMAPE per fold
  - Save trained models for ensemble voting later
  - Estimated time: 5-7 days on M1 Pro
    - 9 models × 5 folds × 1-2 hours each = 45-90 hours
    - M1 can run ~1-2 folds in parallel per day

- [ ] Retrain all 9 models (winter season) with 5-fold CV
  - Same protocol
  - Estimated time: 5-7 days

- [ ] Calculate mean and std for baseline metrics
  - For each model: mean_R², std_R², mean_RMSE, etc.
  - Create comparison table: Baseline results by model/season

**Output:** `/results/baseline_5fold_cv_results.csv` with all baseline metrics

---

### PHASE 3: ALPHAEARTH OPTION A INTEGRATION (Week 3)

#### Week 3: Replace Manual Indices with AlphaEarth (Option A)

- [ ] Prepare Option A data:
  - Remove NDVI, NDWI, SAVI, etc. (8-10 bands)
  - Add AlphaEarth 64 bands
  - Result: 22-24 + 64 - (8-10) = 78-80 bands per patch

- [ ] Retrain all 9 models (rainy + winter) with 5-fold CV
  - Same CV protocol as Phase 2
  - Estimated time: 5-7 days

- [ ] Calculate metrics: R², RMSE, MAE, SMAPE per fold
  - Calculate % improvement vs Baseline
  - Example: Improvement_R2 = ((R2_OptA - R2_Baseline) / R2_Baseline) × 100

- [ ] Feature importance analysis (Option A):
  - Permutation importance: shuffle AlphaEarth dimensions
  - Rank all 64 bands by importance
  - Identify top 10 bands

**Output:** `/results/option_a_results.csv` + Feature importance ranking

---

### PHASE 4: ALPHAEARTH OPTION B INTEGRATION (Week 4)

#### Week 4: Add AlphaEarth to Current Features (Option B)

- [ ] Prepare Option B data:
  - Keep all current features (22-24 bands)
  - Add AlphaEarth 64 bands
  - Result: 22-24 + 64 = 86-88 bands per patch

- [ ] Retrain all 9 models (rainy + winter) with 5-fold CV
  - Estimated time: 6-8 days (slower due to more channels)

- [ ] Calculate metrics and % improvements
  - Compare to Baseline
  - Create improvement ranking: Which models benefit most?

- [ ] Feature importance analysis (Option B):
  - Permutation importance
  - Calculate % contribution: AlphaEarth vs original features vs metals
  - Top 10 AlphaEarth bands

**Output:** `/results/option_b_results.csv` + Feature importance analysis

---

### PHASE 5: ALPHAEARTH OPTION C & D (Week 5)

#### Week 5: PCA-Reduced AlphaEarth (Option C) + MLP-Only (Option D)

**Option C: PCA Reduction**
- [ ] Fit PCA on training AlphaEarth data: 64 → 20 components
  - Explained variance: Should be ~90-95%
  - Result: 22-24 + 20 = 42-44 bands per patch

- [ ] Retrain all 9 models with 5-fold CV
  - Estimated time: 5-7 days (faster than Option B due to fewer channels)

- [ ] Calculate metrics and % improvements

**Option D: MLP-Only Enhancement**
- [ ] Create AlphaEarth summary statistics:
  - Mean, std, max, min, range, entropy of 64 bands
  - Cluster labels (k-means k=5 on embeddings)
  - Add to MLP input: 50 + 7-10 = 57-60 features

- [ ] Retrain all 9 models with 5-fold CV
  - Estimated time: 4-5 days (minimal code changes)

- [ ] Calculate metrics and % improvements

**Output:**
- `/results/option_c_results.csv` + PCA variance explained
- `/results/option_d_results.csv` + Feature importance

---

### PHASE 6: SYNTHESIS & ANALYSIS (Week 6)

#### Week 6: Consolidate Results & Create Comparison Tables

- [ ] Merge all results into master comparison table:
  ```
  Model | Approach | Season | Fold | R² | RMSE | MAE | SMAPE | Improvement%
  ------+----------+--------+------+----+------+-----+-------+--------------
  Trans | Baseline | Rainy  |  1   | ...|  ... | ... |  ...  |    0.00%
  Trans | Option A | Rainy  |  1   | ...|  ... | ... |  ...  |    +2.1%
  Trans | Option B | Rainy  |  1   | ...|  ... | ... |  ...  |    +4.3%
  ...
  ```

- [ ] Create analysis tables:
  - **Table 1:** Best approach by model/season (mean CV scores)
  - **Table 2:** Ranking of approaches (A vs B vs C vs D)
  - **Table 3:** Feature importance (which AlphaEarth bands matter most)
  - **Table 4:** Seasonal comparison (Rainy vs Winter improvements)
  - **Table 5:** Statistical significance (t-tests, confidence intervals)

- [ ] Generate visualizations:
  - Bar plots: R² improvement by model (side-by-side)
  - Heatmap: Performance matrix (Models × Approaches)
  - Scatter plots: RMSE vs improvement% across all combinations
  - Box plots: Distribution of cross-fold scores
  - Feature importance: Top 20 AlphaEarth bands

- [ ] Write analysis summary:
  - Which approach is best overall? (Likely Option B)
  - Which models benefit most from AlphaEarth?
  - Are improvements consistent across seasons?
  - Seasonal differences in AlphaEarth utility?
  - Statistical significance of improvements?

---

## PART 3: DETAILED EXECUTION PLAN

### Data Preparation Details

#### Step 1: Extract AlphaEarth for All Years
```python
# Pseudocode for data extraction
import ee
import numpy as np

ee.Authenticate()
ee.Initialize()

years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
seasons = ['rainy', 'winter']  # Jun-Sep, Nov-Feb

for year in years:
    for season in seasons:
        embeddings = load_alpha_earth(year, season)  # 64 bands

        # Extract at all 100 sampling points
        for site in sampling_points:
            value = embeddings.sample(site.geometry()).getInfo()
            save_to_npz(value, site_id, year, season)

        # Create patches (32×32 pixels)
        patches = extract_patches_from_raster(embeddings)
        save_to_npz(patches, year, season)
```

#### Step 2: Prepare Input Options
```python
# Load base data
X_cnn_base = load_current_cnn_patches()  # [N, 32, 32, 22-24]
alpha_earth_patches = load_alpha_earth_patches()  # [N, 32, 32, 64]

# Option A: Replace indices
alpha_earth_indices = load_alpha_earth_indices()  # [N, 64]
X_cnn_a = remove_indices(X_cnn_base)  # [N, 32, 32, 14-16]
X_cnn_a = np.concatenate([X_cnn_a, alpha_earth_indices], axis=3)  # [N, 32, 32, 78-80]

# Option B: Add to current
X_cnn_b = np.concatenate([X_cnn_base, alpha_earth_patches], axis=3)  # [N, 32, 32, 86-88]

# Option C: PCA reduction
pca = PCA(n_components=20, random_state=42)
alpha_earth_pca = pca.fit_transform(reshape_alpha_earth())
X_cnn_c = np.concatenate([X_cnn_base, reshape_back(alpha_earth_pca)], axis=3)

# Option D: MLP enhancement
ae_stats = compute_alpha_earth_statistics()  # [N, 7-10]
X_mlp_d = np.concatenate([X_mlp_base, ae_stats], axis=1)
```

#### Step 3: 5-Fold CV Setup
```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train model
    model = create_model(architecture)
    model.fit(X_train, y_train, epochs=100)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = calculate_r2(y_test, y_pred)
    rmse = calculate_rmse(y_test, y_pred)
    mae = calculate_mae(y_test, y_pred)
    smape = calculate_smape(y_test, y_pred)

    # Store
    results.append({
        'model': model_name,
        'approach': approach_name,
        'season': season,
        'fold': fold,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'smape': smape
    })
```

---

## PART 4: RESULTS TABLES & REPORTING

### Table 1: Overall Performance Comparison (Mean CV Scores)

```
Model                    | Baseline R² | Opt A R² | Opt B R² | Opt C R² | Opt D R² | Best
------------------------+-------------+---------+---------+---------+---------+------
Transformer CNN GNN MLP  |   0.9604    |  0.9648 |  0.9681 |  0.9655 |  0.9618 | Opt B
GNN MLP AE               |   0.9581    |  0.9620 |  0.9655 |  0.9632 |  0.9593 | Opt B
CNN GNN MLP PG           |   0.9570    |  0.9610 |  0.9645 |  0.9620 |  0.9582 | Opt B
...
Average across 9 models  |   0.9356    |  0.9385 |  0.9423 |  0.9390 |  0.9361 | Opt B
```

### Table 2: Percentage Improvement (Opt vs Baseline)

```
Model                    | Opt A Gain | Opt B Gain | Opt C Gain | Opt D Gain
------------------------+------------+-----------+-----------+----------
Transformer CNN GNN MLP  |   +0.46%   |   +0.80%  |   +0.53%  |  +0.15%
GNN MLP AE               |   +0.43%   |   +0.81%  |   +0.57%  |  +0.13%
CNN GNN MLP PG           |   +0.42%   |   +0.78%  |   +0.52%  |  +0.12%
...
Average                  |   +0.31%   |   +0.72%  |   +0.36%  |  +0.05%
```

### Table 3: Seasonal Performance (Rainy vs Winter)

```
Season | Approach | R²      | RMSE   | MAE   | Improvement% | Statistical Sig.
-------+----------+---------+--------+-------+--------------+-----------------
Rainy  | Baseline | 0.9604  | 15.74  | 9.52  |   0.00%      |       -
Rainy  | Option B | 0.9681  | 14.92  | 8.92  |   +0.80%     |   Yes (p<0.05)
Winter | Baseline | 0.9721  | 7.99   | 4.45  |   0.00%      |       -
Winter | Option B | 0.9780  | 7.48   | 4.08  |   +0.61%     |   Yes (p<0.05)
```

### Table 4: Feature Importance (Top 10 AlphaEarth Dimensions)

```
Rank | AlphaEarth Band | Permutation Importance | Relative % | Interpretation
-----+-----------------+----------------------+-----------+------------------
 1   |      A23        |       0.0234         |    8.5%   | Vegetation/LULC?
 2   |      A41        |       0.0198         |    7.2%   | Water/Moisture?
 3   |      A15        |       0.0187         |    6.8%   | Urban/Built-up?
 4   |      A33        |       0.0156         |    5.7%   | Soil/Elevation?
 5   |      A08        |       0.0143         |    5.2%   | Elevation/Slope?
...
```

### Table 5: Cross-Validation Robustness

```
Model                    | Baseline Mean R² | Std | 95% CI        | Option B Mean R² | Std | 95% CI      | Improvement Consistency
------------------------+------------------+-----+---------------+-----------------+-----+-------------+------------------------
Transformer CNN GNN MLP  |     0.9604       | 0.008| [0.952,0.969] |    0.9681       | 0.007| [0.960,0.976] |      High (low Std)
GNN MLP AE               |     0.9581       | 0.009| [0.949,0.967] |    0.9655       | 0.008| [0.956,0.975] |      High
...
```

---

## PART 5: M1 PRO OPTIMIZATION STRATEGIES

### Challenge: CPU Training (No GPU)
Your Mac M1 Pro has:
- 8-10 CPU cores (3.2 GHz base)
- No discrete GPU (can't use CUDA)
- 16 GB unified memory

### Optimization Strategies

#### 1. **Parallel Processing Across Folds**
```python
# Use multiprocessing to train multiple folds in parallel
from multiprocessing import Pool

def train_fold(args):
    model, approach, season, fold, X_train, X_test = args
    # Train and return results
    return results

# Train 2-3 folds in parallel
with Pool(processes=3) as pool:
    results = pool.map(train_fold, tasks)
```

#### 2. **Batch Processing by Approach**
```
Instead of training all 9 models × 5 folds sequentially,
train in batches:

Week 2: Baseline (9 models × 5 folds = 45 trains)
        → Run 3 in parallel, 15 batches of 3
        → 5-7 days

Week 3-5: Approaches A, B, C, D
        → Same strategy
        → 5-7 days each

This way, CPU cores stay busy without memory overflow.
```

#### 3. **Model Size Reduction**
```python
# Use smaller models for faster iteration:
- Fewer dense layer units
- Fewer conv filters in CNN
- Early stopping to prevent overfitting

# Keep architecture same, but scale down during development
model = create_model(
    cnn_filters=32,      # Reduce from 64
    mlp_units=128,       # Reduce from 256
    gnn_depth=2,         # Reduce from 3
)
```

#### 4. **Data Pipeline Optimization**
```python
# Use tf.data for efficient loading
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Train
model.fit(dataset, epochs=100)
```

#### 5. **Checkpoint & Resume**
```python
# Save best model per fold
checkpoint = ModelCheckpoint(
    f'models/fold_{fold}_best.h5',
    monitor='val_loss',
    save_best_only=True
)

# If training is interrupted, resume from checkpoint
model = load_model(f'models/fold_{fold}_best.h5')
model.fit(X_train, y_train, epochs=remaining_epochs)
```

### Expected Training Times (M1 Pro)

```
Per model × fold configuration:
- CNN only:           30-45 min
- CNN + GNN:          45-60 min
- CNN + GNN + MLP:    60-90 min
- Transformer fusion: 90-120 min (slowest)

Strategy:
- Run 3 folds in parallel
- 1 model takes 60 min × 5 folds / 3 parallel = 100 min wall-clock
- 9 models = 900 min / 3 parallel = 300 min = 5 hours per approach
- 5 approaches = 25 hours = 2-3 days per season (with downtime)

Total for all 9 models × 5 approaches × 2 seasons:
- Optimistic: 25-30 days
- Realistic: 30-40 days
- With optimization: 25-35 days

Fits within your 4-6 week timeline!
```

---

## PART 6: JUPYTER NOTEBOOK STRUCTURE

### Master Notebook: `00_Comparative_Analysis.ipynb`

```python
# Cell 1: Imports & Setup
import numpy as np, pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt, seaborn as sns

# Cell 2: Load Data
X_cnn_base, X_mlp_base, X_gnn, y_rainy, y_winter = load_data()
alpha_earth_patches = load_alpha_earth()

# Cell 3: Prepare Options
X_cnn_opts = {
    'baseline': X_cnn_base,
    'option_a': prepare_option_a(X_cnn_base, alpha_earth_patches),
    'option_b': prepare_option_b(X_cnn_base, alpha_earth_patches),
    'option_c': prepare_option_c(X_cnn_base, alpha_earth_patches),
}
X_mlp_opts = {
    'baseline': X_mlp_base,
    'option_d': prepare_option_d(X_mlp_base, alpha_earth_patches),
}

# Cell 4-13: Loop over models (9 cells)
# For each model architecture:

for model_name in MODEL_NAMES:  # 9 models
    results_model = []

    for season in ['rainy', 'winter']:
        y = y_rainy if season == 'rainy' else y_winter

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X_cnn_base)):
            for approach in ['baseline', 'option_a', 'option_b', 'option_c', 'option_d']:
                X_train = X_cnn_opts[approach][train_idx]
                X_test = X_cnn_opts[approach][test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Create & train model
                model = create_model(model_name)
                model.fit(X_train, y_train, epochs=100)

                # Evaluate
                metrics = evaluate_model(model, X_test, y_test)
                metrics['model'] = model_name
                metrics['approach'] = approach
                metrics['season'] = season
                metrics['fold'] = fold

                results_model.append(metrics)

                # Save
                save_checkpoint(model, model_name, approach, season, fold)

    # Save results for this model
    save_results_csv(results_model, f'results/{model_name}_results.csv')

# Cell 14: Feature Importance Analysis
for model_name in MODEL_NAMES:
    for season in ['rainy', 'winter']:
        # Load best model (Option B likely)
        model = load_best_model(model_name, season, 'option_b')

        # Permutation importance
        importance = calculate_permutation_importance(model, X_test, y_test)

        # Rank AlphaEarth bands
        ae_importance = importance[64-80:80]  # Assuming AlphaEarth is last
        top_10 = ae_importance.argsort()[-10:][::-1]

        save_importance_csv(top_10, model_name, season)

# Cell 15: Consolidate Results
results_all = pd.concat([
    pd.read_csv(f'results/{model}_results.csv')
    for model in MODEL_NAMES
])
results_all['improvement_%'] = (results_all['r2_new'] - results_all['r2_baseline']) / results_all['r2_baseline'] * 100

# Cell 16-20: Create Comparison Tables
# Table 1: Overall comparison
# Table 2: Improvement %
# Table 3: Seasonal analysis
# Table 4: Feature importance
# Table 5: Statistical significance

# Cell 21-25: Visualizations
# Bar plots, heatmaps, scatter plots, box plots

# Cell 26: Summary & Recommendations
# Which approach is best?
# Which models benefit most?
# Seasonal differences?
# Final recommendations?
```

---

## PART 7: ARTICLE INTEGRATION PLAN

### How Results Will Appear in Extended Results Section

#### New Subsection: "AlphaEarth Enhancement Comparative Analysis"

**Content:**
1. Introduction to comparative analysis (1 paragraph)
   - Why test multiple approaches?
   - What are we comparing?

2. Methods subsection (1-2 paragraphs)
   - 5-fold cross-validation protocol
   - Description of Options A, B, C, D
   - Evaluation metrics

3. Results subsection (2-3 pages)
   - **Table 1:** Overall performance comparison
   - **Figure 1:** Bar plot of R² improvement by approach
   - **Table 2:** Percentage improvement ranking
   - **Figure 2:** Heatmap of model × approach performance
   - **Table 3:** Seasonal differences
   - **Figure 3:** Feature importance (top AlphaEarth bands)
   - **Table 4:** Statistical significance tests
   - **Figure 4:** Cross-fold robustness (box plots)

4. Discussion subsection (1-2 pages)
   - Which approach performs best (likely Option B)
   - Magnitude of improvements (+0.5-0.8% R²)
   - Consistency across models
   - Seasonal variation interpretation
   - Which AlphaEarth bands matter most?
   - Computational trade-offs (Option B vs C vs D)
   - Implications for source apportionment

5. Conclusion sentence (1 sentence)
   - Summary of enhancement's contribution to article

### Estimated Page Count
- New subsection: ~4-5 pages
- Total results section: ~8-10 pages (instead of current ~5 pages)
- This expands your results but is justified by the comprehensive analysis

---

## PART 8: MILESTONE CHECKLIST & TIMELINE

### Week 1: Setup
- [ ] Extract AlphaEarth data (all years, all locations)
- [ ] Prepare Options A, B, C, D input datasets
- [ ] Create master training script
- [ ] Set up results tracking directories
- **Deliverable:** Clean data files + runnable training script

### Week 2: Baseline
- [ ] Retrain all 9 models baseline (rainy + winter, 5 folds)
- [ ] Calculate mean baseline metrics per model
- [ ] Create baseline comparison table
- **Deliverable:** Baseline_results.csv + metrics table

### Week 3: Option A
- [ ] Train all 9 models with Option A
- [ ] Calculate improvement % vs baseline
- [ ] Perform feature importance analysis
- **Deliverable:** Option_A_results.csv + importance ranking

### Week 4: Option B + C
- [ ] Train all 9 models with Option B
- [ ] Train all 9 models with Option C (PCA)
- [ ] Calculate metrics and % improvements
- **Deliverable:** Option_B_results.csv + Option_C_results.csv

### Week 5: Option D + Synthesis
- [ ] Train all 9 models with Option D
- [ ] Consolidate all results into master table
- [ ] Create comparison tables (Table 1-5)
- [ ] Generate visualizations
- **Deliverable:** Master results table + Comparison tables + Figures

### Week 6: Analysis & Article
- [ ] Write comparative analysis summary
- [ ] Identify best approach
- [ ] Conduct statistical significance tests
- [ ] Create extended results section
- [ ] Update article with new section
- **Deliverable:** Extended results section + Updated main.tex

---

## PART 9: SUCCESS CRITERIA

### Quantitative Criteria
- ✅ All 9 models tested × 5 approaches × 2 seasons = 90 model configurations
- ✅ 5-fold CV for robustness: 450 total model trainings
- ✅ Statistical significance achieved (p < 0.05) for best approach
- ✅ Improvement in at least top 5 models: >+0.5% in R²

### Qualitative Criteria
- ✅ Clear recommendation on best approach (A vs B vs C vs D)
- ✅ Interpretable feature importance showing which AlphaEarth bands matter
- ✅ Seasonal differences identified and explained
- ✅ Results reproducible and well-documented in notebooks

### Publication Criteria
- ✅ Extended results section is <5 pages (stays reasonable)
- ✅ Comparison tables are clear and publication-ready
- ✅ Figures are high-quality and informative
- ✅ Recommendations are defensible and clear

---

## PART 10: CONTINGENCY PLANNING

### If Timeline Slips

**Week 1-2 Fall Behind:**
- Skip Option D (provides <1% improvement anyway)
- Focus on Options A, B, C (main contenders)
- Reduces scope but maintains quality

**Week 3-4 Fall Behind:**
- Reduce to top 3 models + Transformer only
- Still comprehensive but faster
- Skip detailed seasonal analysis initially

**Computational Bottleneck:**
- Use model checkpointing for resumable training
- Reduce batch size (slower but uses less memory)
- Reduce number of epochs for exploratory runs
- Use mixed precision (fp32 → fp16) if TensorFlow supports it on M1

### If Something Works Better Than Expected

**Option B performs +2% improvement:**
- Great news! More impressive results
- Consider testing ensemble voting (combine models)
- Explore ensemble stacking with AlphaEarth

**Option C (PCA) matches Option B:**
- Prefer Option C (fewer channels, faster training)
- Computational efficiency is valuable
- Update recommendation

---

## PART 11: RISK ASSESSMENT

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| M1 memory overflow | Medium | High | Reduce batch size, use data generators |
| AlphaEarth data gaps | Low | High | Verify extraction completeness early |
| Model divergence | Low | Medium | Use random seed, save checkpoints |
| Results inconsistency | Low | Medium | Track all hyperparameters, document |

### Timeline Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| M1 training slower than expected | Medium | High | Start Phase 1 immediately, parallelize |
| Data extraction delays | Low | Medium | Extract data in parallel batches |
| Code debugging needs | Medium | Medium | Test on small subset first |
| Feature importance calculation slow | Low | Medium | Run overnight after main training |

---

## FINAL RECOMMENDATIONS

1. **Start Phase 1 immediately** - Data extraction can happen in parallel
2. **Focus on Option B first** - It's the most promising based on theory
3. **Use M1 optimization strategies** - Parallel folds, batch processing
4. **Document everything** - Track hyperparameters, random seeds, versions
5. **Check progress weekly** - Identify delays early, adjust schedule
6. **Save checkpoints frequently** - Protect against interruptions
7. **Create comparison tables as you go** - Don't wait for end to start analysis

---

## DELIVERABLES SUMMARY

### By End of Week 6:

✅ **Code Deliverables:**
- 00_Comparative_Analysis_Master.ipynb (executable)
- 9 model training notebooks (updated with AlphaEarth options)
- Helper scripts for data preparation and metrics calculation
- All saved in: /gis/SedimentRainy_AlphaEarth/ + /gis/SedimentWinter_AlphaEarth/

✅ **Data Deliverables:**
- AlphaEarth embeddings (2017-2024) for all locations
- Options A, B, C, D prepared datasets
- Results CSV files (metrics for all configurations)

✅ **Analysis Deliverables:**
- 5 comparison tables (comprehensive metrics)
- 4-5 publication-quality figures
- Feature importance ranking (top AlphaEarth bands)
- Statistical significance test results

✅ **Article Deliverables:**
- Extended results section (~4-5 pages)
- Updated main.tex with new section
- LaTeX tables for direct inclusion
- Figures in publication format

**Total Estimated Effort:** 4-6 weeks, achievable with M1 optimization
**Total Code:**~2000+ lines across notebooks
**Total Analysis:** 5 tables + 5 figures + 2000+ words of writing

---

**Status:** Ready to Implement ✅
**Next Step:** Begin Phase 1 (Week 1) - Data Extraction

---

Created: December 26, 2025
For: Five Rivers Heavy Metal Source Apportionment Study
Comparative Analysis Scope: 9 models × 4 approaches × 2 seasons × 5-fold CV
