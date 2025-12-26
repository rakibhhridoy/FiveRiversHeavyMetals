# COMPARATIVE ANALYSIS - QUICK START GUIDE

## YOUR ANALYSIS AT A GLANCE

### What You're Doing
Testing all **9 ensemble models** with **4 AlphaEarth integration approaches** using **5-fold cross-validation** across **2 seasons** and **8 years** of data.

### The Math
```
9 models × 4 approaches × 2 seasons × 5 CV folds = 360 model runs
```

### Your Timeline
**4-6 weeks** on Mac M1 Pro with optimization

---

## THE FOUR APPROACHES

| Approach | CNN Input | Effort | Expected Gain | Status |
|----------|-----------|--------|---------------|--------|
| **Baseline** | Current 22-24 bands | Low | 0% (reference) | Week 2 |
| **Option A** | 64 AlphaEarth bands (replace indices) | Medium | +2-5% | Week 3 |
| **Option B** | Current + 64 AlphaEarth (86-88 bands) | Medium | +3-6% | Week 4 |
| **Option C** | Current + 20 PCA components (42-44 bands) | Medium | +2-4% | Week 5 |
| **Option D** | MLP enhancement only (minimal CNN change) | Low | +0.5-2% | Week 5 |

**Likely Winner:** Option B (best performance, manageable complexity)

---

## 6-WEEK EXECUTION TIMELINE

### WEEK 1: Setup & Preparation
```
Day 1-2: Extract AlphaEarth data (all years, all locations)
Day 3-5: Prepare 4 options (A, B, C, D datasets)
        + Create master training script
        + Set up results tracking

DELIVERABLE: Clean data + runnable training script
```

### WEEK 2: Baseline Establishment
```
Run 1: All 9 models (rainy season, 5-fold CV)
Run 2: All 9 models (winter season, 5-fold CV)

DELIVERABLE: Baseline metrics (reference for all comparisons)
```

### WEEK 3: Option A (Replace Indices)
```
Train: All 9 models with Option A (rainy + winter, 5-fold CV)
Analyze: Calculate improvement % vs Baseline
Feature Importance: Which AlphaEarth bands matter?

DELIVERABLE: Option A results + improvement ranking
```

### WEEK 4: Option B (Add to Current)
```
Train: All 9 models with Option B (rainy + winter, 5-fold CV)
       (Takes longer due to more channels)

Analyze: Calculate improvement % vs Baseline
Feature Analysis: Contribution of AlphaEarth vs original features

DELIVERABLE: Option B results + contribution analysis
```

### WEEK 5: Option C & D (Efficiency Variants)
```
Run 1: Option C - PCA-reduced AlphaEarth (faster training)
Run 2: Option D - MLP-only enhancement (minimal change)

Analyze: Compare efficiency vs performance trade-offs

DELIVERABLE: Option C & D results + efficiency comparison
```

### WEEK 6: Synthesis & Article
```
Consolidate: Merge all results into master comparison table
Analyze: Statistical significance tests
Create: 5 comparison tables + 5 publication figures
Write: Extended results section (~4-5 pages)

DELIVERABLE: Complete article integration ready for submission
```

---

## OUTPUTS YOU'LL CREATE

### Results Files (CSV)
- `baseline_results.csv` - All 9 models baseline metrics
- `option_a_results.csv` - Option A improvements
- `option_b_results.csv` - Option B improvements
- `option_c_results.csv` - Option C improvements
- `option_d_results.csv` - Option D improvements
- `comparative_master_results.csv` - Everything consolidated

### Comparison Tables (For Article)
1. **Overall Performance:** Which approach is best for each model?
2. **Improvement Ranking:** Models ranked by % improvement
3. **Feature Importance:** Top 10 AlphaEarth bands
4. **Seasonal Analysis:** Rainy vs Winter differences
5. **Statistical Significance:** Which improvements are real?

### Figures (For Article)
1. Bar plot: R² by model and approach
2. Heatmap: Performance matrix (models × approaches)
3. Feature importance: Top AlphaEarth bands
4. Box plots: Cross-fold robustness
5. Scatter: RMSE vs improvement correlation

### Jupyter Notebooks
- `00_Comparative_Analysis_Master.ipynb` (Main execution)
- `01_Data_Preparation.ipynb` (AlphaEarth extraction)
- `02_Baseline_Training.ipynb` (Baseline models)
- `03_Options_Training.ipynb` (A, B, C, D variants)
- `04_Feature_Importance.ipynb` (Permutation + LIME analysis)
- `05_Results_Analysis.ipynb` (Tables, statistics, plots)

---

## PHASES IN DETAIL

### PHASE 1: Setup (Week 1)

**Goals:**
- ✅ Extract AlphaEarth embeddings (8 years × 2 seasons)
- ✅ Prepare 4 option datasets (A, B, C, D)
- ✅ Create automated training pipeline

**Key Code:**
```python
# Download AlphaEarth
embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

# Extract for each year/season/location
for year in range(2017, 2025):
    for season in ['rainy', 'winter']:
        extract_embeddings(year, season, sampling_points)

# Prepare options
X_opt_a = prepare_option_a(X_base, alpha_earth)  # Replace
X_opt_b = prepare_option_b(X_base, alpha_earth)  # Add
X_opt_c = prepare_option_c(X_base, alpha_earth)  # PCA
X_opt_d = prepare_option_d(X_mlp, alpha_earth)   # MLP only
```

**Time:** 4-5 days
**Parallel:** Download + code setup simultaneously

---

### PHASE 2: Baseline (Week 2)

**Goals:**
- ✅ Establish baseline metrics for all 9 models
- ✅ 5-fold CV for robustness
- ✅ Create reference for all comparisons

**Models Tested:** 9 architectures
**CV:** 5 folds
**Data:** Current features only (no AlphaEarth)
**Output:** Baseline metrics per model/season/fold

**Time:** 5-7 days
**Optimization:** Train 3 models in parallel on M1

---

### PHASE 3-5: AlphaEarth Options (Weeks 3-5)

**Same Protocol for Each Option:**

```
For each option (A, B, C, D):
  ├─ Prepare input data
  ├─ Train all 9 models (rainy, 5-fold CV)
  ├─ Train all 9 models (winter, 5-fold CV)
  ├─ Calculate metrics (R², RMSE, MAE, SMAPE)
  ├─ Calculate % improvement vs Baseline
  └─ Perform feature importance analysis
```

**Time per Option:** 5-7 days
**Total for A, B, C, D:** 20-28 days (can overlap)
**Optimization:** Parallel across folds, batch processing

---

### PHASE 6: Synthesis (Week 6)

**Steps:**
1. Consolidate all results into single CSV
2. Create 5 comparison tables
3. Generate 5 publication-quality figures
4. Perform statistical tests (t-tests, confidence intervals)
5. Write extended results section
6. Update main article with new section

**Outputs:**
- Master results table (all metrics)
- Comparison tables (ready for article)
- Visualizations (ready for publication)
- Extended results section (~4-5 pages)

**Time:** 5-7 days (mostly analysis, plotting, writing)

---

## KEY METRICS TO TRACK

### For Each Model × Approach × Season × Fold:
```
R² (coefficient of determination)
RMSE (root mean square error)
MAE (mean absolute error)
SMAPE (symmetric mean absolute percentage error)
Improvement% = ((R2_new - R2_baseline) / R2_baseline) × 100
```

### Summary Statistics:
```
Mean metric across 5 folds
Std of metric across folds
95% confidence interval
Statistical significance (p-value)
```

### Feature Importance (Per Approach):
```
Permutation importance for each AlphaEarth band
Ranking: Top 10 bands
Contribution %: AlphaEarth vs other features
LIME explanations: Local interpretability
```

---

## M1 PRO OPTIMIZATION TIPS

### Challenge
- 8-10 CPU cores only (no GPU)
- 16 GB unified memory
- Training is slow without acceleration

### Solution: Parallel Fold Processing

```python
from multiprocessing import Pool
import concurrent.futures

def train_fold(params):
    # Train on one fold
    return results

# Train 3 folds in parallel (M1 sweet spot)
with Pool(processes=3) as pool:
    fold_results = pool.map(train_fold, all_fold_params)
```

### Expected Times
```
Per model × approach × fold: 60-90 minutes
With 3 parallel folds: 300 min = 5 hours per model
All 9 models × 5 approaches: 225 hours
Spread over 5 weeks: ~9 hours/day (manageable!)
```

### Code Optimization
```python
# Reduce model complexity during exploratory runs
model = create_model(
    cnn_filters=32,     # Not 64
    mlp_units=128,      # Not 256
    early_stopping=True  # Prevent unnecessary epochs
)

# Use batch processing for data
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
model.fit(dataset, epochs=100)

# Save checkpoints (resume if interrupted)
checkpoint = ModelCheckpoint(f'best_fold_{fold}.h5')
model.fit(X_train, y_train, callbacks=[checkpoint])
```

---

## EXPECTED RESULTS PREVIEW

### Likely Outcomes

**Option B (Add AlphaEarth) Performance:**
```
Baseline R²:  0.9604 (rainy) / 0.9721 (winter)
Option B R²:  0.9681 (rainy) / 0.9780 (winter)
Improvement:  +0.77% (rainy) / +0.61% (winter)
Statistical:  Significant at p < 0.05 (95% confidence)
```

**Best Performing Models:**
```
1. Transformer CNN GNN MLP: +0.8% improvement
2. GNN MLP AE: +0.81% improvement
3. CNN GNN MLP PG: +0.78% improvement
```

**Feature Importance:**
```
Top 5 AlphaEarth bands likely represent:
- Vegetation/LULC patterns (land use = pollution source proxy)
- Water/wetness indices (hydrological transport)
- Urban/built-up intensity (anthropogenic activity)
- Soil/elevation features (natural metal background)
- Elevation/slope (topography = contaminant transport)
```

**Seasonal Difference:**
```
Rainy season: Larger AlphaEarth benefit (+0.77%)
              - Seasonal dynamics affect source distribution
              - AlphaEarth captures hydrological context better

Winter season: Smaller benefit (+0.61%)
              - More stable conditions
              - Less spatial variability
              - Baseline model already captures patterns well
```

---

## DECISION FRAMEWORK

### After Week 6, You'll Know:

**Which approach is best?**
- Likely: Option B (best balance of improvement vs complexity)
- Alternative: Option C if speed is critical (similar results, fewer channels)

**How much improvement did we get?**
- Expected: +0.5-0.8% in R²
- This is meaningful: Difference between 0.960 and 0.969 is significant in publications

**Which models benefited most?**
- Probably: Complex models (Transformer, GNN AE) benefit more than simple ones
- Simple models may plateau at feature selection

**Are there seasonal differences?**
- Rainy season likely shows bigger AlphaEarth benefit
- Winter season shows smaller benefit (already stable)

**What should we recommend?**
- **For production:** Use Option B + Transformer model
- **For efficiency:** Use Option C (faster, similar results)
- **For publication:** Highlight Option B in main article, mention C in discussion

---

## CHECKPOINTS & VALIDATION

### Week 1 (End of Setup)
- ✅ AlphaEarth data extracted without errors
- ✅ All 4 options prepared and dimensions correct
- ✅ Training script runs without errors on small subset
- **Action:** If issues, fix before proceeding

### Week 2 (End of Baseline)
- ✅ All 9 models trained on baseline
- ✅ Metrics make sense (R² > 0.85 expected)
- ✅ Baseline comparison table created
- **Action:** If metrics are wrong, check data preparation

### Week 3-5 (During Options)
- ✅ Each option trained without errors
- ✅ Improvement % calculated correctly
- ✅ Feature importance analysis runs
- **Action:** Monitor training times, adjust parallelization if needed

### Week 6 (End of Synthesis)
- ✅ All results consolidated
- ✅ Statistical tests completed
- ✅ Article section written
- ✅ Figures publication-ready
- **Action:** Final review before article submission

---

## FALLBACK PLANS

### If Week 1 Overruns (Delayed Data):
- Start baseline training in parallel
- Download AlphaEarth while training baseline
- Reduces impact on timeline

### If Training Slower Than Expected:
- Reduce epochs (100 → 50) for exploratory runs
- Skip fold parallelization, run sequentially (slower but no memory issues)
- Reduce to top 5 models instead of 9
- Focus on Option B (most promising)

### If Week 2-5 Slips:
- Skip least-promising options (probably Option D)
- Focus on A, B, C (main comparison)
- Reduces scope from 4 to 3 approaches
- Still comprehensive, faster execution

### If Week 6 Compressed:
- Create basic tables, add visualizations later
- Submit article with tables, enhance with figures in revision
- Write summary now, expand discussion in review

---

## SUCCESS METRICS

### Quantitative
- [ ] 360 model runs completed (9 × 4 × 2 × 5)
- [ ] All metrics calculated and verified
- [ ] Statistical significance achieved (p < 0.05)
- [ ] Reproducibility confirmed (same random seed = same results)

### Qualitative
- [ ] Clear recommendation (Option A/B/C/D winner identified)
- [ ] Interpretable feature importance ranking
- [ ] Seasonal patterns explained
- [ ] Results integrated into article seamlessly

### Publication
- [ ] Tables are publication-ready
- [ ] Figures are high-quality and informative
- [ ] Extended results section is <5 pages
- [ ] Recommendations are defensible

---

## FINAL CHECKLIST BEFORE STARTING

- [ ] AlphaEarth account set up (Google Earth Engine)
- [ ] earthengine-api installed locally
- [ ] Current model code available and tested
- [ ] Data directory structure prepared
- [ ] Master script template created
- [ ] Results tracking setup ready
- [ ] Article structure identified (where does new section go?)
- [ ] Computational plan confirmed (parallel processing strategy set)
- [ ] Backup strategy in place (checkpoint saving)

---

## NEXT IMMEDIATE ACTIONS

### TODAY (30 minutes):
1. Review this quick start guide
2. Skim COMPARATIVE_ANALYSIS_PLAN.md (full details)
3. Decide: Ready to start Phase 1?

### TOMORROW (2-3 hours):
1. Set up AlphaEarth data extraction
2. Create data directory structure
3. Start downloading data
4. Create master script template

### THIS WEEK (1 week):
1. Complete Phase 1 (data extraction & preparation)
2. Have clean datasets ready for training
3. Test training script on small subset

### NEXT WEEK (Week 2 starts):
1. Begin baseline model training
2. Start feature importance analysis
3. Create initial results tracking

---

## CONTACT & SUPPORT

**Need clarification?** → Refer to COMPARATIVE_ANALYSIS_PLAN.md (full details)
**Code issues?** → Check Jupyter notebook templates in plan
**Timeline concerns?** → Review M1 optimization and fallback strategies
**Results interpretation?** → See "Expected Results Preview" above

---

## ONE-PAGE SUMMARY

| Item | Detail |
|------|--------|
| **What** | Compare 9 ensemble models with/without AlphaEarth (4 approaches) |
| **How** | 5-fold cross-validation across rainy & winter seasons |
| **Timeline** | 4-6 weeks (achievable on M1 Pro with optimization) |
| **Output** | Extended article section + comparison tables + figures |
| **Best Approach** | Option B (add AlphaEarth, expected +0.5-0.8% R²) |
| **Next Step** | Start Phase 1 (data extraction) |
| **Status** | Ready to implement ✅ |

---

**Document Created:** December 26, 2025
**Purpose:** Quick start guide for comparative analysis execution
**Status:** Ready to begin Phase 1 immediately

🚀 **You're ready to start!** Let's create publication-quality comparative analysis!

