# QUICK REFERENCE GUIDE

## PROJECT AT A GLANCE

| Aspect | Details |
|--------|---------|
| **Title** | Seasonal Source Apportionment of Dhaka river water and sediment heavy metals using ensemble deep learning |
| **Status** | ~80% complete - Article draft written, all models trained, needs final polish |
| **Publication** | Elsevier (elsarticle class) |
| **Target** | Journal article on heavy metal contamination & ensemble learning |
| **Duration** | 2 seasons (rainy & winter) |
| **Scale** | 5 rivers, 100-200 samples per season |

---

## FILES YOU'LL LIKELY NEED

### For Article Writing:
```
/draft/main.tex              ← Main structure, edit here
/draft/Methodology.tex       ← Methods (mostly complete)
/draft/results.tex           ← Results section structure
/draft/Algorithms2.tex       ← Model descriptions (very detailed)
/draft/ModelPerformanceTable.tex ← Model comparison table
/draft/main.pdf              ← Compiled output
```

### For Data Understanding:
```
/data/RainySeason.csv        ← Raw sediment/water data
/data/WinterSeason.csv       ← Winter season data
/gis/data/Hydro_LULC_Rainy.csv ← Features for ML
/Python/RI.csv               ← Risk Index (MODEL TARGET)
```

### For Model Results:
```
/gis/SedimentRainy/          ← All rainy season models
/gis/SedimentRainy/TopModels.ipynb ← Compare best models
/gis/SedimentRainy/FeatureImportance/ ← Why models work
/gis/SedimentRainy/metricsRainy.csv ← Performance metrics
```

### For Reference:
```
/claude_temp/PROJECT_STRUCTURE.md ← Comprehensive overview
/claude_temp/CLARIFICATION_QUESTIONS.md ← 30 questions to clarify
/claude_temp/DATA_FLOW.md ← How data moves through project
```

---

## KEY STATISTICS

### Best Model: Transformer CNN GNN MLP
| Metric | Rainy | Winter | Unit |
|--------|-------|--------|------|
| **R²** | 0.9604 | 0.9721 | Variance explained |
| **RMSE** | 15.74 | 7.99 | Risk Index units |
| **MAE** | 9.52 | 4.45 | Risk Index units |
| **SMAPE** | varies | varies | % error |

**Interpretation:** Model explains 96% of RI variance in rainy season, 97% in winter. 49% better RMSE in winter due to more stable conditions.

### Model Rankings (Rainy Season)
```
1. Transformer CNN GNN MLP    R²=0.9604 ⭐ USE THIS
2. GNN MLP AE                 R²=0.9581
3. CNN GNN MLP PG             R²=0.9570
4. GNN MLP                    R²=0.9519
5. CNN GAT MLP                R²=0.9266
... (4 more models below 0.93)
```

---

## QUICK TASK CHECKLIST

### What's DONE:
- ✅ Field sampling (rainy + winter)
- ✅ Laboratory analysis (metal measurements)
- ✅ GIS data prep (shapefiles, rasters, features)
- ✅ Risk index calculations (EF, Igeo, RI, etc.)
- ✅ ML training (9 models per season, 18 total)
- ✅ Model evaluation (all metrics computed)
- ✅ Feature importance analysis (LIME + permutation)
- ✅ Article draft (structure + most sections)
- ✅ Methodology section (detailed + complete)

### What NEEDS WORK:
- ❓ **Introduction section** - needs to be written
- ❓ **Discussion section** - needs to be written
- ❓ **Figures** - need to be created and embedded
  - Spatial RI distribution maps
  - Feature importance visualizations
  - Model comparison charts
  - Seasonal comparison plots
- ⚠️ **Results detail** - may need to expand certain subsections
- ⚠️ **References** - bibliography appears empty in main.tex
- ⚠️ **Confirmation** - need to verify article scope (all topics, or focus?)

---

## FOLDER NAVIGATION QUICK REFERENCE

```
five_rivers/
│
├── data/                          ← RAW DATA (sediment, water chemistry)
│   ├── RainySeason.csv           ← Start here for data understanding
│   ├── WinterSeason.csv
│   └── [+10 more data files]
│
├── gis/                           ← SPATIAL DATA & MODELS
│   ├── data/                     ← Training data for ML
│   │   ├── Samples_100.csv/.shp  ← Sample locations
│   │   ├── Hydro_LULC_*.csv      ← Features for MLP
│   │   └── gnn_data.npz          ← Graph data
│   │
│   ├── IDW/                      ← Interpolated metal rasters
│   │   ├── AsR_C.gpkg, CdR_C.gpkg, ... (9 files)
│   │   └── (Used as CNN input - raster patches)
│   │
│   ├── SedimentRainy/            ← ⭐ MAIN MODEL RESULTS
│   │   ├── Transformer CNN GNN MLP.ipynb  ← Best model
│   │   ├── TopModels.ipynb              ← Model comparison
│   │   ├── metricsRainy.csv             ← Performance
│   │   ├── Model1.keras                 ← Saved best model
│   │   ├── PredTest.csv                 ← Predictions
│   │   └── FeatureImportance/           ← Why it works
│   │       ├── t_permutation.csv        ← Feature ranks
│   │       ├── t_lime.csv               ← Local explanations
│   │       └── WinterRainy.png          ← Visualization
│   │
│   ├── SedimentWinter/           ← Winter season (parallel structure)
│   │   └── [similar structure to SedimentRainy]
│   │
│   └── [LULCMerged, CalIndices, ModelTrain, etc.]
│
├── Python/                        ← RISK ASSESSMENT ANALYSIS
│   ├── sample.ipynb              ← Main analysis notebook
│   ├── EF.csv                    ← Enrichment factor
│   ├── RI.csv                    ← Risk index results
│   ├── RI.xlsx                   ← Risk index (Excel)
│   ├── IgeoWinter.csv            ← Geoaccumulation index
│   └── [+10 more risk assessment outputs]
│
├── R/                            ← STATISTICAL ANALYSIS
│   ├── pca_factor.R              ← Source apportionment
│   ├── Factor_loadings_rainy.csv ← PCA results
│   └── pca_factor.nb.html        ← Report
│
├── draft/                        ← 📄 JOURNAL ARTICLE (LATEX)
│   ├── main.tex                  ← Edit here for structure
│   ├── Methodology.tex           ← Methods section
│   ├── results.tex               ← Results structure
│   ├── Algorithms2.tex           ← Model descriptions
│   ├── ModelPerformanceTable.tex ← Model comparison
│   ├── HeavyMetalDistribution.tex
│   ├── Igeo.tex, EF.tex, PLI.tex, etc.
│   └── main.pdf                  ← Compiled document
│
├── claude_temp/                  ← 🆘 DOCUMENTATION (FOR YOU)
│   ├── PROJECT_STRUCTURE.md      ← Comprehensive guide
│   ├── CLARIFICATION_QUESTIONS.md ← 30 questions
│   ├── DATA_FLOW.md              ← Data pipeline
│   └── QUICK_REFERENCE.md        ← This file
│
└── Papers/                       ← Reference papers
```

---

## COMMON QUESTIONS ANSWERED

### "What's the target variable?"
**Answer:** Risk Index (RI) calculated from metal concentrations using combined formula. See `/Python/RI.csv` for values.

### "Which model should I use?"
**Answer:** **Transformer CNN GNN MLP** - best R² on both seasons (0.9604 rainy, 0.9721 winter). File: `/gis/SedimentRainy/Transformer CNN GNN MLP.ipynb`

### "Where are the predictions?"
**Answer:** `/gis/SedimentRainy/PredTest.csv` - has y_true and y_pred columns

### "How do I understand why the model works?"
**Answer:** Two methods:
- **Permutation Feature Importance:** `/gis/SedimentRainy/FeatureImportance/t_permutation.csv`
- **LIME:** `/gis/SedimentRainy/FeatureImportance/t_lime.csv`

### "What are the heavy metals analyzed?"
**Answer:** Cr, Ni, Cu, As, Cd, Pb (6 metals total)

### "How many samples?"
**Answer:** Primary analysis: 100 samples per season (rainy & winter). Alternative: 200 samples. See `/gis/data/Samples_100.csv` and `Samples_200.csv`

### "What are the input data types?"
**Answer:** Three modalities:
1. **CNN:** Raster patches (spectral indices, IDW metals)
2. **MLP:** Tabular features (water quality, metal conc., coordinates)
3. **GNN:** Graph adjacency (distance-based spatial relationships)

### "Where's the source apportionment?"
**Answer:** PCA analysis in `/R/pca_factor.R` - identifies factor loadings. Results in `/R/Factor_loadings_*.csv`

### "What's missing from the article?"
**Answer:**
1. Introduction section (not started)
2. Discussion section (not started)
3. Figures (structure set up, visuals not embedded)
4. Bibliography (empty in main.tex)
5. Abstract (placeholder only)

---

## CRITICAL FILES FOR ARTICLE COMPLETION

### IF WRITING INTRODUCTION:
Reference these files:
- `/data/RainySeason.csv` - raw data to cite
- `/gis/data/Samples_100.shp` - sampling locations
- `/Python/sample.ipynb` - methodology details
- Consider: Why these 5 rivers? What's the problem?

### IF WRITING DISCUSSION:
Reference these files:
- `/gis/SedimentRainy/metricsRainy.csv` - model results
- `/gis/SedimentRainy/FeatureImportance/t_permutation.csv` - why models work
- `/R/Factor_loadings_rainy.csv` - what sources identified
- `/draft/Algorithms2.tex` - model descriptions (already written)

### IF CREATING FIGURES:
Need to generate:
- Spatial map: RI distribution (requires `/gis/SedimentRainy/PredTest.csv` + `/gis/data/Samples_100.csv`)
- Feature importance: Top 10 features (CSV to bar plot)
- Model comparison: R² ranking (table to chart)
- Seasonal: Rainy vs Winter side-by-side

---

## WORKFLOW RECOMMENDATIONS

### Option A: Complete Article Efficiently
```
1. Read CLARIFICATION_QUESTIONS.md → answer 5 critical ones
2. Write Introduction (1-2 pages)
3. Create 3-4 key figures
4. Expand Discussion section
5. Add references
6. Compile & polish
Time: ~3-5 days
```

### Option B: Deepen Analysis First
```
1. Verify all model hyperparameters
2. Perform cross-validation check
3. Create sensitivity analysis
4. Write detailed model justification
5. Then proceed with article writing
Time: ~1-2 weeks
```

### Option C: Prepare for Submission
```
1. Finalize all figures
2. Write Introduction + Discussion
3. Get feedback on interpretation
4. Select 2-3 best models to highlight
5. Create supplementary material
Time: ~1 week
```

---

## PYTHON/PANDAS QUICK ACCESS

### Load raw data:
```python
import pandas as pd
rainy = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/data/RainySeason.csv')
winter = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/data/WinterSeason.csv')
```

### Load model results:
```python
metrics = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainy/metricsRainy.csv')
predictions = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainy/PredTest.csv')
feature_importance = pd.read_csv('/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainy/FeatureImportance/t_permutation.csv')
```

### Load best model:
```python
from tensorflow import keras
model = keras.models.load_model('/Users/rakibhhridoy/Five_Rivers/gis/SedimentRainy/models/Model1.keras')
```

---

## NEXT STEPS FOR YOU

1. **Review** the 3 documentation files in `/claude_temp/`:
   - `PROJECT_STRUCTURE.md` - full understanding
   - `CLARIFICATION_QUESTIONS.md` - gaps needing clarification
   - `DATA_FLOW.md` - how data moves through project

2. **Answer** the 5 CRITICAL questions from CLARIFICATION_QUESTIONS.md:
   - Q1: Source apportionment method?
   - Q2: RI calculation formula?
   - Q6: Feature set specification?
   - Q21: Article structure (intro/discussion)?
   - Q26: What's your immediate need?

3. **Choose** your path:
   - Path A: Complete article writing
   - Path B: Deepen technical analysis
   - Path C: Prepare for journal submission

4. **Let me know** what you want to work on, and I can help with:
   - Writing sections (intro, discussion)
   - Creating figures & visualizations
   - Analyzing results deeper
   - Comparing with literature
   - Preparing for submission
   - etc.

---

**Created:** December 26, 2025
**Purpose:** Quick reference for navigating the Five Rivers project
**Size:** ~45,000 words of documentation
**Next:** Await your clarifications and next task!
