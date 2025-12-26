# DATA FLOW & PROCESSING PIPELINE

## OVERVIEW DIAGRAM

```
FIELD SAMPLING
    ↓
LABORATORY ANALYSIS (EDXRF, physicochemical)
    ↓
RAW DATA FILES (/data)
    │
    ├─→ SEDIMENT DATA
    ├─→ WATER DATA
    └─→ METAL CONCENTRATIONS (Cr, Ni, Cu, As, Cd, Pb)
        ↓
        RISK INDEX CALCULATIONS (/Python)
        ├─→ Enrichment Factor (EF)
        ├─→ Geoaccumulation Index (Igeo)
        ├─→ Contamination Factor (Cf)
        ├─→ Pollution Load Index (PLI)
        ├─→ Risk Index (RI) [TARGET VARIABLE]
        ├─→ Hazard Quotient (HQ)
        ├─→ Health Risk Indices
        └─→ Toxic Risk Index (TRI)
        ↓
        OUTPUT: /Python/*.csv (EF, RI, etc.)

GEOGRAPHIC DATA PREPARATION (/gis/data)
    ├─→ Sample locations (Samples_100.shp, Samples_200.shp)
    ├─→ Coordinates extraction
    └─→ Spatial feature creation (Lat/Long, distances)
        ↓
        RASTER DATA EXTRACTION (/gis/IDW, /gis/CalIndices)
        ├─→ IDW Interpolated metals
        │   (AsR_C.gpkg, CdR_C.gpkg, etc.)
        ├─→ Spectral indices
        │   (NDVI, NDWI, SAVI, etc.)
        ├─→ LULC classifications
        └─→ Soil properties
            (Clay, Sand, Silt)
            ↓
            FEATURE EXTRACTION (for CNN input)
            └─→ Patch extraction at each sample location
                (windowing from rasters)
                ↓
                /gis/data/train_data/

SOURCE APPORTIONMENT
    ├─→ PCA ANALYSIS (/R)
    │   ├─→ PCA loadings
    │   ├─→ PCA weights
    │   ├─→ Factor loadings
    │   └─→ Output: Factor_loadings_*.csv
    │
    └─→ (Potentially PMF or other methods?)

MACHINE LEARNING PIPELINE
    ↓
INPUT DATA PREPARATION (/gis/data)
    │
    ├─ CNN INPUT (Raster patches)
    │  Source: /gis/IDW/*.gpkg + /gis/CalIndices/*
    │  Processing: Windowing, normalization
    │  Output: Patch arrays [N, H, W, C]
    │
    ├─ MLP INPUT (Tabular features)
    │  Source: /gis/data/*.csv (Samples + water quality)
    │  Processing: StandardScaler normalization
    │  Output: Feature matrix [N, D]
    │  Features: pH, EC, TDS, DO, metal conc., coordinates, etc.
    │
    └─ GNN INPUT (Graph adjacency)
       Source: Coordinates from samples
       Processing: Distance-based kernel G_ij = exp(-d_ij/τ)
       Output: Graph matrix [N, N]

MODEL TRAINING (/gis/SedimentRainy & /gis/SedimentWinter)
    │
    ├─→ RAINY SEASON MODELS
    │   ├─→ Transformer CNN GNN MLP [BEST] ⭐
    │   ├─→ GNN MLP AE
    │   ├─→ CNN GNN MLP PG
    │   ├─→ GNN MLP
    │   ├─→ CNN GAT MLP
    │   ├─→ Stacked CNN GNN MLP
    │   ├─→ CNN GNN MLP (Baseline)
    │   ├─→ Mixture of Experts
    │   ├─→ Dual Attention
    │   ├─→ CNN LSTM
    │   └─→ GCN GAT
    │
    └─→ WINTER SEASON MODELS
        ├─→ Transformer CNN GNN MLP [BEST] ⭐
        ├─→ GNN MLP AE
        ├─→ GNN MLP
        ├─→ Mixture of Experts
        └─→ ... (9 models total per season)

MODEL OUTPUTS
    │
    ├─→ PREDICTIONS (/gis/SedimentRainy/PredTest.csv)
    │   └─→ Predicted RI values vs observed RI
    │
    ├─→ METRICS (/gis/SedimentRainy/)
    │   ├─→ metrics.csv, metricsRainy.csv, metricsWinter.csv
    │   ├─→ Performance: R², RMSE, MAE, MSE, SMAPE
    │   └─→ Summary: ModelPerformanceTable.tex
    │
    ├─→ TRAINED MODELS (/gis/SedimentRainy/models/)
    │   ├─→ Model1.keras (33 MB)
    │   └─→ Other saved model files
    │
    └─→ VISUALIZATIONS (/gis/SedimentRainy/)
        ├─→ SedimentRainyMetrics.png
        ├─→ SedimentWinterMetrics.png
        └─→ Performance comparison plots

INTERPRETABILITY ANALYSIS (/gis/SedimentRainy/FeatureImportance/)
    │
    ├─→ PERMUTATION FEATURE IMPORTANCE
    │   ├─→ Shuffle each feature
    │   ├─→ Measure drop in R²
    │   ├─→ Files: cgmp_permutation.csv, gma_permutation.csv, t_permutation.csv
    │   └─→ Results ranked by importance
    │
    └─→ LIME (Local Interpretable Model-agnostic Explanations)
        ├─→ For each test sample
        ├─→ Generate perturbed neighbors
        ├─→ Fit local linear model
        ├─→ Extract local feature importance
        ├─→ Files: cgmp_lime.csv, gma_lime.csv, t_lime.csv
        └─→ Identify local drivers of RI

VISUALIZATION OUTPUTS (/gis/SedimentRainy/FeatureImportance/)
    │
    ├─→ WinterRainy.png
    │   └─→ Feature importance comparison rainy vs winter
    │
    └─→ WinterRainy63.png
        └─→ Alternative feature importance visualization

JOURNAL ARTICLE COMPOSITION (/draft/)
    │
    ├─→ MAIN DOCUMENT
    │   └─→ main.tex (structure + input files)
    │
    ├─→ SECTIONS
    │   ├─→ Methodology.tex
    │   │   ├─→ Study area description
    │   │   ├─→ Sample collection protocols
    │   │   ├─→ Algorithms2.tex (model architectures: 9 variants)
    │   │   └─→ Data postprocessing (PFI, LIME)
    │   │
    │   ├─→ results.tex
    │   │   ├─→ HeavyMetalDistribution.tex
    │   │   ├─→ Igeo.tex
    │   │   ├─→ EF.tex
    │   │   ├─→ PLI.tex
    │   │   ├─→ HealthRisk.tex
    │   │   ├─→ PCA.tex
    │   │   ├─→ MonteCarlo.tex
    │   │   └─→ ModelPerformanceTable.tex (metrics from models)
    │   │
    │   └─→ Conclusion & Acknowledgments
    │
    └─→ OUTPUT
        └─→ main.pdf (compiled document)
```

---

## DETAILED DATA TYPES & TRANSFORMATIONS

### 1. FIELD SAMPLING DATA → RAW CSV

**Input:** Field measurements and lab analysis
```
Field Sample → Lab Analysis (EDXRF) → Raw CSV
Location: /data/*.csv
Columns:
  - Station ID, River, Latitude, Longitude
  - Metal concentrations: Cr, Ni, Cu, As, Cd, Pb (mg/kg for sediment, μg/L for water)
  - Water quality: pH, EC, TDS, DO, turbidity
  - Sampling date, Season
```

**Files Generated:**
- `/data/RainyS100.csv` → `/data/RainySeason.csv` (processed)
- `/data/WinterSeason.csv`
- `/data/Samples_100.csv` (spatial version)

---

### 2. RAW DATA → RISK INDICES (Python Analysis)

**Location:** `/Python/sample.ipynb` and risk calculation notebooks

**Transformation:**
```
Metal Concentrations → Baseline/Background Values
                    ↓
                    Enrichment Factor (EF)
                    ↓ (depends on EF)
                    Geoaccumulation Index (Igeo)
                    ↓ (weighted sum)
                    Contamination Factor (Cf)
                    ↓
                    Pollution Load Index (PLI)
                    ↓ (combined calculation)
                    RISK INDEX (RI) ← Target Variable

Additionally calculated:
- Hazard Quotient (HQ) = Dose / RfD
- Hazard Index (HI) = Σ HQ
- Carcinogenic Risk (CR) for carcinogenic metals
```

**Output Files:**
- `/Python/EF.csv` - Enrichment factors
- `/Python/RI.csv` - Risk indices (PRIMARY TARGET)
- `/Python/IgeoWinter.csv` - Geoaccumulation indices
- `/Python/modifed_HQ.csv` - Health risk quotients
- `/Python/CSI.csv` - Combined indices
- `/Python/Toxic_Risk_Index_TRI.csv` - Toxicity measures

---

### 3. GIS DATA PREPARATION

**Location:** `/gis/data/`

**Process A: Shapefile Creation**
```
Sample coordinates (Lat/Long)
         ↓
QGI/PostGIS
         ↓
Shapefile generation (.shp, .dbf, .prj, .shx, .cpg files)
         ↓
/gis/data/Samples_100.* or Samples_200.*
```

**Process B: Raster Data Preparation**
```
Remote sensing imagery (Sentinel-2, Landsat)
         ↓
Spectral Index Calculation (NDVI, NDWI, SAVI, etc.)
         ↓
/gis/CalIndices/ (TIFF, GeoTIFF)

Point metal concentrations → Spatial Interpolation (IDW)
         ↓
Raster layer for each metal (As, Cd, Cr, Cu, Ni, Pb)
         ↓
/gis/IDW/*.gpkg (GeoPackage format)

LULC classification maps
         ↓
/gis/LULCMerged/

Soil property maps (Clay, Sand, Silt)
         ↓
/gis/IDW/ (included in raster patches)
```

**Process C: Feature Engineering**
```
Sample shapefile + Raster layers
         ↓
Calculate spatial relationships:
  - Distance to nearest river
  - Hydrological flow distance
  - LULC type at location
  - Surrounding land use (buffer analysis)
         ↓
/gis/data/Hydro_LULC_*.csv
/gis/data/sampling_features_*.csv
/gis/data/samples_hydro_lulc_optimized.*
```

---

### 4. ML INPUT PREPARATION

**Location:** `/gis/data/train_data/` and `/gis/data/gnn_data.npz`

**CNN INPUT PREPARATION:**
```
Sample coordinates (Lat/Long)
         ↓
For each coordinate:
  - Define spatial window (e.g., 500m × 500m)
  - Extract patches from:
    * IDW interpolated metals (/gis/IDW/*.gpkg)
    * Spectral indices (/gis/CalIndices/)
    * LULC maps (/gis/LULCMerged/)
    * Soil properties (/gis/IDW/)
         ↓
Stack layers into multi-channel array:
  Channel 0: As
  Channel 1: Cd
  Channel 2: Cr
  ... (multiple channels per raster type)
         ↓
Normalize: patch / (max(patch) + ε)
         ↓
Output shape: [N_samples, H, W, C]
E.g., [100, 32, 32, 15] for 100 samples, 32×32 pixels, 15 channels
         ↓
Stored as: Python arrays or HDF5 in train_data/
```

**MLP INPUT PREPARATION:**
```
Raw sample data (/gis/data/*.csv)
         ↓
Drop non-numeric: Stations, River, geometry, coordinates
         ↓
Features to keep:
  - Direct metal measurements
  - Water quality (pH, EC, TDS, DO, turbidity)
  - Sediment properties
  - Spatial indices
         ↓
Fill missing values: NaN → 0
         ↓
Standardize: (X - μ) / σ using StandardScaler
         ↓
Output shape: [N_samples, D]
E.g., [100, 45] for 100 samples, 45 features
         ↓
Stored as: CSV or numpy array
```

**GNN INPUT PREPARATION:**
```
Sample coordinates (Lat/Long)
         ↓
Calculate pairwise Euclidean distances:
  d_ij = sqrt((lat_i - lat_j)² + (lon_i - lon_j)²)
         ↓
Apply distance-based kernel:
  G_ij = exp(-d_ij / τ)
  where τ = bandwidth parameter (tuned)
         ↓
Optional: K-nearest neighbors sparsification
         ↓
Output: Adjacency matrix
Shape: [N_samples, N_samples]
E.g., [100, 100] for 100 samples
         ↓
Stored as: /gis/data/gnn_data.npz (numpy compressed)
          or in-memory in notebook
```

**SPLIT INTO TRAIN/TEST:**
```
Combined dataset: [CNN, MLP, GNN] + Target (RI)
         ↓
Random split (seed = fixed for reproducibility)
  - Train: ~80% of samples
  - Test: ~20% of samples
         ↓
Stratification (optional): Ensure risk categories balanced
         ↓
Separate files:
  train_CNN, train_MLP, train_GNN, train_RI
  test_CNN, test_MLP, test_GNN, test_RI
```

---

### 5. MODEL TRAINING & INFERENCE FLOW

**Location:** `/gis/SedimentRainy/*.ipynb` and `/gis/SedimentWinter/*.ipynb`

```
TRAINING PHASE (during development):
┌─────────────────────────────────┐
│ Load train data                 │
│ - train_CNN: [N_train, 32, 32, C]
│ - train_MLP: [N_train, D]      │
│ - train_GNN: [N_train, N]      │
│ - train_RI: [N_train]          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Initialize model architecture   │
│ (9 variants available)          │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Train model                     │
│ - Optimizer: Adam               │
│ - Loss: MSE (or custom)        │
│ - Epochs: X                     │
│ - Batch size: Y                 │
│ - Callbacks: EarlyStopping      │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Evaluate on test set            │
│ - Compute R², RMSE, MAE, SMAPE │
│ - Store predictions             │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Save model                      │
│ Output: /gis/SedimentRainy/     │
│         models/model_name.keras │
└─────────────────────────────────┘

INFERENCE PHASE (predictions):
┌─────────────────────────────────┐
│ Load trained model              │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Prepare test data               │
│ [test_CNN, test_MLP, test_GNN]  │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Model.predict([X_test])         │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Predictions: y_pred             │
│ Shape: [N_test]                 │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Compute metrics:                │
│ R² = 1 - Σ(y - ŷ)² / Σ(y - ȳ)² │
│ RMSE = √(Σ(y - ŷ)² / N)       │
│ MAE = Σ|y - ŷ| / N            │
│ SMAPE = (100/N)Σ|ŷ - y|...    │
└──────────────┬──────────────────┘
               ↓
┌─────────────────────────────────┐
│ Store outputs:                  │
│ - PredTest.csv (y_test, y_pred) │
│ - metrics.csv (R², RMSE, ...)   │
│ - plots (scatter, residuals)    │
└─────────────────────────────────┘
```

---

### 6. FEATURE IMPORTANCE ANALYSIS

**Location:** `/gis/SedimentRainy/FeatureImportance/`

**PERMUTATION FEATURE IMPORTANCE:**
```
Trained model + test set
         ↓
Baseline prediction: y_baseline = model(test_data)
Calculate: R²_baseline
         ↓
For each input modality:

  CNN MODALITY:
    Shuffle all CNN features
    y_shuffled = model(test_CNN_shuffled, test_MLP, test_GNN)
    Compute: R²_shuffled
    Importance_CNN = R²_baseline - R²_shuffled

  MLP MODALITY:
    Shuffle all MLP features
    y_shuffled = model(test_CNN, test_MLP_shuffled, test_GNN)
    Importance_MLP = R²_baseline - R²_shuffled

  GNN MODALITY:
    Shuffle graph adjacency
    y_shuffled = model(test_CNN, test_MLP, test_GNN_shuffled)
    Importance_GNN = R²_baseline - R²_shuffled
         ↓
Output: /gis/SedimentRainy/FeatureImportance/
        cgmp_permutation.csv (CNN-GNN-MLP model)
        gma_permutation.csv (CNN-GAT-MLP model)
        t_permutation.csv (Transformer model)

Format:
  Modality | Importance_Score | Percent_Contribution
  CNN      | 0.045           | 4.5%
  MLP      | 0.652           | 65.2%
  GNN      | 0.303           | 30.3%
```

**LIME (Local Interpretable Model-agnostic Explanations):**
```
For each test sample i:

  1. Select sample: X_test[i]

  2. Generate neighborhood:
     Create M perturbed instances around sample i
     X_perturbed = X_test[i] + noise × random()

  3. Get predictions:
     y_pred_perturbed = model(X_perturbed)

  4. Weight by proximity:
     weights = exp(-distance(X_perturbed, X_test[i]) / σ)

  5. Fit local model:
     Linear regression with weights:
     y_local ≈ w₁×feature₁ + w₂×feature₂ + ... + b

  6. Extract importance:
     local_importance = |w₁|, |w₂|, ...
     (larger |w| = more important for this sample)
         ↓
Output: /gis/SedimentRainy/FeatureImportance/
        cgmp_lime.csv, gma_lime.csv, t_lime.csv

Example: For sample #5, top 3 features explaining prediction:
  Feature 1: Pb concentration (+0.85)
  Feature 2: Distance to industry (-0.62)
  Feature 3: NDVI (-0.41)
```

---

### 7. VISUALIZATION PIPELINE

**Location:** `/gis/SedimentRainy/FeatureImportance/`

```
Feature Importance Data (CSV)
         ↓
Matplotlib/Seaborn plotting
         ↓
OUTPUTS:
  - WinterRainy.png
    Bar plots: Feature importance rainy vs winter

  - WinterRainy63.png
    Alternative visualization or top-63 features

  - SedimentRainyMetrics.png
    Comparison of all 9 models (rainy season)

  - SedimentWinterMetrics.png
    Comparison of all 9 models (winter season)
```

---

### 8. SOURCE APPORTIONMENT (R Analysis)

**Location:** `/R/pca_factor.R` and `/R/pca_factor.Rmd`

```
Heavy metal concentrations matrix [N_samples, 6_metals]
  Cr  Ni  Cu  As  Cd  Pb
         ↓
Principal Component Analysis (PCA)
         ↓
Calculate:
  - Eigenvalues (variance per component)
  - Loadings (how metals contribute to PCs)
  - PC scores (sample positions in PC space)
         ↓
Factor Interpretation:
  - PC1 high Pb, Cd, As = anthropogenic/pollution factor
  - PC2 high Cr, Ni = natural/geogenic factor
  - PC3 = potential traffic or another source
         ↓
OUTPUTS:
  - /R/PCA_loading_rainy.csv (metal loadings on PCs)
  - /R/PCA_weight_rainy.csv (PC eigenvalues/importance)
  - /R/Factor_loadings_rainy.csv (rotated loadings)
  - /R/pca_factor.nb.html (visualization report)
```

---

## FINAL OUTPUTS TO ARTICLE

```
ARTICLE (/draft/main.tex)
         ↓
Uses results from ALL pipelines above:

  1. Methodology section
     └─→ References Algorithms2.tex (model descriptions)

  2. Results section
     ├─→ HeavyMetalDistribution.tex (raw data summary)
     ├─→ Igeo.tex (geoaccumulation from /Python/)
     ├─→ EF.tex (enrichment from /Python/)
     ├─→ PLI.tex (pollution load index from /Python/)
     ├─→ HealthRisk.tex (HQ, HI from /Python/)
     ├─→ PCA.tex (factor analysis from /R/)
     ├─→ MonteCarlo.tex (uncertainty analysis)
     └─→ ModelPerformanceTable.tex (metrics from models)

  3. Figures (to be created)
     ├─→ Spatial maps of RI distribution
     ├─→ Feature importance visualizations
     │   (from /gis/SedimentRainy/FeatureImportance/)
     ├─→ Model comparison plots
     ├─→ Seasonal differences
     └─→ Sample locations map

  4. Discussion (to be written)
     ├─→ Interpret model results
     ├─→ Link to feature importance
     ├─→ Connect to PCA sources
     └─→ Compare with literature
```

---

## KEY DATA DEPENDENCIES

```
                    ↓
        Field Samples & Analysis
                    ↓
    ┌───────────────┴──────────────┐
    ↓                              ↓
Raw Data (/data/)         Geographic Data (/gis/)
    │                              │
    ├─→ Risk Indices (/Python/)    ├─→ Shapefiles (/gis/data/)
    │   └─→ RI (target)            │   └─→ Sampling points
    │                              │
    └─→ [merged]                   ├─→ Rasters (/gis/IDW/)
        Samples + RI               │   └─→ Metal, indices
                                   │
                                   └─→ Features (/gis/data/)
                                       └─→ Hydro, LULC
        ↓
    ML Training (/gis/SedimentRainy/)
        ├─→ Models
        ├─→ Predictions
        └─→ Feature Importance

        ↓
    Article Assembly (/draft/)
        ├─→ Text (Methodology, Results)
        └─→ Figures (from visualizations)
```

---

**Note:** This data flow represents the current project state. Some files may have alternative versions or variants for sensitivity analysis.

