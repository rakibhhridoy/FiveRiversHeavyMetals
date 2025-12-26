# Five Rivers Heavy Metal Source Apportionment Project
## Comprehensive Project Understanding Document

### PROJECT OVERVIEW
**Full Title:** Seasonal Source Apportionment of Dhaka river water and sediment heavy metal using novel graph based ensemble architecture of deep learning

**Target Journal:** Elsevier (elsarticle class with preprint format)

**Authors:**
- Md Rakib Hasan (Corresponding)
- Arafat Rahman
- Shoumik Zubyer
- Yeasmin Nahar Jolly

**Affiliations:**
- Department of Soil, Water And Environment, University of Dhaka, Bangladesh
- Atomic Energy Center (Atmospheric And Environment Lab), Dhaka, Bangladesh

---

## 1. RESEARCH OBJECTIVE
Analyze seasonal source apportionment of heavy metals in water and sediment from five major rivers around Dhaka, Bangladesh using advanced ensemble deep learning models with graph neural networks.

### Key Study Areas (5 Rivers):
1. Buriganga
2. Shitalakshya
3. Turag
4. Dhaleshwari
5. Balu

### Heavy Metals of Interest:
- Chromium (Cr)
- Nickel (Ni)
- Copper (Cu)
- Arsenic (As)
- Cadmium (Cd)
- Lead (Pb)

### Seasonal Coverage:
- **Winter Season:** November-February
- **Rainy Season:** June-September

---

## 2. DIRECTORY STRUCTURE & FILE ORGANIZATION

### 2.1 ROOT LEVEL DIRECTORIES

```
/Users/rakibhhridoy/Five_Rivers/
├── data/                      # Raw and processed data files
├── gis/                       # Geographic data, models, outputs
├── Python/                    # Python analysis scripts & notebooks
├── R/                         # R statistical analysis
├── draft/                     # LaTeX journal article files
├── Papers/                    # Reference papers
├── EnhancedFiveRivers.ipynb  # Main notebook
└── sample.ipynb              # Sample analysis
```

### 2.2 DATA DIRECTORY (/data)
**Purpose:** Contains sediment and water quality data

**Key Files:**
```
/data/
├── FiveRiverSedimentProperties.csv     # Sediment characteristics
├── Main.csv                            # Main dataset
├── RainyS100.csv                       # Rainy season (100 samples)
├── RainySeason.csv                     # Processed rainy season
├── WinterSeason.csv                    # Processed winter season
├── WinterSeason1.csv                   # Winter season variant
├── RISediment - Rainy.csv              # Risk Index for sediment (rainy)
├── RISediment - Winter.csv             # Risk Index for sediment (winter)
├── Sediments Data.docx                 # Raw data documentation
├── Shitalakshya.xlsx                   # Shitalakshya river specific data
├── data.csv                            # Base dataset
└── train_data/                         # Training datasets for models
```

**Data Content Types:**
- Heavy metal concentrations (Cr, Ni, Cu, As, Cd, Pb)
- Physicochemical parameters (pH, turbidity, TDS, EC, DO)
- Spatial coordinates (Latitude, Longitude)
- Risk Index (RI) - target variable
- Sediment properties

---

### 2.3 GIS DIRECTORY (/gis)
**Purpose:** Geographic information system data, spatial analysis, model training

#### 2.3.1 Subdirectories

**gis/data/** - Spatial datasets for machine learning
```
/gis/data/
├── Samples_100.* (shp, dbf, prj, cpg, csv)    # 100 sampling points shapefile
├── Samples_200.* (shp, dbf, prj, cpg, csv)    # 200 sampling points shapefile
├── Samples_100W.csv                            # 100 winter samples
├── Samples_100WW.*                             # 100 winter samples shapefile
├── Samples_200test.csv                         # 200 test samples
├── river_200_samples_rainy.csv                 # 200 rainy samples
├── Hydro_LULC_Combine.csv                      # Combined hydrological + LULC features
├── Hydro_LULC_Rainy.csv                        # Hydrological + LULC rainy season
├── Hydro_LULC_Winter.csv                       # Hydrological + LULC winter season
├── LULC_5km_Variations.csv                     # Land Use Land Cover variations
├── LULC_Transition_Matrix.csv                  # LULC transitions
├── hydrologicalF.csv                           # Hydrological features
├── merged_df.csv                               # Merged dataset
├── samples_hydro_lulc_optimized.*              # Optimized sampling with hydro/LULC
├── sampling_features_with_hydro_lulc.*         # Sampling features combined
├── sampling_features_hydro_lulc.* (shp files) # Shapefile versions
├── gnn_data.npz                                # Pre-processed GNN input data
└── train_data/                                 # Prepared training datasets
```

**Feature Types Included:**
- Hydrological features (distance to rivers, flow properties)
- LULC features (land use/land cover classifications)
- Combined environmental indices
- Graph-based spatial relationships

**gis/SedimentRainy/** - Rainy season model training & analysis
```
/gis/SedimentRainy/
├── CNN GAT MLP.ipynb               # CNN + Graph Attention Network + MLP
├── CNN GNN MLP.ipynb               # Baseline CNN-GNN-MLP ensemble
├── CNN GNN MLP PG.ipynb            # CNN-GNN-MLP with PMF-GWR
├── CNN LSTM.ipynb                  # CNN with LSTM temporal
├── Dual Attention.ipynb            # Dual attention mechanism variant
├── GCN GAT.ipynb                   # Graph Convolutional Network + GAT
├── GNN MLP AE.ipynb                # GNN-MLP with autoencoder
├── GNN MLP.ipynb                   # Pure GNN-MLP variant
├── Mixture of Experts.ipynb        # Mixture of experts ensemble
├── Stacked CNN GNN MLP.ipynb       # Stacked architecture variant
├── Transformer CNN GNN MLP.ipynb   # BEST MODEL - Transformer fusion (R²=0.9604)
├── ModelTrain.ipynb                # Primary training notebook
├── TopModels.ipynb                 # Comparison of top models
├── Model1.keras                    # Pre-trained model file (33 MB)
├── PredTest.csv                    # Model predictions on test set
├── metrics.csv                     # Performance metrics
├── metricsRainy.csv                # Rainy season metrics
├── metricsWinter.csv               # Winter season metrics
├── SedimentRainyMetrics.png        # Visualization of rainy metrics
├── SedimentWinterMetrics.png       # Visualization of winter metrics
├── TransformerFI.csv               # Feature importance for Transformer
├── TransformerFIB.csv              # Feature importance variant
├── FeatureImportance/              # Feature importance analysis outputs
│   ├── WinterRainy.png             # Combined importance visualization
│   ├── WinterRainy63.png           # Alternative visualization
│   ├── cgmp_lime.csv               # LIME explanations (CNN-GNN-MLP)
│   ├── cgmp_permutation.csv        # Permutation importance (CNN-GNN-MLP)
│   ├── gma_lime.csv                # LIME explanations (GAT)
│   ├── gma_permutation.csv         # Permutation importance (GAT)
│   ├── t_lime.csv                  # LIME explanations (Transformer)
│   └── t_permutation.csv           # Permutation importance (Transformer)
└── models/                         # Saved model files
```

**gis/SedimentWinter/** - Winter season model training & analysis (parallel structure)
```
/gis/SedimentWinter/
├── [Similar structure to SedimentRainy]
├── FeaturesImportance/             # Feature importance outputs
├── TruePred/                       # True vs predicted visualizations
└── [11 model notebooks]
```

**gis/IDW/** - Inverse Distance Weighting spatial interpolation
```
/gis/IDW/
├── AsR_C.gpkg                      # Arsenic raster (GeoPackage)
├── CdR_C.gpkg                      # Cadmium raster
├── ClayR_C.gpkg                    # Clay content raster
├── CrR_C.gpkg                      # Chromium raster
├── CuR_C.gpkg                      # Copper raster
├── NiR_C.gpkg                      # Nickel raster
├── PbR_C.gpkg                      # Lead raster
├── SandR_C.gpkg                    # Sand content raster
├── SiltR_C.gpkg                    # Silt content raster
├── [Associated .gpkg-shm and .gpkg-wal files for each]
```
*These are spatial raster files used as CNN input (patches extracted at sample locations)*

**gis/IDWW/** - Alternative IDW interpolation outputs

**gis/LULCMerged/** - Merged Land Use Land Cover data

**gis/CalIndices/** - Calculated remote sensing indices
- Spectral vegetation and water indices used as CNN input

**gis/ModelTrain/** - Alternative model training directory
- Contains model architectures and data exports

**gis/Pallete/** - QGIS color palettes
- SAVIC, EVIC, NDVI, NDWI, NDSIC color schemes for visualization

**gis/qgis_style_idw/** - QGIS styling for IDW outputs

**gis/output_idw/** - IDW output products

---

### 2.4 PYTHON DIRECTORY (/Python)
**Purpose:** Python-based analysis and risk assessment

**Key Notebooks:**
```
/Python/
├── CarcinogenicHealthRisk.ipynb    # Carcinogenic health risk assessment
├── CarcinogenEffect.ipynb          # Carcinogenic effect calculation
├── HQ_HI.ipynb                     # Hazard Quotient & Hazard Index
├── sample.ipynb                    # Main analysis notebook (835 KB)
├── [EF/ directory]                 # Enrichment Factor analysis
├── [Figure/ directory]             # Generated figures
├── [data/ directory]               # Python-specific data
├── [doc/ directory]                # Documentation
├── EF.csv / EF.xlsx                # Enrichment Factor results
├── IgeoWinter.csv                  # Igeo index winter
├── RI.csv / RI.xlsx                # Risk Index calculations
├── Toxic_Risk_Index_TRI.csv        # Toxic Risk Index
├── cf_ModifiedCD.csv               # Modified contamination factor
├── mERMQ.csv                       # Ecological Risk quotient
├── CSI.csv / CSI_mERMQ.csv         # Combined indices
├── modifed_HQ.csv                  # Modified hazard quotient
├── rainy.csv / winter.csv          # Seasonal processed data
```

**Analysis Types:**
- Enrichment Factor (EF)
- Igeo Index (Geoaccumulation Index)
- Contamination Factor (Cf)
- Risk Index (RI)
- Hazard Quotient (HQ) / Hazard Index (HI)
- Carcinogenic/Non-carcinogenic health risk
- Toxic Risk Index (TRI)

---

### 2.5 R DIRECTORY (/R)
**Purpose:** Statistical analysis and PCA

**Key Files:**
```
/R/
├── pca_factor.R              # PCA analysis script
├── pca_factor.Rmd            # R Markdown report
├── pca_factor.nb.html        # HTML output
├── PCA_loading_rainy.csv     # PCA loadings (rainy)
├── PCA_loading_Winter.csv    # PCA loadings (winter)
├── PCA_weight_rainy.csv      # PCA weights (rainy)
├── PCA_weight_Winter.csv     # PCA weights (winter)
├── Factor_loadings_rainy.csv # Factor loadings (rainy)
└── Factor_loadings_Winter.csv # Factor loadings (winter)
```

---

### 2.6 DRAFT DIRECTORY (/draft) - LATEX ARTICLE
**Purpose:** Journal article manuscript in LaTeX format

**Main Files:**
```
/draft/
├── main.tex                      # Master document
├── main.pdf                      # Compiled PDF
├── Methodology.tex               # Methods section
├── results.tex                   # Results & Discussion
├── Algorithms2.tex               # Detailed algorithm descriptions
├── Algorithms1.tex               # Alternative algorithms
├── Algorithms.tex                # Base algorithm definitions
├── HeavyMetalDistribution.tex    # Heavy metal results subsection
├── Igeo.tex                      # Igeo index results
├── EF.tex                        # Enrichment Factor results
├── PLI.tex                       # Pollution Load Index results
├── HealthRisk.tex                # Health risk assessment results
├── PCA.tex                       # PCA analysis results
├── MonteCarlo.tex                # Monte Carlo simulation results
├── ModelPerformanceTable.tex     # Model comparison table
```

**Article Structure:**
1. Introduction
2. Methodology
   - Study Area & Sample Collection
   - Sample Preparation & Analysis
   - Algorithm descriptions (9 ensemble models)
   - Data Post Processing (Feature Importance, LIME)
3. Results and Discussion
   - Heavy Metal Distribution
   - Igeo Index
   - Enrichment Factor (EF)
   - Pollution Load Index (PLI)
   - Health Risk Assessment
   - PCA/Source Apportionment
   - Monte Carlo Analysis
   - Model Performance Comparison
4. Conclusion
5. Acknowledgments

---

## 3. MACHINE LEARNING MODELS

### 3.1 MODEL ARCHITECTURES (9 Ensemble Variants)

#### Best Performing Model:
**Transformer CNN GNN MLP**
- Rainy Season: R² = 0.9604, RMSE = 15.7421
- Winter Season: R² = 0.9721, RMSE = 7.9921 (49.38% improvement)
- Uses multi-head attention for CNN-MLP-GNN fusion

#### Other Models Ranked by Performance (Rainy Season):
1. GNN MLP AE (Autoencoder variant) - R² = 0.9581
2. CNN GNN MLP PG (with PMF-GWR) - R² = 0.9570
3. GNN MLP (Graph-only) - R² = 0.9519
4. CNN GAT MLP (Graph Attention) - R² = 0.9266
5. Stacked CNN GNN MLP - R² = 0.9240
6. CNN GNN MLP (Baseline) - R² = 0.9089
7. Mixture of Experts - R² = 0.9070
8. Dual Attention - R² = 0.8608

### 3.2 INPUT DATA MODALITIES

**Three Complementary Data Streams:**

1. **CNN Input (Raster Patches)**
   - Extracted from spatial raster layers at each sampling location
   - Includes:
     - Spectral indices (NDVI, NDWI, SAVI, NDBI, etc.)
     - IDW interpolated heavy metal concentrations (As, Cd, Cr, Cu, Ni, Pb)
     - Land Use Land Cover (LULC) classifications
     - Soil properties (Clay, Sand, Silt)
   - Processing: Windowing technique via rasterio, normalized to avoid division by zero
   - Architecture: 2x Conv2D + MaxPooling + Flatten + Dense

2. **MLP Input (Tabular Features)**
   - Standardized numeric features from sampling data
   - Includes:
     - Water quality parameters (pH, EC, TDS, turbidity, DO)
     - Metal concentrations (direct measurements)
     - Spatial coordinates (Latitude, Longitude)
     - Sediment properties
   - Processing: StandardScaler normalization
   - Architecture: Dense layers with ReLU activation

3. **GNN Input (Graph-based Spatial Adjacency)**
   - Spatial relationships between sampling sites
   - Constructed via distance-based kernel: G_ij = exp(-d_ij/τ)
   - Where d_ij = Euclidean distance between sites i and j
   - τ = kernel bandwidth parameter
   - Encodes spatial autocorrelation and neighborhood effects
   - Architecture: Graph convolution or attention mechanisms

### 3.3 FUSION MECHANISMS

**Concatenation-based:** Simple concatenation of CNN, MLP, GNN outputs

**Transformer-based (Best):**
- Multi-head attention over concatenated outputs
- 3 sequence elements (CNN, MLP, GNN features)
- Query, Key, Value projections
- Layer normalization & residual connections

**Mixture of Experts:**
- Gating network learns weight for each expert output
- Softmax normalization of gate weights

**Autoencoder-based:**
- GNN & MLP outputs form latent representation
- Decoder reconstructs target variable

### 3.4 INTERPRETABILITY METHODS

**Permutation Feature Importance (PFI):**
- Measures importance by shuffling each feature
- Computes drop in R² when feature is permuted
- Applied to CNN, MLP, and GNN modalities separately
- Results in feature importance rankings

**LIME (Local Interpretable Model-agnostic Explanations):**
- Approximates black-box model with linear surrogate
- For each prediction instance:
  - Generate perturbed instances around it
  - Weight by proximity to original
  - Fit local linear model
  - Extract local feature importance from coefficients
- Provides sample-level explanations

---

## 4. EVALUATION METRICS

### 4.1 Regression Performance Metrics:
- **R² (Coefficient of Determination):** Variance explained
- **RMSE (Root Mean Square Error):** Absolute prediction deviation
- **MAE (Mean Absolute Error):** Robust to outliers
- **MSE (Mean Square Error):** Raw squared error
- **SMAPE (Symmetric Mean Absolute Percentage Error):** Normalized to magnitude

### 4.2 Model Comparison:
- Cross-season comparison (Rainy vs Winter)
- Rainy season: More challenging (higher flows, sediment redistribution)
- Winter season: More stable (reduced flows, predictable factors)

---

## 5. RISK ASSESSMENT INDICES

**Calculated from metal concentrations:**

1. **Geoaccumulation Index (Igeo)**
   - Measures degree of contamination relative to background

2. **Enrichment Factor (EF)**
   - Quantifies metal enrichment from anthropogenic sources
   - Categories: Minimal (<2), Minor (2-5), Moderate (5-20), High (20-40), Very High (>40)

3. **Contamination Factor (Cf)**
   - Ratio of measured to background concentrations

4. **Pollution Load Index (PLI)**
   - Cumulative multi-metal contamination measure

5. **Risk Index (RI)**
   - Target variable for model predictions
   - Incorporates multiple contamination indices
   - Quantifies ecological/health risk

6. **Hazard Quotient (HQ) & Hazard Index (HI)**
   - Health risk assessment
   - HQ = exposure dose / reference dose
   - HI = sum of HQs for all contaminants

7. **Carcinogenic Risk (CR)**
   - For carcinogenic metals (As, Cd, Cr, Pb)

8. **Toxic Risk Index (TRI)**
   - Integrated toxicity assessment

---

## 6. KEY STATISTICS & FINDINGS

### 6.1 Model Performance Summary

**Rainy Season (Most Challenging):**
- Best Model: Transformer CNN GNN MLP (R² = 0.9604)
- Performance range: 0.8608 - 0.9604 across 9 models
- Top 3 models all R² > 0.95
- RMSE: 15.74 (Transformer, best) to 29.50 (Dual Attention, worst)

**Winter Season (Most Stable):**
- Best Model: Transformer CNN GNN MLP (R² = 0.9721)
- Performance range: 0.8402 - 0.9721
- Nearly all models R² > 0.95 (7 out of 9)
- RMSE: 6.55 (Transformer, best) - 49.38% improvement vs rainy season
- Most stable predictions (MAE = 4.45)

### 6.2 Model Architecture Insights:
- **Transformer fusion significantly outperforms** simpler concatenation
- **GNN components critical** - models ignoring spatial relationships perform worse
- **Multi-modal fusion necessary** - models with single input type underperform
- **Winter predictions easier** - more stable environmental conditions

### 6.3 Feature Importance:
- Feature importance CSV files in FeatureImportance/ directory
- LIME and permutation methods both employed
- Top features likely: metal concentrations, proximity to industrial areas, hydrological distance

---

## 7. SAMPLING STRATEGY

**Sampling Points:**
- 100-sample schema (primary)
- 200-sample schema (larger dataset)
- Winter variants (100W, 100WW)
- Test set (200test)

**Spatial Locations:**
- Sampling near industrial outfalls
- Urban settlement areas
- Stratified across 5 rivers
- Shapefiles with geographic coordinates (.shp/.dbf/.prj files)

**Temporal Coverage:**
- Winter: November-February
- Rainy: June-September
- 2 distinct seasonal datasets per analysis

---

## 8. ENVIRONMENTAL INDICES (CNN Input)

**Remote Sensing Indices:**
- NDVI: Normalized Difference Vegetation Index
- NDWI: Normalized Difference Water Index
- SAVI: Soil-Adjusted Vegetation Index
- NDBI: Normalized Difference Built-up Index
- NDBSIC: Normalized Difference Bare Soil Index
- MNDWI: Modified NDWI
- And additional indices (palettes in Pallete/ directory)

**Soil Properties:**
- Clay, Sand, Silt percentages
- Used in CNN input patches

---

## 9. CURRENT PROJECT STATUS

### Completed:
✅ Field sampling (rainy & winter seasons)
✅ Laboratory analysis (metal concentrations)
✅ GIS data preparation (spatial interpolation, feature extraction)
✅ Model training (9 ensemble architectures)
✅ Model evaluation (all metrics calculated)
✅ Feature importance analysis (LIME & permutation)
✅ Risk assessment calculations
✅ LaTeX article draft written

### In Progress / Needs Clarification:
❓ Final results section completion
❓ Figure finalization
❓ Discussion section depth
❓ Journal submission preparation

---

## 10. KEY FILES REFERENCE TABLE

| Purpose | File Location | Key Info |
|---------|---------------|----------|
| Raw data | `/data/*.csv` | Sediment & water chemistry |
| Spatial data | `/gis/data/*.shp` | Sample locations & features |
| IDW rasters | `/gis/IDW/*.gpkg` | Interpolated heavy metals |
| Training data | `/gis/data/train_data/` | Prepared ML datasets |
| Rainy models | `/gis/SedimentRainy/*.ipynb` | 11 model architectures |
| Winter models | `/gis/SedimentWinter/*.ipynb` | 11 model variants |
| Feature importance | `/gis/SedimentRainy/FeatureImportance/` | LIME & permutation results |
| Risk assessment | `/Python/*.ipynb` | Health risk & indices |
| PCA analysis | `/R/*.csv` | Source apportionment |
| Article | `/draft/main.tex` | LaTeX manuscript |
| Metrics table | `/draft/ModelPerformanceTable.tex` | Model comparison |

---

## 11. TECHNICAL STACK

**Data Processing & ML:**
- Python (Jupyter notebooks, TensorFlow/Keras)
- Pandas, NumPy, Scikit-learn
- Rasterio (spatial data)
- GeoPandas (shapefiles)

**Statistical Analysis:**
- R (PCA, source apportionment)
- RMarkdown for documentation

**Spatial Analysis:**
- QGIS (visualization, styling)
- GeoPackage format for rasters

**Document Preparation:**
- LaTeX (Elsevier elsarticle class)
- PDFLaTeX compilation

---

## 12. NOTES FOR FUTURE WORK

1. **Model Interpretability:** Feature importance already generated; need to interpret which environmental drivers matter most

2. **Spatial Visualization:** Maps of predicted RI, top features should be generated

3. **Publication:** Article nearly complete; needs final polish and discussion deepening

4. **Model Deployment:** Transformer CNN GNN MLP recommended for production use

5. **Cross-validation:** Models appear to use train/test split; consider cross-validation for robustness

6. **Uncertainty Quantification:** Could add confidence intervals to predictions

---

**Document Created:** 2025-12-26
**Purpose:** Serve as reference for understanding project structure, data flows, and analysis approaches
