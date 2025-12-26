# CLARIFICATION QUESTIONS FOR PROJECT COMPLETION

## RESEARCH & METHODOLOGY QUESTIONS

### 1. **Source Apportionment Methods**
- The article mentions PCA analysis in results, but I see R scripts for PCA factor analysis
- **Q: Are you using PCA as the primary source apportionment method, or is it complementary to the deep learning models?**
- **Q: How do the PCA source profiles (from R/) relate to the model predictions?**
- Are there other source apportionment techniques planned (e.g., PMF - Positive Matrix Factorization, CMB - Chemical Mass Balance)?

### 2. **Risk Index (RI) Target Variable**
- I see RI is the target variable for model predictions
- **Q: How is RI calculated? Is it a weighted combination of Cf, Igeo, EF, and other indices?**
- **Q: Are the RI values the same for rainy and winter seasons, or are they season-specific?**
- **Q: What is the range and distribution of RI values in your dataset?**

### 3. **Water vs Sediment Analysis**
- The title mentions "water and sediment" but the GIS/models appear focused on sediment
- **Q: Is water sample analysis included in the paper, or is this primarily sediment-focused?**
- If water: **Q: Are water metal concentrations used as features or as additional targets?**

### 4. **Sampling Strategy Details**
- I see 100-sample and 200-sample schemas
- **Q: Is 100 samples the primary analysis, or is 200 samples the final version?**
- **Q: How many replicates were taken at each station? (You mention triplicate analysis in methods)**
- **Q: What is the specific number of stations per river?**

### 5. **Seasonal Representation**
- You have winter and rainy data
- **Q: Were samples collected in the SAME locations across both seasons?**
- **Q: How many sampling campaigns in total? One per season or multiple?**

---

## DATA & FEATURES QUESTIONS

### 6. **Feature Set Definition**
Looking at the data files, I can identify:
- Direct metal measurements (Cr, Ni, Cu, As, Cd, Pb)
- Water quality (pH, EC, TDS, DO, turbidity)
- Hydrological features (Hydro_LULC_*.csv)
- Remote sensing indices (NDVI, NDWI, etc.)
- LULC classifications

**Q: Are all of these used simultaneously in the models, or do different models use different feature sets?**

**Q: Which features are used for:
- CNN input (specify which rasters)?
- MLP input (specify which tabular features)?
- GNN input (only distance-based kernel or additional spatial features)?**

### 7. **Patch Size for CNN**
- CNN uses windowing technique on rasters
- **Q: What is the patch size (in pixels)? What is the spatial resolution of the rasters?**
- **Q: Example: if raster is 30m resolution and patch is 32x32 pixels, that's 960m coverage**

### 8. **GNN Graph Construction**
- I see G_ij = exp(-d_ij/τ) in algorithms
- **Q: What is the value of τ (tau)? Was it tuned?**
- **Q: Do you use k-nearest neighbors to sparsify the graph, or use full all-to-all connectivity?**
- **Q: How many edges per node on average?**

### 9. **Background/Reference Concentrations**
- EF and Igeo require background values
- **Q: What reference/background values are used for each metal?**
- **Q: Are they from literature, local geology, or calculated from your data?**

### 10. **Missing Data Handling**
- Algorithms2.tex mentions "FillNaN with zeros"
- **Q: Were there many missing values? In which features?**
- **Q: Is zero-filling appropriate, or were imputation methods considered?**

---

## MODEL & TRAINING QUESTIONS

### 11. **Train/Test Split**
- I see PredTest.csv but no detailed split information
- **Q: What is the train/test split ratio?**
- **Q: Is it random split, spatial split (by location), or temporal split (by campaign)?**
- **Q: Are rainy and winter models trained separately on their own data, or jointly?**

### 12. **Model Training Details**
- Keras files available but training parameters unclear
- **Q: What loss function is used (MSE, MAE, Huber)?**
- **Q: What optimizer (Adam, SGD)? Learning rate?**
- **Q: Batch size? Number of epochs? Early stopping criteria?**
- **Q: Are hyperparameters the same across all 9 model variants?**

### 13. **Cross-validation**
- **Q: Was cross-validation performed? If so, what type (k-fold, spatial k-fold)?**
- Are the reported metrics from holdout test set or cross-validation?

### 14. **Seasonal Model Linkage**
- You have separate rainy and winter models
- **Q: Are predictions for each season independent, or somehow linked?**
- **Q: For producing risk maps, do you average predictions or select one per season?**

### 15. **Model Uncertainty**
- Predictions appear deterministic (single R² value per model)
- **Q: Do the models output confidence intervals or uncertainty estimates?**
- **Q: How are you handling prediction uncertainty in the results?**

---

## RESULTS & INTERPRETATION QUESTIONS

### 16. **Feature Importance Results**
- FeatureImportance/ directory has LIME and permutation CSVs
- **Q: What are the top 5 most important features for the best model (Transformer)?**
- **Q: Do spatial features (GNN) or spectral features (CNN) or tabular features (MLP) dominate?**
- **Q: Do important features differ between rainy and winter seasons?**

### 17. **Source Attribution**
- The title emphasizes "source apportionment" but models predict RI (risk index), not sources
- **Q: How do you identify sources (industrial, traffic, geogenic, agricultural) from model outputs?**
- **Q: Is PCA factor analysis the source identification method?**
- **Q: Can you link high prediction errors to specific sources (spatial misattribution)?**

### 18. **Seasonal Patterns**
- Results show Transformer performs better in winter (R² 0.9721 vs 0.9604)
- **Q: What is the physical reason for the 49% RMSE improvement in winter?**
- **Q: Are specific metals or rivers more predictable in one season?**
- **Q: Do high-flow rainy season effects make certain areas unpredictable?**

### 19. **Model Generalization**
- All models tested on same rainy/winter data
- **Q: If you had a third season (summer, spring), would models generalize well?**
- **Q: Are there indications of overfitting in any models?**

### 20. **Prediction Accuracy in Terms of Risk**
- RMSE ~15-16 for rainy, ~6-8 for winter
- **Q: What is the typical RI value range? (Example: 0-100, 0-1000)?**
- **Q: What does RMSE of 15 mean in terms of risk classification accuracy?**
- **Q: Can the model differentiate high-risk from low-risk areas reliably?**

---

## ARTICLE & PRESENTATION QUESTIONS

### 21. **Article Structure**
Looking at draft files:
- Main sections: Methodology, Results (with subsections), Conclusion
- Missing: Introduction text, Discussion section
- **Q: Is the Introduction section to be written separately?**
- **Q: Should results subsections (Igeo, EF, PLI) be condensed or expanded?**

### 22. **Figures & Tables**
- I see metrics table in ModelPerformanceTable.tex
- **Q: What figures are needed for the article?**
  - Maps of predicted RI spatial distribution?
  - Feature importance visualizations?
  - Model prediction vs observed scatter plots?
  - Confusion matrices for risk categories?

### 23. **Comparison with Literature**
- Very few citations in current draft
- **Q: Should the discussion compare results with other river studies in Bangladesh or globally?**
- **Q: Are there standard RI or health risk thresholds to compare against?**

### 24. **Monte Carlo Results**
- MonteCarlo.tex is referenced but empty in results section
- **Q: What is being analyzed with Monte Carlo? Uncertainty? Sensitivity?**
- **Q: Should this be included or removed from the article?**

### 25. **Health Risk Assessment**
- HealthRisk.tex exists with HQ, HI, carcinogenic risk results
- **Q: What is the population group assessed (children, adults, vulnerable)?**
- **Q: Should the discussion include mitigation recommendations based on HI results?**

---

## PRACTICAL/WORKFLOW QUESTIONS

### 26. **What's Your Immediate Need?**
**Q: What specific task do you want help with right now?**
- ✓ Completing the article (writing sections)?
- ✓ Analyzing feature importance results?
- ✓ Creating visualizations?
- ✓ Performing additional statistical tests?
- ✓ Comparing models more rigorously?
- ✓ Preparing for journal submission?
- ✓ Something else?

### 27. **Model Selection**
- **Q: Have you decided which model(s) to feature in the final paper?**
- Recommendation: Transformer CNN GNN MLP (best across both seasons)
- But should you discuss all 9 or just top 3-5?

### 28. **Publication Target**
- Current format: Elsevier elsarticle (good choice)
- **Q: Have you identified a specific Elsevier journal (Environmental Science & Technology, Science of Total Environment, etc.)?**
- **Q: Does the target journal have specific requirements for ML/DL papers?**

### 29. **Code Availability**
- **Q: Will you publish code with the article (GitHub, supplementary)?**
- If yes, which notebooks/scripts should be included?

### 30. **Data Availability**
- Raw data in /data and processed data in /gis/data
- **Q: Will raw data be made publicly available?**
- **Q: Are there confidentiality concerns (sampling locations near industries)?**

---

## PRIORITIZED RECOMMENDATIONS

Based on my analysis, here are the **most critical clarifications**:

### CRITICAL (affects article content):
1. **Source apportionment method** - How do you identify sources? (Q1, Q17)
2. **RI calculation** - Formula for target variable (Q2)
3. **Feature specification** - Which features go to which branches? (Q6)
4. **Article completion** - Introduction and Discussion sections missing (Q21)

### IMPORTANT (affects interpretation):
5. **Train/test strategy** - Affects generalization claims (Q11)
6. **Top features** - Essential for discussion (Q16)
7. **Seasonal explanation** - Why winter better than rainy? (Q18)
8. **Model selection** - Which to feature prominently (Q27)

### USEFUL (improves robustness):
9. **Hyperparameter details** - For reproducibility (Q12)
10. **Cross-validation** - For robustness assessment (Q13)

---

**Please address these questions so I can provide more targeted assistance!**
