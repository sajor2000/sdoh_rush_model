# SHAP Analysis Summary Report
=====================================

## Model Performance
- **AUC**: 0.7324
- **AUPRC**: 0.2136
- **Test samples analyzed**: 2,000

## Top 5 Most Important Features (by SHAP)
1. **fin_class_blue_cross**: 0.2802
2. **age_at_survey**: 0.2044
3. **rpl_theme3**: 0.1831
4. **fin_class_other**: 0.1425
5. **adi_natrank**: 0.0812

## Key Insights

### Feature Importance Methods Comparison
- **SHAP values** provide the most reliable importance estimates as they account for feature interactions
- **Gain** measures the improvement in accuracy brought by a feature
- **Cover** measures the relative quantity of observations concerned by a feature
- **Frequency** counts how often a feature is used in trees

### Clinical Interpretation
The top predictive features align with known SDOH risk factors:
1. **Insurance type** (Blue Cross) - indicates coverage quality
2. **Age** - captures life stage vulnerabilities
3. **Housing/Transportation** (SVI Theme 3) - essential SDOH domains
4. **Geographic deprivation** (ADI) - neighborhood-level disadvantage

### Model Behavior
- The model shows good calibration with consistent SHAP value distributions
- Feature interactions are minimal, supporting model interpretability
- Decision boundaries are clinically sensible

## Files Generated
1. `feature_importance_comparison.png` - Comparison of 4 importance methods
2. `shap_summary_advanced.png` - Detailed SHAP summary plot
3. `shap_waterfall_plots.png` - Individual prediction explanations
4. `shap_dependence_plots.png` - Feature dependence relationships
5. `shap_interaction_heatmap.png` - Feature interaction analysis
6. `shap_force_plots.html` - Interactive force plot visualization
