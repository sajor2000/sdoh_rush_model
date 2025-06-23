#!/usr/bin/env python3
"""
Create final clean folder structure for paper publication and reproducible science
"""

import os
import shutil
from pathlib import Path
import json

def create_clean_structure():
    """Create clean publication-ready structure"""
    
    print("Creating final clean structure for paper publication...")
    
    # Define what to keep for reproducible science
    keep_files = {
        # Core model files
        'models/xgboost_scientific_calibrated.joblib',
        'models/scientific_model_metadata.json',
        'models/xgboost_scientific_base.json',
        
        # Main production code
        'src/sdoh_risk_screener.py',
        'src/config.py',
        
        # Key training script
        'scripts/train_scientifically_correct.py',
        
        # Analysis scripts for paper
        'scripts/comprehensive_fairness_analysis.py',
        'scripts/generate_tripod_figures_jama.py',
        'scripts/generate_table1_jama.py',
        'scripts/senior_clinic_threshold_analysis.py',
        'scripts/verify_test_metrics.py',
        'scripts/create_simplified_executive_report.py',
        
        # Documentation
        'README.md',
        'LICENSE',
        'requirements.txt',
        'RELEASE_NOTES.md',
        'TRIPOD_AI_COMPLIANCE_REPORT.md',
        
        # Configuration files
        '.gitignore',
        '.gitattributes',
        '_config.yml',
        
        # Key results for paper
        'results/figures/jama/',
        'results/reports/fairness_metrics_summary.csv',
        'results/reports/comprehensive_fairness_report.txt',
        'results/tables/table1_jama.csv',
        'results/tables/table1_jama.tex',
        'results/tables/table1_jama.txt',
        
        # Executive report
        'index.html',
        'results/simplified_executive_report.html',
    }
    
    # Create backup of current structure
    backup_dir = Path('../SDOH_Prediction_Model_BACKUP')
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    
    print("Creating backup...")
    shutil.copytree('.', backup_dir, ignore=shutil.ignore_patterns('.git'))
    
    # Remove unnecessary files and folders
    remove_items = [
        # Old model versions
        'models/model_artifact.joblib',
        'models/xgboost_best.json',
        'models/xgboost_balanced_final.json',
        'models/xgboost_balanced_calibrated_final.joblib',
        'models/balanced_final_metadata.json',
        'models/xgboost_realdata_base.json',
        'models/xgboost_realdata_calibrated.joblib',
        'models/realdata_model_metadata.json',
        'models/xgboost_calibrated_final.joblib',
        'models/model_metadata_final.json',
        
        # Unnecessary scripts
        'scripts/refit_model_real_data.py',
        'scripts/refit_model_real_data_fixed.py',
        'scripts/train_balanced_calibrated_model.py',
        'scripts/train_balanced_fast.py',
        'scripts/train_final_balanced.py',
        'scripts/plot_real_calibration.py',
        'scripts/create_executive_report.py',
        'scripts/create_integrated_executive_report.py',
        'scripts/create_professional_shap_plots.py',
        
        # Old documentation files
        'docs/guides/',
        'docs/summaries/',
        'docs/executive_report/',
        'PROJECT_STRUCTURE.md',
        'GIT_RELEASE_CHECKLIST.md',
        'GIT_RELEASE_SUMMARY.md',
        'prepare_for_git.sh',
        
        # Excess result files
        'results/figures/performance_balanced_final.png',
        'results/figures/feature_importance_final.png',
        'results/figures/threshold_analysis_final.png',
        'results/figures/threshold_tradeoffs.png',
        'results/figures/performance_scientifically_valid.png',
        'results/figures/validation_vs_test_comparison.png',
        'results/figures/real_calibration_analysis.png',
        'results/figures/decision_curve_final.png',
        'results/figures/model_training_results.png',
        'results/figures/shap_summary_advanced.png',
        'results/figures/shap_waterfall_*.png',
        'results/figures/shap_dependence_plots.png',
        'results/figures/shap_interaction_heatmap.png',
        'results/figures/threshold_comparison_simple.png',
        
        # Old reports
        'results/reports/senior_clinic_threshold_recommendations.txt',
        'results/tables/table1_summary_statistics.txt',
        
        # Notebooks and other files
        'notebooks/',
        'tests/',
        'data/',
    ]
    
    print("Removing unnecessary files...")
    for item in remove_items:
        item_path = Path(item)
        if item_path.exists():
            if item_path.is_dir():
                shutil.rmtree(item_path)
                print(f"  Removed directory: {item}")
            else:
                item_path.unlink()
                print(f"  Removed file: {item}")
    
    # Create clean directory structure
    clean_dirs = [
        'src',
        'models',
        'scripts',
        'results/figures/jama',
        'results/figures/publication',
        'results/reports',
        'results/tables',
        'docs',
    ]
    
    print("Creating clean directory structure...")
    for dir_path in clean_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Move remaining important figures to publication folder
    pub_figures = [
        'results/figures/comprehensive_fairness_dashboard.png',
        'results/figures/senior_clinic_threshold_analysis.png',
        'results/figures/senior_clinic_implementation_flowchart.png',
        'results/figures/feature_importance_professional_labels.png',
        'results/figures/detailed_fairness_*.png',
    ]
    
    print("Organizing publication figures...")
    for pattern in pub_figures:
        for fig_path in Path('.').glob(pattern):
            if fig_path.exists():
                dest_path = Path('results/figures/publication') / fig_path.name
                shutil.copy2(fig_path, dest_path)
                print(f"  Copied to publication: {fig_path.name}")
    
    # Create clean utils file
    create_utils_file()
    
    # Create final README
    create_final_readme()
    
    # Create clean requirements
    create_clean_requirements()
    
    print("\n✅ Clean structure created successfully!")
    print("Backup saved to:", backup_dir.absolute())
    
    return True

def create_utils_file():
    """Create a clean utils file"""
    utils_content = '''"""
Utility functions for SDOH Risk Screening Model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_performance_metrics(y_true, y_pred_proba, threshold=0.05):
    """Calculate comprehensive performance metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # Calculated metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Performance metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    ece = calculate_expected_calibration_error(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    return {
        'auc': auc,
        'auprc': auprc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'ece': ece,
        'brier_score': brier_score,
        'screening_rate': y_pred.mean(),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def load_test_data(data_path, test_size=0.2, random_seed=2025):
    """Load and split data maintaining same test set as training"""
    df = pd.read_csv(data_path)
    
    # Use same methodology as training script
    np.random.seed(random_seed)
    n_total = len(df)
    n_test = int(n_total * test_size)
    test_indices = df.index[-n_test:]
    
    return df.iloc[test_indices].copy()
'''
    
    with open('src/utils.py', 'w') as f:
        f.write(utils_content)
    
    print("  Created src/utils.py")

def create_final_readme():
    """Create final clean README"""
    readme_content = '''# SDOH Risk Screening Model - Publication Version

[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxx-blue)](https://doi.org/10.xxxx/xxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

Production-ready machine learning model for identifying patients at risk for social determinants of health (SDOH) needs. Uses XGBoost with calibration to achieve fair, accurate predictions across all demographic groups.

## Model Performance (Test Set, n=78,745)

- **AUC**: 0.765
- **Sensitivity**: 72.2% (at 5% threshold)
- **Specificity**: 67.9%
- **PPV**: 13.8%
- **Calibration Error**: 0.008 (excellent)
- **Fairness**: Verified across age, sex, race, ethnicity

## Repository Structure

```
├── src/                          # Core production code
│   ├── sdoh_risk_screener.py    # Main model class
│   ├── config.py                # Configuration
│   └── utils.py                 # Utility functions
├── models/                      # Trained models
│   ├── xgboost_scientific_calibrated.joblib
│   └── scientific_model_metadata.json
├── scripts/                     # Analysis scripts
│   ├── train_scientifically_correct.py
│   ├── comprehensive_fairness_analysis.py
│   ├── generate_tripod_figures_jama.py
│   └── generate_table1_jama.py
├── results/                     # Generated outputs
│   ├── figures/jama/           # TRIPOD-AI figures
│   ├── figures/publication/    # Additional figures
│   ├── reports/               # Analysis reports
│   └── tables/                # Publication tables
├── docs/                       # Documentation
├── index.html                  # Executive report
└── requirements.txt            # Dependencies
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/sajor2000/sdoh_rush_model.git
cd sdoh_rush_model

# Install dependencies
pip install -r requirements.txt

# Use the model
from src.sdoh_risk_screener import SDOHRiskScreener

screener = SDOHRiskScreener()
results = screener.predict_risk(patient_data)
```

## Data Requirements

The model requires patient demographics and census tract linkage to:
- CDC Social Vulnerability Index (SVI) data
- Area Deprivation Index (ADI) data

## Thresholds for Different Settings

- **General Population**: 5.0% (screens 34.8%, PPV 13.8%)
- **Geriatric Clinics (65+)**: 8.4% (screens 7.7%, PPV 19.5%)

## Reproducibility

All analyses use:
- Fixed random seed (2025)
- Proper 60/20/20 train/validation/test split
- TRIPOD-AI compliant methodology
- No data leakage

## Files for Paper Submission

### Main Figures (JAMA Format)
- `results/figures/jama/figure1_model_performance.png`
- `results/figures/jama/figure2_feature_importance.png`
- `results/figures/jama/figure3_subgroup_performance.png`
- `results/figures/jama/figure4_decision_curve.png`

### Tables
- `results/tables/table1_jama.tex` (LaTeX format)
- `results/tables/table1_jama.csv` (Data format)

### Supplementary Materials
- `results/reports/fairness_metrics_summary.csv`
- `results/reports/comprehensive_fairness_report.txt`
- `TRIPOD_AI_COMPLIANCE_REPORT.md`

## Model Training

To retrain the model:
```bash
python scripts/train_scientifically_correct.py
```

## Fairness Analysis

To reproduce fairness analysis:
```bash
python scripts/comprehensive_fairness_analysis.py
```

## Citation

```bibtex
@article{sdoh_risk_model_2024,
  title={AI-Powered Social Determinants of Health Screening: A Fair and Accurate Model for Clinical Implementation},
  author={[Your Name] and [Co-authors]},
  journal={[Journal Name]},
  year={2024},
  doi={10.xxxx/xxxx}
}
```

## License

MIT License - See LICENSE file for details.

## Contact

For questions about implementation or research collaboration:
- Technical: [your-email@institution.edu]
- Clinical: [clinical-contact@institution.edu]

---
*Model Version 2.0 | TRIPOD-AI Compliant | Publication Ready*
'''
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("  Updated README.md for publication")

def create_clean_requirements():
    """Create minimal requirements file"""
    requirements = '''# Core dependencies for SDOH Risk Screening Model
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
joblib>=1.1.0

# For analysis and visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# For statistical analysis
scipy>=1.7.0

# For progress bars (optional)
tqdm>=4.62.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("  Updated requirements.txt")

def main():
    """Main function"""
    try:
        success = create_clean_structure()
        if success:
            print("\n" + "="*60)
            print("✅ FINAL CLEAN STRUCTURE CREATED!")
            print("="*60)
            print("\nRepository is now publication-ready with:")
            print("• Core model and production code")
            print("• TRIPOD-AI compliant training script")
            print("• Publication figures and tables")
            print("• Fairness analysis and reports")
            print("• Clean documentation")
            print("• Minimal dependencies")
            print("\nBackup of original structure saved in ../SDOH_Prediction_Model_BACKUP")
            print("\nReady for:")
            print("• Paper submission")
            print("• Code sharing")
            print("• Reproducible science")
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()