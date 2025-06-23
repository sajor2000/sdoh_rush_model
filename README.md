# SDOH Risk Screening Model - Publication Version

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
