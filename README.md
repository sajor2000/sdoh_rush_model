# SDOH Risk Screening Model

## Overview
This repository contains a production-ready machine learning model for identifying patients at risk for social determinants of health (SDOH) needs. The model uses XGBoost with calibration to predict which patients are likely to have 2 or more social needs.

## Model Performance
- **AUC:** 0.765
- **Sensitivity:** 72.2% (at 5% threshold)
- **Specificity:** 66.8%
- **PPV:** 13.8% (2.1x better than baseline)
- **Calibration Error:** 0.028 (excellent)

## Key Features
- Fair across all demographic groups (age, sex, race, ethnicity)
- Uses CDC Social Vulnerability Index (SVI) and Area Deprivation Index (ADI)
- Optimized thresholds for different clinical settings
- Production-ready with built-in explanations

## Data Setup

**Important**: The data file is not included in this repository for privacy reasons. 

To use the training scripts:
1. Obtain the data file `sdoh2_ml_final_all_svi.csv`
2. Either:
   - Set the environment variable: `export SDOH_DATA_PATH=/path/to/your/sdoh2_ml_final_all_svi.csv`
   - Or update the `DATA_PATH` in `config.py` to point to your local data file

## Quick Start

```python
from sdoh_risk_screener import SDOHRiskScreener

# Initialize the screener
screener = SDOHRiskScreener()

# Load patient data (requires demographics + SVI/ADI features)
patient_data = pd.read_csv('your_patient_data.csv')

# Get risk predictions
results = screener.predict_risk(patient_data)

# For geriatric populations (65+), use adjusted threshold
senior_results = screener.predict_risk(senior_data, use_geriatric_threshold=True)
```

## Installation

```bash
pip install -r requirements.txt
```

## Model Files
- `models/xgboost_scientific_calibrated.joblib` - Calibrated production model
- `models/scientific_model_metadata.json` - Model metadata and feature names

## Fairness & Bias
The model has been extensively tested for fairness:
- Statistical parity difference < 10% across all groups
- Equal opportunity difference < 10% across all groups  
- Disparate impact ratio > 0.8 for all groups

## Clinical Integration
Recommended integration points:
- Primary care annual visits
- Hospital discharge planning
- Emergency department screening
- Geriatric assessments

## Threshold Recommendations
- **General Population:** 5.0% (screens 34.8%, PPV 13.8%)
- **Geriatric Clinics:** 8.4% (screens 7.7%, PPV 19.5%)
- **High Sensitivity:** 3.0% (screens 52.3%, PPV 10.7%)
- **Resource-Limited:** 8.0% (screens 18.5%, PPV 18.4%)

## Contributing
See CONTRIBUTING.md for guidelines on model updates and validation requirements.

## License
MIT License - See LICENSE file for details.

## Citation
If using this model in research, please cite:
```
[Your Organization] SDOH Risk Screening Model v2.0. 
Available at: https://github.com/[your-org]/sdoh-risk-model
```

## Contact
- Technical Questions: data-science@[your-org].org
- Clinical Questions: population-health@[your-org].org
