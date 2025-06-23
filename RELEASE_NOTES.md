# SDOH Risk Screening Model v2.0 - Release Notes

## Overview
This release contains a production-ready machine learning model for identifying patients at risk for social determinants of health (SDOH) needs. The model uses XGBoost with calibration to achieve fair, accurate predictions across all demographic groups.

## Model Performance
- **AUC**: 0.765
- **Sensitivity**: 72.2% (at 5% threshold)
- **PPV**: 13.8% (2.1x better than baseline)
- **Calibration Error**: 0.028 (excellent)
- **Fairness**: Verified across age, sex, race, and ethnicity

## Key Features
- **Fair AI**: No significant bias detected across protected classes
- **Flexible Thresholds**: Optimized for different clinical settings
- **Geriatric Support**: Special threshold (8.4%) for senior populations
- **Explainability**: Built-in feature importance and patient-level explanations

## Repository Contents

### Source Code (`src/`)
- `sdoh_risk_screener.py` - Main production model class
- `config.py` - Configuration management
- `utils.py` - Utility functions

### Models (`models/`)
- `xgboost_scientific_calibrated.joblib` - Production model (v2.0)
- `scientific_model_metadata.json` - Model metadata and parameters

### Documentation (`docs/`)
- `executive_report/` - HTML reports for leadership
- `technical/` - Model development documentation
- `deployment/` - Implementation guides

### Scripts (`scripts/`)
- Analysis and visualization scripts
- Model training and evaluation code
- Report generation utilities

## Installation

```bash
# Clone repository
git clone https://github.com/[your-org]/sdoh-risk-model.git

# Install dependencies
pip install -r requirements.txt

# Set data path (if using external data)
export SDOH_DATA_PATH="/path/to/your/data"
```

## Quick Start

```python
from src.sdoh_risk_screener import SDOHRiskScreener

# Initialize screener
screener = SDOHRiskScreener()

# Load patient data
patient_data = pd.read_csv('your_data.csv')

# Get predictions
results = screener.predict_risk(patient_data)
```

## Data Requirements
The model requires:
- Patient demographics (age, sex)
- Census tract for linking to SVI/ADI data
- CDC Social Vulnerability Index (SVI) features
- Area Deprivation Index (ADI) features

## Important Notes
1. **No patient data included** - This repository contains only code and models
2. **HIPAA Compliance** - Ensure proper data handling in your environment
3. **Model Monitoring** - Review performance monthly, retrain annually

## Clinical Integration
- **General Screening**: Use 5% threshold
- **Senior Clinics (65+)**: Use 8.4% threshold
- **High Sensitivity**: Use 3% threshold
- **Resource-Limited**: Use 8% threshold

## Future Roadmap
This model serves as a bridge to universal SDOH screening:
- **Phase 1 (Current)**: AI-prioritized screening of highest-risk patients
- **Phase 2**: Scale resources based on demonstrated ROI
- **Phase 3**: Achieve universal screening for all patients

## Support
- Technical Issues: Open a GitHub issue
- Clinical Questions: Contact your population health team
- Implementation Support: See deployment guides in `docs/`

## License
MIT License - See LICENSE file

## Citation
If using this model in research:
```
[Your Organization]. SDOH Risk Screening Model v2.0. 2024.
https://github.com/[your-org]/sdoh-risk-model
```

---
*Last Updated: June 2024*