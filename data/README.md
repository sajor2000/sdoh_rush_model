# SDOH Prediction Model Datasets

This directory contains all dataset partitions used for the SDOH prediction model.

## Dataset Information

- **Created**: 2025-06-23T08:40:35.597859
- **Random Seed**: 2025 (for reproducibility)
- **Split Ratios**: 60% train / 20% validation / 20% test
- **Target Variable**: sdoh_two_yes

## Dataset Sizes

- Full dataset: 393,725 samples
- Training set: 157,490 samples
- Validation set: 78,745 samples
- Test set: 78,745 samples

## Important Notes

1. **Race/Ethnicity Exclusion**: The model does not use race or ethnicity as features. These columns have been removed from all datasets to prevent bias.

2. **Features**: The model uses patient demographics (age, gender) and address-based social determinants (SVI/ADI from Census tract data).

3. **File Formats**: Each dataset is saved in both CSV and pickle formats:
   - CSV files for human readability and compatibility
   - Pickle files for faster loading in Python

4. **Reproducibility**: Using seed=2025 ensures the exact same splits can be recreated.

## Target Distribution

### Full Dataset
- No SDOH need: 367,621 (93.4%)
- SDOH need: 26,104 (6.6%)

### Test Dataset (Used for Final Evaluation)
- No SDOH need: 73,524 (93.4%)
- SDOH need: 5,221 (6.6%)

## Loading Datasets

```python
import pandas as pd

# Load CSV
test_df = pd.read_csv('data/test/test_data.csv')

# Load pickle (faster)
test_df = pd.read_pickle('data/test/test_data.pkl')
```

## Model Performance on Test Set

- AUC: 0.765
- Average Precision: 0.211
- Sensitivity: 72.2%
- Specificity: 67.9%
- PPV: 13.8%
- NPV: 97.1%
- Screening Rate: 34.8%
