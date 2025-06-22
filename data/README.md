# Data Directory

## Overview

This directory is designed to hold the datasets used for training and evaluating the SDOH prediction model. **Note: No actual patient data is included in this repository due to privacy considerations.**

## Expected Data Structure

### Training Data Format

The model expects a CSV file with the following structure:

```csv
patient_id,age_at_survey,sex_female,fin_class_blue_cross,...,sdoh_two_yes
12345,45,1,0,...,1
12346,32,0,1,...,0
...
```

### Required Features (46 total)

#### Demographics
- `age_at_survey`: Patient age at time of survey
- `sex_female`: Binary indicator (1=female, 0=male)

#### Insurance/Financial
- `fin_class_blue_cross`: Blue Cross insurance indicator
- `fin_class_other`: Other insurance indicator
- `fin_class_medicare`: Medicare indicator
- `fin_class_medicaid`: Medicaid indicator
- `fin_class_self_pay`: Self-pay indicator

#### Geographic/Social Vulnerability Index (SVI)
- `rpl_theme1`: SVI Theme 1 (Socioeconomic status)
- `rpl_theme2`: SVI Theme 2 (Household composition)
- `rpl_theme3`: SVI Theme 3 (Race/ethnicity/language)
- `rpl_theme4`: SVI Theme 4 (Housing/transportation)
- `ep_pov150`: Percent poverty estimate
- `ep_minrty`: Percent minority estimate
- `ep_noveh`: Percent households with no vehicle

#### Area Deprivation Index (ADI)
- `adi_natrank`: ADI national ranking percentile
- `adi_staternk`: ADI state ranking percentile

#### Target Variable
- `sdoh_two_yes`: Binary outcome (1=2+ SDOH needs, 0=<2 SDOH needs)

### Data Requirements

1. **Sample Size**: Minimum 10,000 patients for stable model training
2. **Positive Rate**: ~6-7% patients with 2+ SDOH needs
3. **Missing Data**: <20% missing values per feature
4. **Data Quality**: Recent data (<2 years old) preferred

## Data Sources

### Primary Data Elements
- **Electronic Health Records (EHR)**: Demographics, insurance, utilization
- **Geographic Data**: Patient addresses for SVI/ADI calculation
- **SDOH Screening Results**: Validated SDOH assessment outcomes

### External Data Sources
- **CDC Social Vulnerability Index**: Census tract-level social vulnerability
- **Area Deprivation Index**: Neighborhood-level deprivation scores
- **US Census**: Demographic and socioeconomic data

## Data Preprocessing

### Steps Required Before Model Training

1. **Address Geocoding**: Convert patient addresses to census tracts
2. **SVI/ADI Linking**: Link geographic codes to vulnerability indices  
3. **Missing Value Imputation**: Handle missing data appropriately
4. **Feature Scaling**: Standardize numerical features
5. **Train/Test Split**: Stratified split maintaining outcome balance

### Privacy Considerations

- **De-identification**: Remove all direct identifiers (names, addresses, etc.)
- **Safe Harbor**: Follow HIPAA Safe Harbor guidelines
- **Limited Dataset**: Use only features necessary for model training
- **Access Controls**: Restrict data access to authorized personnel

## Getting Your Data Ready

### 1. Data Collection
```sql
-- Example SQL query structure
SELECT 
    patient_id,
    age_at_survey,
    sex_female,
    fin_class_blue_cross,
    -- ... other features
    sdoh_two_yes
FROM patient_cohort
WHERE survey_date >= '2020-01-01'
    AND survey_date <= '2023-12-31'
```

### 2. Geographic Data Processing
```python
# Example geocoding and SVI linkage
import pandas as pd

# Load patient addresses
addresses = pd.read_csv('patient_addresses.csv')

# Geocode to census tracts
# (Use your organization's geocoding service)

# Link to SVI data
svi_data = pd.read_csv('SVI_2020_US.csv')
final_data = addresses.merge(svi_data, on='FIPS')
```

### 3. Data Validation
```python
# Check data quality
def validate_data(df):
    print(f"Dataset shape: {df.shape}")
    print(f"Missing data:\n{df.isnull().sum()}")
    print(f"Outcome distribution:\n{df['sdoh_two_yes'].value_counts()}")
    
    # Check for required features
    required_features = ['age_at_survey', 'sex_female', ...]
    missing_features = set(required_features) - set(df.columns)
    if missing_features:
        print(f"Missing required features: {missing_features}")
```

## File Naming Convention

- `training_data_YYYY.csv`: Training dataset by year
- `test_data_YYYY.csv`: Hold-out test dataset
- `validation_data_YYYY.csv`: Validation dataset
- `feature_dictionary.csv`: Data dictionary with feature descriptions

## Data Usage Agreement

Before using this model with your data:

1. Ensure appropriate IRB approval for research use
2. Verify compliance with institutional data governance
3. Confirm patient consent for data use (if required)
4. Follow all applicable privacy regulations (HIPAA, GDPR, etc.)

## Contact

For questions about data requirements or formatting:
- Data Team: [contact information]
- Technical Lead: [contact information]
- Privacy Officer: [contact information]