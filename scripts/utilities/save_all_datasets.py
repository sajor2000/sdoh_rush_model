#!/usr/bin/env python3
"""
Save all dataset partitions (train/validation/test) using the exact methodology from model training.
This ensures reproducibility and eliminates future data access issues.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
import datetime

# CRITICAL: Use the exact same seed as in training for reproducibility
SEED = 2025

def load_and_preprocess_data():
    """Load the original dataset and preprocess it exactly as done during training."""
    print("Loading original dataset...")
    
    # Load the full dataset
    df = pd.read_csv('/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv')
    
    print(f"Original dataset shape: {df.shape}")
    
    # Target column
    target_col = 'sdoh_two_yes'
    
    # Remove identifier columns (same as training)
    id_cols = ['member_id', 'pat_key', 'patient_id', 'encounter_id', 'claim_id']
    for col in id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Remove date/time columns
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'year', 'month'])]
    df = df.drop(columns=date_cols, errors='ignore')
    
    # IMPORTANT: Remove race and ethnicity columns (model doesn't use them)
    race_eth_cols = ['race_category', 'ethnicity_category']
    df = df.drop(columns=race_eth_cols, errors='ignore')
    print(f"Dataset shape after preprocessing: {df.shape}")
    
    # Handle remaining categorical columns - convert to numeric
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    # Convert categorical to dummy variables
    if len(categorical_cols) > 0:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Verify target column exists
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Count target distribution
    target_counts = df[target_col].value_counts()
    print(f"\nTarget distribution:")
    print(f"  No SDOH need (0): {target_counts.get(0, 0):,}")
    print(f"  SDOH need (1): {target_counts.get(1, 0):,}")
    print(f"  Prevalence: {target_counts.get(1, 0) / len(df) * 100:.1f}%")
    
    return df, target_col

def create_splits(df, target_col):
    """Create train/validation/test splits using the exact methodology from training."""
    print("\nCreating 60/20/20 split with seed=2025...")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # First split: 60% train+val, 40% temp (will be split into 20% val + 20% test)
    X_train_val, X_temp, y_train_val, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=SEED, stratify=y
    )
    
    # Second split: Split the 40% temp into 20% val and 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    
    # Final split: Split the 60% train_val to get final train set
    X_train, _, y_train, _ = train_test_split(
        X_train_val, y_train_val, test_size=0.333333, random_state=SEED, stratify=y_train_val
    )
    
    # Recombine features and target for saving
    train_df = X_train.copy()
    train_df[target_col] = y_train
    
    val_df = X_val.copy()
    val_df[target_col] = y_val
    
    test_df = X_test.copy()
    test_df[target_col] = y_test
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify test set size matches expected
    if len(test_df) != 78745:
        print(f"WARNING: Test set size ({len(test_df)}) doesn't match expected (78,745)")
    
    return train_df, val_df, test_df

def save_datasets(full_df, train_df, val_df, test_df, target_col):
    """Save all datasets in both CSV and pickle formats."""
    base_path = Path('/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/SDOH_Prediction_Model/data')
    
    datasets = {
        'full': full_df,
        'train': train_df,
        'validation': val_df,
        'test': test_df
    }
    
    print("\nSaving datasets...")
    for name, df in datasets.items():
        # Save as CSV
        csv_path = base_path / name / f'{name}_data.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved {name} CSV: {csv_path}")
        
        # Save as pickle for faster loading
        pkl_path = base_path / name / f'{name}_data.pkl'
        df.to_pickle(pkl_path)
        print(f"  Saved {name} pickle: {pkl_path}")
    
    # Save metadata
    metadata = {
        'created_date': datetime.datetime.now().isoformat(),
        'random_seed': SEED,
        'split_ratios': '60/20/20 (train/validation/test)',
        'target_column': target_col,
        'dataset_sizes': {
            'full': len(full_df),
            'train': len(train_df),
            'validation': len(val_df),
            'test': len(test_df)
        },
        'feature_columns': [col for col in full_df.columns if col != target_col],
        'excluded_columns': ['race_category', 'ethnicity_category'],
        'target_distribution': {
            'full': {
                'no_sdoh_need': int((full_df[target_col] == 0).sum()),
                'sdoh_need': int((full_df[target_col] == 1).sum()),
                'prevalence': float((full_df[target_col] == 1).mean())
            },
            'train': {
                'no_sdoh_need': int((train_df[target_col] == 0).sum()),
                'sdoh_need': int((train_df[target_col] == 1).sum()),
                'prevalence': float((train_df[target_col] == 1).mean())
            },
            'validation': {
                'no_sdoh_need': int((val_df[target_col] == 0).sum()),
                'sdoh_need': int((val_df[target_col] == 1).sum()),
                'prevalence': float((val_df[target_col] == 1).mean())
            },
            'test': {
                'no_sdoh_need': int((test_df[target_col] == 0).sum()),
                'sdoh_need': int((test_df[target_col] == 1).sum()),
                'prevalence': float((test_df[target_col] == 1).mean())
            }
        }
    }
    
    metadata_path = base_path / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata: {metadata_path}")
    
    # Create README
    readme_content = f"""# SDOH Prediction Model Datasets

This directory contains all dataset partitions used for the SDOH prediction model.

## Dataset Information

- **Created**: {metadata['created_date']}
- **Random Seed**: {SEED} (for reproducibility)
- **Split Ratios**: 60% train / 20% validation / 20% test
- **Target Variable**: {target_col}

## Dataset Sizes

- Full dataset: {len(full_df):,} samples
- Training set: {len(train_df):,} samples
- Validation set: {len(val_df):,} samples
- Test set: {len(test_df):,} samples

## Important Notes

1. **Race/Ethnicity Exclusion**: The model does not use race or ethnicity as features. These columns have been removed from all datasets to prevent bias.

2. **Features**: The model uses patient demographics (age, gender) and address-based social determinants (SVI/ADI from Census tract data).

3. **File Formats**: Each dataset is saved in both CSV and pickle formats:
   - CSV files for human readability and compatibility
   - Pickle files for faster loading in Python

4. **Reproducibility**: Using seed={SEED} ensures the exact same splits can be recreated.

## Target Distribution

### Full Dataset
- No SDOH need: {metadata['target_distribution']['full']['no_sdoh_need']:,} ({(1-metadata['target_distribution']['full']['prevalence'])*100:.1f}%)
- SDOH need: {metadata['target_distribution']['full']['sdoh_need']:,} ({metadata['target_distribution']['full']['prevalence']*100:.1f}%)

### Test Dataset (Used for Final Evaluation)
- No SDOH need: {metadata['target_distribution']['test']['no_sdoh_need']:,} ({(1-metadata['target_distribution']['test']['prevalence'])*100:.1f}%)
- SDOH need: {metadata['target_distribution']['test']['sdoh_need']:,} ({metadata['target_distribution']['test']['prevalence']*100:.1f}%)

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
"""
    
    readme_path = base_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"Created README: {readme_path}")

def main():
    """Main function to load, split, and save all datasets."""
    print("Starting dataset creation process...")
    print("=" * 60)
    
    # Load and preprocess data
    full_df, target_col = load_and_preprocess_data()
    
    # Create splits
    train_df, val_df, test_df = create_splits(full_df, target_col)
    
    # Save all datasets
    save_datasets(full_df, train_df, val_df, test_df, target_col)
    
    print("\n" + "=" * 60)
    print("Dataset creation completed successfully!")
    print("All datasets saved in: data/")

if __name__ == "__main__":
    main()