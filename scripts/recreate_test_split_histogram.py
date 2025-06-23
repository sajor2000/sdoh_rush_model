#!/usr/bin/env python3
"""
Recreate the exact test split using the same seed and methodology
from train_scientifically_correct.py to generate real risk probability histogram
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# JAMA publication specifications
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.5

# CRITICAL: Use the exact same seed as in training
SEED = 2025

def recreate_test_split_and_histogram():
    """Recreate exact test split and generate histogram from real predictions"""
    
    print("Recreating exact test split from training...")
    print(f"Using seed: {SEED}")
    
    # Load the trained model
    model = joblib.load('models/xgboost_scientific_calibrated.joblib')
    print("✓ Model loaded successfully")
    
    # Load the original data
    print("\nLoading original dataset...")
    try:
        # Try to find the original data file
        data_paths = [
            '../sdoh2_ml_final_all_svi.csv',
            '../../sdoh2_ml_final_all_svi.csv',
            '../../../sdoh2_ml_final_all_svi.csv',
            'data/sdoh2_ml_final_all_svi.csv'
        ]
        
        df = None
        for path in data_paths:
            try:
                df = pd.read_csv(path)
                print(f"✓ Data loaded from: {path}")
                print(f"  Shape: {df.shape}")
                break
            except:
                continue
        
        if df is None:
            raise FileNotFoundError("Cannot find original data file")
        
        # Recreate the exact preprocessing from training script
        # Identify target column
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
        
        # Handle remaining categorical columns - convert to numeric
        categorical_cols = df.select_dtypes(include=['object']).columns
        if target_col in categorical_cols:
            categorical_cols = categorical_cols.drop(target_col)
        
        # Convert categorical to dummy variables
        if len(categorical_cols) > 0:
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Separate features and target
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        print(f"\nTarget distribution:")
        print(f"  Positive cases: {y.sum():,} ({y.mean()*100:.1f}%)")
        
        # CRITICAL: Recreate exact same splits with same seed
        # First split: 80% train+val, 20% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y
        )
        
        # Second split: 75% train (60% of total), 25% val (20% of total)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=SEED, stratify=y_temp
        )
        
        print(f"\nSplit sizes (matching training):")
        print(f"  Train: {len(X_train):,} samples")
        print(f"  Val: {len(X_val):,} samples")
        print(f"  Test: {len(X_test):,} samples")
        
        # Verify test size matches metadata
        assert len(X_test) == 78745, f"Test size mismatch: expected 78,745, got {len(X_test):,}"
        
        # Get predictions from the calibrated model
        print("\nGenerating predictions on test set...")
        probabilities = model.predict_proba(X_test)[:, 1]
        
        # Verify metrics match
        from sklearn.metrics import roc_auc_score, average_precision_score
        test_auc = roc_auc_score(y_test, probabilities)
        test_ap = average_precision_score(y_test, probabilities)
        
        print(f"\nTest metrics (should match metadata):")
        print(f"  AUC: {test_auc:.3f} (expected: 0.765)")
        print(f"  AP: {test_ap:.3f} (expected: 0.211)")
        
        # Calculate threshold-based metrics
        threshold = 0.05
        predictions = (probabilities >= threshold).astype(int)
        tp = ((predictions == 1) & (y_test == 1)).sum()
        fp = ((predictions == 1) & (y_test == 0)).sum()
        tn = ((predictions == 0) & (y_test == 0)).sum()
        fn = ((predictions == 0) & (y_test == 1)).sum()
        
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        screening_rate = (predictions == 1).mean()
        
        print(f"\nThreshold-based metrics at {threshold}:")
        print(f"  Sensitivity: {sensitivity:.3f} (expected: 0.722)")
        print(f"  Specificity: {specificity:.3f} (expected: 0.679)")
        print(f"  PPV: {ppv:.3f} (expected: 0.138)")
        print(f"  Screening rate: {screening_rate:.3f} (expected: 0.348)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCannot proceed without the original data file.")
        print("Please ensure sdoh2_ml_final_all_svi.csv is accessible.")
        return
    
    # Create the histogram with REAL data
    print("\nCreating histogram from real test predictions...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top panel: Overall distribution
    n, bins, patches = ax1.hist(probabilities, bins=50, alpha=0.8, 
                               color='#3498DB', edgecolor='black', linewidth=0.5)
    
    # Add threshold lines
    ax1.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='General Threshold (5%)')
    ax1.axvline(x=0.084, color='orange', linestyle='--', linewidth=2, label='Geriatric Threshold (8.4%)')
    
    # Add statistics
    median_prob = np.median(probabilities)
    mean_prob = np.mean(probabilities)
    ax1.axvline(x=median_prob, color='green', linestyle='-', linewidth=2, label=f'Median ({median_prob:.3f})')
    
    ax1.set_xlabel('Predicted Risk Probability', fontsize=14)
    ax1.set_ylabel('Number of Patients', fontsize=14)
    ax1.set_title(f'Distribution of SDOH Risk Scores in Test Dataset (n={len(probabilities):,})', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics box with REAL data
    stats_text = f'Total Patients: {len(probabilities):,}\n'
    stats_text += f'Positive Cases: {int(y_test.sum()):,} ({y_test.mean()*100:.1f}%)\n'
    stats_text += f'Screen at 5%: {(probabilities >= 0.05).sum():,} ({(probabilities >= 0.05).mean()*100:.1f}%)\n'
    stats_text += f'Screen at 8.4%: {(probabilities >= 0.084).sum():,} ({(probabilities >= 0.084).mean()*100:.1f}%)\n'
    stats_text += f'AUC: {test_auc:.3f}'
    
    ax1.text(0.98, 0.97, stats_text, transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             verticalalignment='top', horizontalalignment='right', fontsize=11)
    
    # Bottom panel: Stratified by outcome
    # Determine appropriate bin range based on data
    max_prob_display = np.percentile(probabilities, 99)  # Show up to 99th percentile
    bins2 = np.linspace(0, max_prob_display, 31)
    
    prob_positive = probabilities[y_test == 1]
    prob_negative = probabilities[y_test == 0]
    
    ax2.hist(prob_negative, bins=bins2, alpha=0.6, color='#3498DB', 
             edgecolor='black', linewidth=0.5, label='No SDOH Need', density=True)
    ax2.hist(prob_positive, bins=bins2, alpha=0.6, color='#E74C3C', 
             edgecolor='black', linewidth=0.5, label='SDOH Need', density=True)
    
    ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=0.084, color='orange', linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('Predicted Risk Probability', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title('Risk Score Distribution by Actual SDOH Status', fontsize=16, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max_prob_display)
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'results/figures/risk_probability_histogram_test_data.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    # Save to JAMA figures
    jama_path = 'results/figures/jama/figure5_risk_distribution.png'
    plt.savefig(jama_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(jama_path.replace('.png', '.pdf'), bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"\n✅ Real test data histogram saved to:")
    print(f"   - {output_path}")
    print(f"   - {jama_path}")
    
    # Create simple clinical version
    plt.figure(figsize=(10, 6))
    
    n, bins, patches = plt.hist(probabilities, bins=50, alpha=0.8, 
                               edgecolor='black', linewidth=0.5)
    
    # Color bars by risk level
    for i, patch in enumerate(patches):
        if bins[i] >= 0.084:
            patch.set_facecolor('#E74C3C')  # Red for high risk
        elif bins[i] >= 0.05:
            patch.set_facecolor('#F39C12')  # Orange for medium risk
        else:
            patch.set_facecolor('#27AE60')  # Green for low risk
    
    plt.axvline(x=0.05, color='black', linestyle='--', linewidth=2, label='General Screening (5%)')
    plt.axvline(x=0.084, color='black', linestyle=':', linewidth=2, label='Geriatric Screening (8.4%)')
    
    plt.xlabel('Predicted SDOH Risk Probability', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Patients', fontsize=14, fontweight='bold')
    plt.title('SDOH Risk Score Distribution - Clinical Thresholds', 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle=':')
    
    # Add count annotations based on real data
    low_count = (probabilities < 0.05).sum()
    med_count = ((probabilities >= 0.05) & (probabilities < 0.084)).sum()
    high_count = (probabilities >= 0.084).sum()
    
    plt.text(0.98, 0.97, f'Low Risk: {low_count:,} patients ({low_count/len(probabilities)*100:.1f}%)\nMedium Risk: {med_count:,} patients ({med_count/len(probabilities)*100:.1f}%)\nHigh Risk: {high_count:,} patients ({high_count/len(probabilities)*100:.1f}%)', 
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9),
             verticalalignment='top', horizontalalignment='right', fontsize=10)
    
    plt.xlim(0, max_prob_display)
    plt.tight_layout()
    
    simple_path = 'results/figures/risk_probability_histogram_clinical.png'
    plt.savefig(simple_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"   - {simple_path} (clinical version)")
    
    # Print detailed statistics
    print(f"\nDetailed Risk Distribution Statistics (REAL DATA):")
    print(f"  Minimum probability: {probabilities.min():.4f}")
    print(f"  5th percentile: {np.percentile(probabilities, 5):.4f}")
    print(f"  25th percentile: {np.percentile(probabilities, 25):.4f}")
    print(f"  Median: {median_prob:.4f}")
    print(f"  Mean: {mean_prob:.4f}")
    print(f"  75th percentile: {np.percentile(probabilities, 75):.4f}")
    print(f"  95th percentile: {np.percentile(probabilities, 95):.4f}")
    print(f"  Maximum probability: {probabilities.max():.4f}")
    
    plt.close('all')

if __name__ == "__main__":
    recreate_test_split_and_histogram()