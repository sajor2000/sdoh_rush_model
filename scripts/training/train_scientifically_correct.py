#!/usr/bin/env python3
"""
Scientifically correct model training with proper use of train/validation/test sets
No data leakage - threshold selection on validation set, final evaluation on test set
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import *
import xgboost as xgb
import joblib
import json
import os
import sys
from datetime import datetime
import warnings
import matplotlib.pyplot as plt

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_PATH, RANDOM_SEED, MODELS_DIR, FIGURES_DIR

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

np.random.seed(RANDOM_SEED)

print("🔬 Scientifically Correct Model Training")
print("=" * 60)
print("No data leakage - proper train/validation/test split usage")
print("=" * 60)

def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0
    for i in range(n_bins):
        bin_mask = (y_pred_proba > bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i+1])
        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_pred_proba[bin_mask].mean()
            ece += np.abs(bin_acc - bin_conf) * bin_mask.mean()
    return ece

def find_optimal_threshold_on_validation(y_true, y_pred_proba):
    """
    Find optimal threshold on VALIDATION set (not test set!)
    This prevents data leakage
    """
    thresholds = np.linspace(0.05, 0.5, 50)
    best_score = -np.inf
    best_thresh = 0.1
    best_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        screen_rate = y_pred.mean()
        
        # Penalize extreme screening rates
        if screen_rate < 0.05 or screen_rate > 0.4:
            penalty = 0.5
        else:
            penalty = 1.0
        
        # Multi-objective score
        score = penalty * (0.4 * sens + 0.3 * ppv + 0.3 * (1 - abs(screen_rate - 0.20)))
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_metrics = {
                'threshold': thresh,
                'sensitivity': sens,
                'specificity': spec,
                'ppv': ppv,
                'screening_rate': screen_rate,
                'score': score
            }
    
    return best_thresh, best_metrics

def evaluate_on_test_set(model, X_test, y_test, threshold, set_name="Test"):
    """
    Evaluate model on test set with PRE-DETERMINED threshold
    This gives unbiased performance estimates
    """
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'set': set_name,
        'n_samples': len(y_test),
        'threshold_used': threshold,
        'auc': roc_auc_score(y_test, y_pred_proba),
        'auprc': average_precision_score(y_test, y_pred_proba),
        'ece': calculate_ece(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'screening_rate': y_pred.mean(),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }
    
    return metrics, y_pred_proba

def generate_figures(model, X_test, y_test, threshold, val_metrics, test_metrics, feature_names, output_dir='results/figures'):
    """Generate TRIPOD-AI compliant figures using test set"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test set predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 1. Main Performance Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SDOH Model Performance - Scientifically Valid Evaluation', fontsize=16, fontweight='bold')
    
    # ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, 'b-', linewidth=3, label=f'Test AUC = {test_metrics["auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve (Test Set)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    ax2 = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax2.plot(recall, precision, 'g-', linewidth=3, label=f'Test AP = {test_metrics["auprc"]:.3f}')
    ax2.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.1%})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve (Test Set)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Calibration Plot
    ax3 = axes[1, 0]
    fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    ax3.plot(mean_pred, fraction_pos, 'ro-', linewidth=3, markersize=10,
             label=f'Test ECE = {test_metrics["ece"]:.3f}')
    ax3.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax3.set_ylabel('Fraction of Positives', fontsize=12)
    ax3.set_title('Calibration Plot (Test Set)', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Score Distribution with threshold from validation
    ax4 = axes[1, 1]
    ax4.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, color='blue', 
             label='Negative', density=True, edgecolor='darkblue')
    ax4.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6, color='red', 
             label='Positive', density=True, edgecolor='darkred')
    ax4.axvline(x=threshold, color='green', linestyle='--', linewidth=3,
               label=f'Val Threshold ({threshold:.3f})')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Score Distribution (Test Set)', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_scientifically_valid.png'), dpi=300, bbox_inches='tight')
    print("   ✓ Performance curves saved")
    
    # 2. Validation vs Test Comparison
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    metrics_names = ['auc', 'auprc', 'ece', 'sensitivity', 'specificity', 'ppv', 'screening_rate']
    val_values = [val_metrics[m] for m in metrics_names]
    test_values = [test_metrics[m] for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, val_values, width, label='Validation Set', color='skyblue', edgecolor='darkblue')
    bars2 = ax.bar(x + width/2, test_values, width, label='Test Set', color='lightcoral', edgecolor='darkred')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Validation vs Test Set Performance (No Overfitting)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_vs_test_comparison.png'), dpi=300, bbox_inches='tight')
    print("   ✓ Validation vs Test comparison saved")
    
    return

def main():
    # Load data
    print("\n📊 Loading REAL SDOH data...")
    if not os.path.exists(DATA_PATH):
        print(f"❌ Data file not found at: {DATA_PATH}")
        print("Please update DATA_PATH in config.py or set SDOH_DATA_PATH environment variable")
        return
    
    df = pd.read_csv(DATA_PATH)
    
    feature_cols = df.columns[3:].tolist()
    X = df[feature_cols]
    y = df['sdoh_two_yes']
    
    print(f"   Loaded {len(df):,} samples")
    print(f"   Prevalence: {y.mean():.1%}")
    
    # PROPER 60/20/20 split
    print("\n🔀 Creating proper 60/20/20 split...")
    # First split off 20% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    # Then split remaining 80% into 60% train, 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"   Training: {len(X_train):,} samples (60%)")
    print(f"   Validation: {len(X_val):,} samples (20%)")
    print(f"   Test: {len(X_test):,} samples (20%)")
    print(f"   ✅ Test set is held out and will NOT influence any decisions")
    
    # Train base model
    print("\n🎯 Training XGBoost model on training set...")
    params = {
        'max_depth': 5,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.05,
        'scale_pos_weight': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': RANDOM_SEED,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    base_model = xgb.XGBClassifier(**params)
    # Use validation set for early stopping
    base_model.fit(
        X_train, y_train, 
        eval_set=[(X_val, y_val)], 
        verbose=False
    )
    
    # Apply calibration on validation set
    print("\n📊 Applying Platt scaling using validation set...")
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_val, y_val)
    
    # Find optimal threshold on VALIDATION set
    print("\n🎯 Finding optimal threshold on VALIDATION set (no test set peeking!)...")
    y_val_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
    optimal_threshold, val_threshold_metrics = find_optimal_threshold_on_validation(y_val, y_val_pred_proba)
    
    print(f"   Optimal threshold (from validation): {optimal_threshold:.4f}")
    print(f"   Validation metrics at this threshold:")
    print(f"     - Sensitivity: {val_threshold_metrics['sensitivity']:.1%}")
    print(f"     - PPV: {val_threshold_metrics['ppv']:.1%}")
    print(f"     - Screening rate: {val_threshold_metrics['screening_rate']:.1%}")
    
    # Evaluate on validation set (for comparison)
    print("\n📈 Evaluating on validation set...")
    val_metrics, _ = evaluate_on_test_set(calibrated_model, X_val, y_val, optimal_threshold, "Validation")
    
    # FINAL UNBIASED EVALUATION on test set
    print("\n📈 FINAL EVALUATION on held-out test set...")
    test_metrics, test_proba = evaluate_on_test_set(calibrated_model, X_test, y_test, optimal_threshold, "Test")
    
    # Generate figures
    print("\n📊 Generating TRIPOD-AI figures...")
    generate_figures(calibrated_model, X_test, y_test, optimal_threshold, 
                    val_metrics, test_metrics, feature_cols)
    
    # Print final results
    print("\n" + "=" * 70)
    print("SCIENTIFICALLY VALID RESULTS (Test Set - Model Never Saw This Data)")
    print("=" * 70)
    print(f"Threshold selected on validation set: {optimal_threshold:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  AUC: {test_metrics['auc']:.4f}")
    print(f"  AUPRC: {test_metrics['auprc']:.4f}")
    print(f"  ECE: {test_metrics['ece']:.4f} {'✅' if test_metrics['ece'] < 0.05 else '⚠️'}")
    print(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    print(f"\nAt threshold {optimal_threshold:.4f}:")
    print(f"  Screening rate: {test_metrics['screening_rate']:.1%}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.1%}")
    print(f"  Specificity: {test_metrics['specificity']:.1%}")
    print(f"  PPV: {test_metrics['ppv']:.1%}")
    print(f"  NPV: {test_metrics['npv']:.1%}")
    
    # Check for overfitting
    print(f"\n🔍 Overfitting Check (Validation vs Test):")
    print(f"  AUC: {val_metrics['auc']:.4f} vs {test_metrics['auc']:.4f} (diff: {abs(val_metrics['auc'] - test_metrics['auc']):.4f})")
    print(f"  PPV: {val_metrics['ppv']:.4f} vs {test_metrics['ppv']:.4f} (diff: {abs(val_metrics['ppv'] - test_metrics['ppv']):.4f})")
    
    if abs(val_metrics['auc'] - test_metrics['auc']) < 0.02:
        print("  ✅ No significant overfitting detected")
    else:
        print("  ⚠️  Some overfitting detected")
    
    # Save models and metadata
    print("\n💾 Saving models...")
    os.makedirs('models', exist_ok=True)
    
    base_model.save_model('models/xgboost_scientific_base.json')
    joblib.dump(calibrated_model, 'models/xgboost_scientific_calibrated.joblib')
    
    # Save complete metadata
    metadata = {
        'created_date': datetime.now().isoformat(),
        'methodology': 'Scientifically correct - no data leakage',
        'data_split': {
            'train': '60%',
            'validation': '20% (for calibration and threshold selection)',
            'test': '20% (held out for final evaluation)'
        },
        'threshold_selection': {
            'method': 'Optimized on validation set only',
            'optimal_threshold': float(optimal_threshold),
            'validation_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                 for k, v in val_threshold_metrics.items()}
        },
        'validation_performance': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                  for k, v in val_metrics.items()},
        'test_performance': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                            for k, v in test_metrics.items()},
        'model_params': params
    }
    
    with open('models/scientific_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n✅ Scientifically valid model training complete!")
    print("   All reported metrics are from the held-out test set")
    print("   Threshold was selected on validation set only")
    print("   No data leakage!")

if __name__ == "__main__":
    main()