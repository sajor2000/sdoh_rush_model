#!/usr/bin/env python3
"""
Fast version - Train balanced calibrated SDOH model with TRIPOD figures
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
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

print("🚀 Fast Balanced Model Training")
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

def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold balancing multiple objectives"""
    thresholds = np.linspace(0.05, 0.5, 50)
    best_score = -np.inf
    best_thresh = 0.1
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        screen_rate = y_pred.mean()
        
        # Penalize extreme screening rates
        if screen_rate < 0.05 or screen_rate > 0.4:
            penalty = 0.5
        else:
            penalty = 1.0
        
        score = penalty * (0.5 * sens + 0.3 * ppv + 0.2 * (1 - abs(screen_rate - 0.15)))
        
        if score > best_score:
            best_score = score
            best_thresh = thresh
    
    return best_thresh

def generate_tripod_figures(model, X_test, y_test, feature_names, output_dir='results/figures'):
    """Generate all TRIPOD-AI figures"""
    os.makedirs(output_dir, exist_ok=True)
    print("\n📊 Generating TRIPOD-AI figures...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    optimal_threshold = find_optimal_threshold(y_test, y_pred_proba)
    
    # 1. Main Performance Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Model Performance Analysis - Balanced Calibration', fontsize=16, fontweight='bold')
    
    # ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    ax1.plot(fpr, tpr, 'b-', linewidth=3, label=f'AUC = {auc:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # PR Curve
    ax2 = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    ax2.plot(recall, precision, 'g-', linewidth=3, label=f'AP = {ap:.3f}')
    ax2.axhline(y=y_test.mean(), color='r', linestyle='--', label=f'Baseline ({y_test.mean():.1%})')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Calibration Plot
    ax3 = axes[1, 0]
    fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    ece = calculate_ece(y_test, y_pred_proba)
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect', linewidth=2)
    ax3.plot(mean_pred, fraction_pos, 'ro-', linewidth=3, markersize=10,
             label=f'Model (ECE={ece:.3f})')
    ax3.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax3.set_ylabel('Fraction of Positives', fontsize=12)
    ax3.set_title('Calibration Plot', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Score Distribution
    ax4 = axes[1, 1]
    ax4.hist(y_pred_proba[y_test==0], bins=30, alpha=0.6, color='blue', 
             label='Negative', density=True, edgecolor='darkblue')
    ax4.hist(y_pred_proba[y_test==1], bins=30, alpha=0.6, color='red', 
             label='Positive', density=True, edgecolor='darkred')
    ax4.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=3,
               label=f'Threshold ({optimal_threshold:.3f})')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_balanced_final.png'), dpi=300, bbox_inches='tight')
    print("   ✓ Performance curves saved")
    
    # 2. Feature Importance
    if hasattr(model, 'base_estimator'):
        base_model = model.base_estimator
    else:
        base_model = model
    
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    importance = base_model.base_estimator.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(15)
    
    ax.barh(importance_df['feature'], importance_df['importance'], color='skyblue', edgecolor='darkblue')
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'], i, f'{row["importance"]:.3f}', 
                va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance_final.png'), dpi=300, bbox_inches='tight')
    print("   ✓ Feature importance saved")
    
    # 3. Threshold Analysis
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
    
    thresholds = np.linspace(0.05, 0.5, 50)
    metrics = {'sensitivity': [], 'ppv': [], 'screening_rate': []}
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        screen_rate = y_pred.mean()
        
        metrics['sensitivity'].append(sens)
        metrics['ppv'].append(ppv)
        metrics['screening_rate'].append(screen_rate)
    
    # Sensitivity vs PPV trade-off
    ax1 = axes3[0]
    ax1.plot(metrics['ppv'], metrics['sensitivity'], 'b-', linewidth=3)
    ax1.scatter([metrics['ppv'][np.argmin(np.abs(thresholds - optimal_threshold))]], 
                [metrics['sensitivity'][np.argmin(np.abs(thresholds - optimal_threshold))]], 
                color='red', s=200, zorder=5, label=f'Optimal ({optimal_threshold:.3f})')
    ax1.set_xlabel('PPV (Precision)', fontsize=12)
    ax1.set_ylabel('Sensitivity (Recall)', fontsize=12)
    ax1.set_title('Sensitivity vs PPV Trade-off', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Screening Rate vs Sensitivity
    ax2 = axes3[1]
    ax2.plot(metrics['screening_rate'], metrics['sensitivity'], 'g-', linewidth=3)
    ax2.scatter([metrics['screening_rate'][np.argmin(np.abs(thresholds - optimal_threshold))]], 
                [metrics['sensitivity'][np.argmin(np.abs(thresholds - optimal_threshold))]], 
                color='red', s=200, zorder=5)
    ax2.set_xlabel('Screening Rate', fontsize=12)
    ax2.set_ylabel('Sensitivity', fontsize=12)
    ax2.set_title('Screening Rate vs Sensitivity', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis_final.png'), dpi=300, bbox_inches='tight')
    print("   ✓ Threshold analysis saved")
    
    return optimal_threshold

def main():
    # Load data
    print("\n📊 Loading REAL SDOH data...")
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    df = pd.read_csv(data_path)
    
    feature_cols = df.columns[3:].tolist()
    X = df[feature_cols]
    y = df['sdoh_two_yes']
    
    print(f"   Loaded {len(df):,} samples")
    print(f"   Prevalence: {y.mean():.1%}")
    
    # Split data 60/20/20
    print("\n🔀 Creating 60/20/20 split...")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_SEED)
    
    # Train base model with parameters that encourage less conservative predictions
    print("\n🎯 Training balanced XGBoost model...")
    params = {
        'max_depth': 5,  # Deeper trees for more complex patterns
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.05,
        'scale_pos_weight': 0.8,  # Slightly reduce weight to avoid over-conservatism
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'seed': RANDOM_SEED,
        'tree_method': 'hist',
        'n_jobs': -1
    }
    
    base_model = xgb.XGBClassifier(**params)
    base_model.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=False)
    
    # Apply calibration
    print("\n📊 Applying Platt scaling...")
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)
    
    # Generate figures
    optimal_threshold = generate_tripod_figures(calibrated_model, X_test, y_test, feature_cols)
    
    # Evaluate
    print("\n📈 Final evaluation...")
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Metrics at optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp)
    screen_rate = y_pred.mean()
    
    # Overall metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    ece = calculate_ece(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    print("\n" + "=" * 60)
    print("BALANCED MODEL RESULTS")
    print("=" * 60)
    print(f"AUC: {auc:.4f}")
    print(f"AUPRC: {auprc:.4f}")
    print(f"ECE: {ece:.4f} {'✅' if ece < 0.05 else '⚠️'}")
    print(f"Brier Score: {brier:.4f}")
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"Screening rate: {screen_rate:.1%}")
    print(f"Sensitivity: {sens:.1%}")
    print(f"Specificity: {spec:.1%}")
    print(f"PPV: {ppv:.1%}")
    
    # Save models
    print("\n💾 Saving models...")
    os.makedirs('models', exist_ok=True)
    
    base_model.save_model('models/xgboost_balanced_final.json')
    joblib.dump(calibrated_model, 'models/xgboost_balanced_calibrated_final.joblib')
    
    # Save metadata
    metadata = {
        'created_date': datetime.now().isoformat(),
        'model_type': 'Balanced Calibrated XGBoost',
        'data_split': '60/20/20',
        'optimal_threshold': float(optimal_threshold),
        'metrics': {
            'auc': float(auc),
            'auprc': float(auprc),
            'ece': float(ece),
            'brier_score': float(brier),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'ppv': float(ppv),
            'screening_rate': float(screen_rate)
        },
        'parameters': params
    }
    
    with open('models/balanced_final_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print("\n✅ Complete! Model and figures saved.")

if __name__ == "__main__":
    main()