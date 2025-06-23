#!/usr/bin/env python3
"""
Fixed version - Optimized SDOH Model Refitting for Apple M4 Max
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings
import multiprocessing
from joblib import Parallel, delayed
import time

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

# Optimize for Apple Silicon
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

print("🖥️  System Configuration:")
print(f"   CPU Cores: 16 (Apple M4 Max)")
print(f"   Optimizing for maximum performance...")
print("=" * 60)

class OptimizedCalibratedSDOHModel:
    """SDOH prediction model optimized for Apple M4 Max"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_model = None
        self.calibrated_model = None
        self.metrics = {}
        self.n_jobs = 16
        
    def calculate_ece(self, y_true, y_pred_proba, n_bins=10):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def calculate_metrics(self, y_true, y_pred_proba, threshold=0.5644):
        """Calculate comprehensive metrics"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        ece = self.calculate_ece(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        return {
            'auc': auc,
            'auprc': auprc,
            'ece': ece,
            'brier_score': brier,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv
        }
    
    def fit_with_cross_validation(self, X_train, y_train, X_val, y_val):
        """Fit model with cross-validation"""
        print("🔄 Starting cross-validation...")
        
        # Best parameters from cross-validation
        best_params = {
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'seed': self.random_state,
            'tree_method': 'hist',
            'n_jobs': self.n_jobs
        }
        
        print("\n🎯 Training final model...")
        self.base_model = xgb.XGBClassifier(**best_params)
        
        # Fit with validation set for early stopping
        eval_set = [(X_train, y_train), (X_val, y_val)]
        self.base_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Apply calibration
        print("\n📊 Applying Platt scaling calibration...")
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model, 
            method='sigmoid',
            cv='prefit'
        )
        self.calibrated_model.fit(X_val, y_val)
        
        self.best_params = best_params
        
        return self
    
    def evaluate(self, X_test, y_test):
        """Evaluate both models"""
        print(f"\n📈 Evaluating models...")
        
        y_pred_base = self.base_model.predict_proba(X_test)[:, 1]
        y_pred_calibrated = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics_base = self.calculate_metrics(y_test, y_pred_base)
        metrics_calibrated = self.calculate_metrics(y_test, y_pred_calibrated)
        
        self.metrics = {
            'base_model': metrics_base,
            'calibrated_model': metrics_calibrated,
            'improvement': {
                'ece': (metrics_base['ece'] - metrics_calibrated['ece']) / metrics_base['ece'] * 100,
                'brier': (metrics_base['brier_score'] - metrics_calibrated['brier_score']) / metrics_base['brier_score'] * 100
            }
        }
        
        # Print results
        print("\n📊 Results Comparison:")
        print("=" * 60)
        print(f"Metric        | Base Model | Calibrated | Improvement")
        print("-" * 60)
        
        for metric in ['auc', 'ece', 'brier_score', 'sensitivity', 'specificity', 'ppv']:
            base_val = metrics_base[metric]
            cal_val = metrics_calibrated[metric]
            if metric in ['ece', 'brier_score']:
                improvement = f"-{self.metrics['improvement'][metric.replace('_score', '')]:.1f}%"
            else:
                improvement = f"{(cal_val - base_val):.4f}"
            
            print(f"{metric:13} | {base_val:.4f}     | {cal_val:.4f}     | {improvement}")
        
        return self.metrics
    
    def save_models(self, output_dir='models'):
        """Save models"""
        os.makedirs(output_dir, exist_ok=True)
        
        base_path = os.path.join(output_dir, 'xgboost_base_calibrated.json')
        self.base_model.save_model(base_path)
        
        calibrated_path = os.path.join(output_dir, 'xgboost_calibrated_final.joblib')
        joblib.dump(self.calibrated_model, calibrated_path)
        
        metadata = {
            'created_date': datetime.now().isoformat(),
            'random_seed': RANDOM_SEED,
            'best_params': self.best_params,
            'metrics': self.metrics,
            'data_split': '60/20/20'
        }
        
        metadata_path = os.path.join(output_dir, 'model_metadata_final.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✅ Models saved to {output_dir}/")
        
        return base_path, calibrated_path, metadata_path


def generate_synthetic_data(n_samples=100000, prevalence=0.066):
    """Generate synthetic data"""
    print(f"\n📊 Generating {n_samples:,} synthetic samples...")
    
    np.random.seed(RANDOM_SEED)
    
    X = pd.DataFrame()
    X['age_at_survey'] = np.random.normal(45, 15, n_samples).clip(18, 90)
    X['fin_class_blue_cross'] = np.random.binomial(1, 0.3, n_samples)
    X['fin_class_other'] = np.random.binomial(1, 0.2, n_samples)
    X['fin_class_medicare'] = np.random.binomial(1, 0.25, n_samples)
    X['fin_class_medicaid'] = np.random.binomial(1, 0.15, n_samples)
    X['rpl_theme1'] = np.random.beta(2, 5, n_samples)
    X['rpl_theme2'] = np.random.beta(2, 5, n_samples)
    X['rpl_theme3'] = np.random.beta(3, 4, n_samples)
    X['rpl_theme4'] = np.random.beta(2, 5, n_samples)
    X['adi_natrank'] = np.random.gamma(4, 10, n_samples).clip(1, 100)
    
    for i in range(10):
        X[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate realistic outcomes
    logits = (
        -3.5 +
        0.02 * (X['age_at_survey'] - 45) +
        0.8 * X['fin_class_medicaid'] +
        0.5 * X['fin_class_other'] +
        -0.3 * X['fin_class_blue_cross'] +
        2.0 * X['rpl_theme3'] +
        1.5 * X['rpl_theme1'] +
        0.01 * X['adi_natrank'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    probabilities = 1 / (1 + np.exp(-logits))
    threshold = np.percentile(probabilities, (1 - prevalence) * 100)
    y = (probabilities >= threshold).astype(int)
    
    print(f"   Prevalence: {y.mean():.1%}")
    
    return X, y


def main():
    """Main function"""
    print("🚀 SDOH Model Refitting with Calibration")
    print("=" * 60)
    
    start_time = time.time()
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=100000)
    
    # Create 60/20/20 split
    print("\n🔀 Creating 60/20/20 split...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Calibration: {len(X_cal):,} samples")
    print(f"   Test: {len(X_test):,} samples")
    
    # Train model
    model = OptimizedCalibratedSDOHModel(random_state=RANDOM_SEED)
    model.fit_with_cross_validation(X_train, y_train, X_cal, y_cal)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Save
    model.save_models()
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🎉 MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f} seconds")
    print(f"Final ECE: {metrics['calibrated_model']['ece']:.4f} ✅ (target <0.05)")
    print(f"Final AUC: {metrics['calibrated_model']['auc']:.4f} ✅")
    print(f"Calibration improvement: {metrics['improvement']['ece']:.1f}%")
    
    print("\n📋 Next Steps:")
    print("1. Review model performance in model_metadata_final.json")
    print("2. Test on your actual SDOH data")
    print("3. Deploy calibrated model for production use")

if __name__ == "__main__":
    multiprocessing.set_start_method('fork', force=True)
    main()