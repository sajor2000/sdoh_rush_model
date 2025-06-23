#!/usr/bin/env python3
"""
Refit SDOH Model with REAL DATA using proper calibration
Uses the actual SDOH dataset with 60/20/20 split
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
import time

warnings.filterwarnings('ignore')

# Set random seed
RANDOM_SEED = 2025
np.random.seed(RANDOM_SEED)

# Optimize for Apple Silicon
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'

print("🖥️  SDOH Model Training with REAL DATA")
print("=" * 60)
print(f"   CPU Cores: 16 (Apple M4 Max)")
print(f"   Random Seed: {RANDOM_SEED}")
print("=" * 60)

class CalibratedSDOHModel:
    """SDOH prediction model with calibration"""
    
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
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        ece = self.calculate_ece(y_true, y_pred_proba)
        brier = brier_score_loss(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        auprc = average_precision_score(y_true, y_pred_proba)
        
        # Calculate screening rate
        screening_rate = (y_pred == 1).mean()
        
        return {
            'auc': auc,
            'auprc': auprc,
            'ece': ece,
            'brier_score': brier,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'screening_rate': screening_rate,
            'threshold': threshold
        }
    
    def fit_with_cross_validation(self, X_train, y_train, X_val, y_val):
        """Fit model with cross-validation focused on calibration"""
        print("\n🔄 Starting cross-validation with REAL data...")
        start_time = time.time()
        
        # Parameter grid optimized for SDOH prediction
        param_combinations = []
        for _ in range(20):  # Test 20 random combinations
            params = {
                'max_depth': np.random.choice([3, 4, 5]),
                'learning_rate': np.random.choice([0.01, 0.05, 0.1]),
                'n_estimators': np.random.choice([100, 200, 300]),
                'subsample': np.random.choice([0.7, 0.8, 0.9]),
                'colsample_bytree': np.random.choice([0.7, 0.8, 0.9]),
                'min_child_weight': np.random.choice([1, 3, 5]),
                'gamma': np.random.choice([0, 0.1, 0.2]),
                'scale_pos_weight': len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle imbalance
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'seed': self.random_state,
                'tree_method': 'hist',
                'n_jobs': self.n_jobs
            }
            param_combinations.append(params)
        
        # 5-fold CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            cv_scores = []
            cv_eces = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
                
                y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                
                auc = roc_auc_score(y_fold_val, y_pred_proba)
                ece = self.calculate_ece(y_fold_val, y_pred_proba)
                
                # Weight calibration more heavily for clinical use
                combined_score = 0.6 * auc + 0.4 * (1 - ece)
                
                cv_scores.append(combined_score)
                cv_eces.append(ece)
            
            mean_score = np.mean(cv_scores)
            mean_ece = np.mean(cv_eces)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                print(f"   New best: Score={mean_score:.4f}, ECE={mean_ece:.4f}")
        
        print(f"\n✅ CV completed in {time.time() - start_time:.1f} seconds")
        
        # Train final model
        print("\n🎯 Training final model with best parameters...")
        self.base_model = xgb.XGBClassifier(**best_params)
        
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
        print(f"\n📈 Evaluating on test set ({len(y_test):,} samples)...")
        
        y_pred_base = self.base_model.predict_proba(X_test)[:, 1]
        y_pred_calibrated = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        metrics_base = self.calculate_metrics(y_test, y_pred_base)
        metrics_calibrated = self.calculate_metrics(y_test, y_pred_calibrated)
        
        self.metrics = {
            'base_model': metrics_base,
            'calibrated_model': metrics_calibrated,
            'improvement': {
                'ece': (metrics_base['ece'] - metrics_calibrated['ece']) / metrics_base['ece'] * 100 if metrics_base['ece'] > 0 else 0,
                'brier': (metrics_base['brier_score'] - metrics_calibrated['brier_score']) / metrics_base['brier_score'] * 100
            }
        }
        
        # Print results
        print("\n📊 REAL DATA Results:")
        print("=" * 70)
        print(f"Metric          | Base Model | Calibrated | Improvement")
        print("-" * 70)
        
        for metric in ['auc', 'auprc', 'ece', 'brier_score', 'sensitivity', 'specificity', 'ppv', 'screening_rate']:
            base_val = metrics_base[metric]
            cal_val = metrics_calibrated[metric]
            
            if metric in ['ece', 'brier_score']:
                improvement = f"{self.metrics['improvement'][metric.replace('_score', '')]:.1f}%"
            else:
                improvement = f"{(cal_val - base_val):.4f}"
            
            print(f"{metric:15} | {base_val:.4f}     | {cal_val:.4f}     | {improvement}")
        
        # Clinical summary
        print(f"\n🏥 Clinical Impact Summary:")
        print(f"   • Screening rate: {metrics_calibrated['screening_rate']:.1%} of patients")
        print(f"   • PPV: {metrics_calibrated['ppv']:.1%} (vs {y_test.mean():.1%} baseline)")
        print(f"   • Sensitivity: {metrics_calibrated['sensitivity']:.1%} of true positives caught")
        print(f"   • Calibration: ECE = {metrics_calibrated['ece']:.4f} {'✅ Excellent' if metrics_calibrated['ece'] < 0.05 else '⚠️  Needs improvement'}")
        
        return self.metrics
    
    def save_models(self, output_dir='models'):
        """Save models with metadata"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        base_path = os.path.join(output_dir, 'xgboost_realdata_base.json')
        self.base_model.save_model(base_path)
        
        calibrated_path = os.path.join(output_dir, 'xgboost_realdata_calibrated.joblib')
        joblib.dump(self.calibrated_model, calibrated_path)
        
        # Save comprehensive metadata
        metadata = {
            'created_date': datetime.now().isoformat(),
            'data_source': 'REAL SDOH DATA',
            'random_seed': RANDOM_SEED,
            'data_split': '60/20/20',
            'best_params': self.best_params,
            'metrics': self.metrics,
            'clinical_notes': {
                'threshold': 0.5644,
                'target_prevalence': 0.066,
                'calibration_method': 'Platt scaling'
            }
        }
        
        metadata_path = os.path.join(output_dir, 'realdata_model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"\n✅ Models saved to {output_dir}/")
        
        return base_path, calibrated_path, metadata_path


def load_real_sdoh_data():
    """Load the actual SDOH dataset"""
    print("\n📊 Loading REAL SDOH data...")
    
    # Path to the real dataset
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} samples")
    
    # Prepare features and target
    # Exclude first 3 columns as per previous analysis
    feature_cols = df.columns[3:].tolist()  # All columns after sdoh_two_yes
    
    X = df[feature_cols]
    y = df['sdoh_two_yes']  # Target variable
    
    print(f"   Features: {len(feature_cols)} columns")
    print(f"   Target prevalence: {y.mean():.1%} (expecting ~6.6%)")
    print(f"   Sample features: {feature_cols[:5]} ...")
    
    return X, y


def main():
    """Main function with real data"""
    print("🚀 SDOH Model Training - REAL DATA")
    print("=" * 60)
    
    overall_start = time.time()
    
    # Load REAL data
    X, y = load_real_sdoh_data()
    
    # Create 60/20/20 split
    print("\n🔀 Creating 60/20/20 split...")
    
    # First split: 80/20 for (train+cal)/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    
    # Second split: 75/25 of the 80% for train/cal (= 60/20 of total)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"   Training: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   Calibration: {len(X_cal):,} samples ({len(X_cal)/len(X)*100:.1f}%)")
    print(f"   Test: {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"   Prevalence - Train: {y_train.mean():.2%}, Cal: {y_cal.mean():.2%}, Test: {y_test.mean():.2%}")
    
    # Initialize and train model
    model = CalibratedSDOHModel(random_state=RANDOM_SEED)
    
    # Fit with cross-validation
    model.fit_with_cross_validation(X_train, y_train, X_cal, y_cal)
    
    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test)
    
    # Save models
    model.save_models()
    
    total_time = time.time() - overall_start
    
    print("\n" + "=" * 60)
    print("🎉 REAL DATA MODEL TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n🎯 Key Results with REAL DATA:")
    print(f"   • AUC: {metrics['calibrated_model']['auc']:.4f}")
    print(f"   • ECE: {metrics['calibrated_model']['ece']:.4f}")
    print(f"   • PPV: {metrics['calibrated_model']['ppv']:.4f}")
    print(f"   • Sensitivity: {metrics['calibrated_model']['sensitivity']:.4f}")
    
    print("\n📋 This is your production-ready model trained on REAL patient data!")

if __name__ == "__main__":
    main()