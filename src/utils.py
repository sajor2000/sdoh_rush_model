"""
Utility functions for SDOH Risk Screening Model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve

def calculate_expected_calibration_error(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_performance_metrics(y_true, y_pred_proba, threshold=0.05):
    """Calculate comprehensive performance metrics"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Basic metrics
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    
    # Calculated metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # Performance metrics
    auc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    ece = calculate_expected_calibration_error(y_true, y_pred_proba)
    brier_score = brier_score_loss(y_true, y_pred_proba)
    
    return {
        'auc': auc,
        'auprc': auprc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'ece': ece,
        'brier_score': brier_score,
        'screening_rate': y_pred.mean(),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }

def load_test_data(data_path, test_size=0.2, random_seed=2025):
    """Load and split data maintaining same test set as training"""
    df = pd.read_csv(data_path)
    
    # Use same methodology as training script
    np.random.seed(random_seed)
    n_total = len(df)
    n_test = int(n_total * test_size)
    test_indices = df.index[-n_test:]
    
    return df.iloc[test_indices].copy()
