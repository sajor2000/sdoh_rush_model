#!/usr/bin/env python3
"""
Model Calibration Script
=======================

This script applies calibration methods (Platt scaling and Isotonic regression)
to improve the XGBoost model's probability estimates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model_evaluation import SDOHPredictor


class XGBoostClassifierWrapper:
    """Wrapper to make XGBoost Booster compatible with scikit-learn's CalibratedClassifierCV."""
    
    def __init__(self, booster, feature_names=None):
        self.booster = booster
        self.feature_names = feature_names
        self.classes_ = np.array([0, 1])
        
    def predict_proba(self, X):
        """Predict probabilities."""
        if isinstance(X, pd.DataFrame) and self.feature_names:
            X = X[self.feature_names]
        dmatrix = xgb.DMatrix(X)
        pos_proba = self.booster.predict(dmatrix)
        # Return probabilities for both classes
        return np.vstack([1 - pos_proba, pos_proba]).T
    
    def predict(self, X):
        """Predict classes."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def fit(self, X, y):
        """Dummy fit method for compatibility."""
        return self


def calculate_ece(y_true, y_prob, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_prob : array-like
        Predicted probabilities
    n_bins : int
        Number of bins for calibration
        
    Returns
    -------
    float
        Expected calibration error
    """
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


def plot_calibration_comparison(y_true, prob_uncalibrated, prob_platt, prob_isotonic, save_path):
    """
    Create calibration plots comparing uncalibrated and calibrated models.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    prob_uncalibrated : array-like
        Uncalibrated probabilities
    prob_platt : array-like
        Platt-scaled probabilities
    prob_isotonic : array-like
        Isotonic regression probabilities
    save_path : Path
        Where to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Calculate calibration curves
    n_bins = 10
    
    # Plot for each calibration method
    for idx, (probs, title, ax) in enumerate([
        (prob_uncalibrated, 'Uncalibrated XGBoost', axes[0]),
        (prob_platt, 'Platt Scaling', axes[1]),
        (prob_isotonic, 'Isotonic Regression', axes[2])
    ]):
        # Calculate calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, probs, n_bins=n_bins, strategy='uniform'
        )
        
        # Calculate metrics
        ece = calculate_ece(y_true, probs, n_bins=n_bins)
        brier = brier_score_loss(y_true, probs)
        
        # Plot calibration curve
        ax.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', linewidth=2, label=f'Model (ECE={ece:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        
        # Add confidence histogram
        ax2 = ax.twinx()
        ax2.hist(probs, bins=30, alpha=0.3, color='gray', edgecolor='none')
        ax2.set_ylabel('Count', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Formatting
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{title}\nBrier Score: {brier:.3f}')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_reliability_diagram(y_true, probabilities_dict, save_path):
    """
    Create a reliability diagram comparing multiple calibration methods.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    probabilities_dict : dict
        Dictionary mapping method names to probabilities
    save_path : Path
        Where to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for idx, (name, probs) in enumerate(probabilities_dict.items()):
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, probs, n_bins=10, strategy='uniform'
        )
        
        ece = calculate_ece(y_true, probs)
        
        plt.plot(mean_predicted_value, fraction_of_positives, 
                marker='o', linewidth=2, color=colors[idx], 
                label=f'{name} (ECE={ece:.3f})')
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Add shaded regions for over/under confidence
    plt.fill_between([0, 1], [0, 1], 1, alpha=0.1, color='red', 
                     label='Over-confident region')
    plt.fill_between([0, 1], 0, [0, 1], alpha=0.1, color='blue', 
                     label='Under-confident region')
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Comparison: Reliability Diagram', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_calibrated_models(model_path, artifact_path, X_train, y_train, X_test, y_test):
    """
    Create calibrated versions of the XGBoost model.
    
    Parameters
    ----------
    model_path : Path
        Path to XGBoost model
    artifact_path : Path
        Path to model artifact
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
        
    Returns
    -------
    dict
        Dictionary containing calibrated models and predictions
    """
    print("Loading original model...")
    # Load the model
    predictor = SDOHPredictor(model_path, artifact_path)
    
    # Create wrapper for scikit-learn compatibility
    model_wrapper = XGBoostClassifierWrapper(
        predictor.model, 
        predictor.feature_names
    )
    
    # Get uncalibrated predictions
    print("Getting uncalibrated predictions...")
    prob_uncalibrated_test = predictor.predict_proba(X_test, scale=False)
    prob_uncalibrated_train = predictor.predict_proba(X_train, scale=False)
    
    # Platt Scaling (using cross-validation on training set)
    print("Applying Platt scaling...")
    platt_calibrator = CalibratedClassifierCV(
        model_wrapper, 
        method='sigmoid', 
        cv=3
    )
    platt_calibrator.fit(X_train, y_train)
    prob_platt_test = platt_calibrator.predict_proba(X_test)[:, 1]
    
    # Isotonic Regression
    print("Applying Isotonic regression...")
    isotonic_calibrator = CalibratedClassifierCV(
        model_wrapper, 
        method='isotonic', 
        cv=3
    )
    isotonic_calibrator.fit(X_train, y_train)
    prob_isotonic_test = isotonic_calibrator.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'uncalibrated': {
            'ece': calculate_ece(y_test, prob_uncalibrated_test),
            'brier': brier_score_loss(y_test, prob_uncalibrated_test),
            'log_loss': log_loss(y_test, prob_uncalibrated_test)
        },
        'platt': {
            'ece': calculate_ece(y_test, prob_platt_test),
            'brier': brier_score_loss(y_test, prob_platt_test),
            'log_loss': log_loss(y_test, prob_platt_test)
        },
        'isotonic': {
            'ece': calculate_ece(y_test, prob_isotonic_test),
            'brier': brier_score_loss(y_test, prob_isotonic_test),
            'log_loss': log_loss(y_test, prob_isotonic_test)
        }
    }
    
    return {
        'models': {
            'platt': platt_calibrator,
            'isotonic': isotonic_calibrator
        },
        'predictions': {
            'uncalibrated': prob_uncalibrated_test,
            'platt': prob_platt_test,
            'isotonic': prob_isotonic_test
        },
        'metrics': metrics
    }


def save_calibrated_models(models, base_path):
    """Save calibrated models to disk."""
    models_path = base_path / 'models'
    
    # Save Platt-scaled model
    joblib.dump(models['platt'], models_path / 'xgboost_platt_calibrated.joblib')
    print(f"Saved Platt-calibrated model to {models_path / 'xgboost_platt_calibrated.joblib'}")
    
    # Save Isotonic model
    joblib.dump(models['isotonic'], models_path / 'xgboost_isotonic_calibrated.joblib')
    print(f"Saved Isotonic-calibrated model to {models_path / 'xgboost_isotonic_calibrated.joblib'}")


def generate_synthetic_test_data(n_samples=10000, n_features=20, random_state=42):
    """
    Generate synthetic test data for demonstration purposes.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of features
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (X_train, y_train, X_test, y_test, feature_names)
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear decision boundary
    weights = np.random.randn(n_features)
    linear_combination = X @ weights
    
    # Add non-linearity
    nonlinear_term = 0.5 * np.sin(X[:, 0] * X[:, 1]) + 0.3 * (X[:, 2] ** 2)
    
    # Generate probabilities with some miscalibration
    true_probs = 1 / (1 + np.exp(-(linear_combination + nonlinear_term)))
    
    # Add miscalibration (overconfidence)
    miscalibrated_probs = np.power(true_probs, 0.7)
    
    # Generate labels
    y = (np.random.rand(n_samples) < miscalibrated_probs).astype(int)
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Split into train and test
    split_idx = int(0.7 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to DataFrames
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    
    return X_train, y_train, X_test, y_test, feature_names


def main():
    """Main execution function."""
    # Set up paths
    base_path = Path(__file__).parent.parent
    model_path = base_path / 'models' / 'xgboost_best.json'
    artifact_path = base_path / 'models' / 'model_artifact.joblib'
    results_path = base_path / 'results' / 'figures'
    
    # Check if model exists
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    print("SDOH Model Calibration Tool")
    print("=" * 50)
    
    # Load or generate test data
    # In a real scenario, you would load your actual test data here
    print("\nGenerating synthetic test data for demonstration...")
    print("Note: In production, replace this with your actual test data")
    
    X_train, y_train, X_test, y_test, feature_names = generate_synthetic_test_data()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive rate: {y_test.mean():.2%}")
    
    # Create calibrated models
    print("\nCreating calibrated models...")
    results = create_calibrated_models(
        model_path, artifact_path, 
        X_train, y_train, 
        X_test, y_test
    )
    
    # Print metrics comparison
    print("\nCalibration Metrics Comparison:")
    print("-" * 50)
    print(f"{'Method':<15} {'ECE':<10} {'Brier Score':<12} {'Log Loss':<10}")
    print("-" * 50)
    
    for method, metrics in results['metrics'].items():
        print(f"{method.capitalize():<15} "
              f"{metrics['ece']:<10.4f} "
              f"{metrics['brier']:<12.4f} "
              f"{metrics['log_loss']:<10.4f}")
    
    # Generate calibration plots
    print("\nGenerating calibration plots...")
    
    # Comparison plot
    plot_calibration_comparison(
        y_test,
        results['predictions']['uncalibrated'],
        results['predictions']['platt'],
        results['predictions']['isotonic'],
        results_path / 'calibration_comparison.png'
    )
    print(f"Saved calibration comparison to {results_path / 'calibration_comparison.png'}")
    
    # Reliability diagram
    plot_reliability_diagram(
        y_test,
        {
            'Uncalibrated': results['predictions']['uncalibrated'],
            'Platt Scaling': results['predictions']['platt'],
            'Isotonic': results['predictions']['isotonic']
        },
        results_path / 'reliability_diagram.png'
    )
    print(f"Saved reliability diagram to {results_path / 'reliability_diagram.png'}")
    
    # Save calibrated models
    print("\nSaving calibrated models...")
    save_calibrated_models(results['models'], base_path)
    
    # Create updated performance curves
    print("\nUpdating performance curves with calibrated model...")
    create_updated_performance_curves(
        y_test, 
        results['predictions'], 
        results_path / 'performance_curves_calibrated.png'
    )
    
    print("\nCalibration complete!")
    print("\nNext steps:")
    print("1. Review the calibration plots in results/figures/")
    print("2. Use the calibrated models for predictions:")
    print("   - models/xgboost_platt_calibrated.joblib")
    print("   - models/xgboost_isotonic_calibrated.joblib")
    print("3. Update your prediction pipeline to use calibrated probabilities")


def create_updated_performance_curves(y_true, predictions_dict, save_path):
    """Create updated performance curves showing calibrated models."""
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = {'uncalibrated': 'red', 'platt': 'blue', 'isotonic': 'green'}
    
    # ROC Curves
    ax = axes[0]
    for name, probs in predictions_dict.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, color=colors[name], linewidth=2, 
                label=f'{name.capitalize()} (AUC={auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Calibration Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    ax = axes[1]
    for name, probs in predictions_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, probs)
        auprc = np.trapz(precision[:-1], recall[:-1])
        ax.plot(recall, precision, color=colors[name], linewidth=2,
                label=f'{name.capitalize()} (AUPRC={auprc:.3f})')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Calibration Comparison')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved updated performance curves to {save_path}")


if __name__ == '__main__':
    main()