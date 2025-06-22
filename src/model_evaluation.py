"""
Model Evaluation and Prediction Module
=====================================

Handles model loading, prediction, and evaluation.
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from pathlib import Path
from typing import Union, Dict, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score, f1_score, brier_score_loss
)


class SDOHPredictor:
    """
    SDOH Risk Prediction Model.
    
    This class handles loading the trained XGBoost model and making predictions
    for SDOH screening.
    """
    
    def __init__(self, model_path: Union[str, Path], artifact_path: Optional[Union[str, Path]] = None):
        """
        Initialize the predictor.
        
        Parameters
        ----------
        model_path : str or Path
            Path to the XGBoost model file (.json format)
        artifact_path : str or Path, optional
            Path to the complete model artifact (.joblib format) containing
            scaler and feature names
        """
        self.model_path = Path(model_path)
        self.model = self._load_model()
        
        if artifact_path:
            self.artifact_path = Path(artifact_path)
            self._load_artifact()
        else:
            self.scaler = None
            self.feature_names = None
            
        # Clinical thresholds
        self.thresholds = {
            'standard': 0.5644,  # 25% screening rate
            'high_ppv': 0.7726,  # 5% screening rate
            'high_sensitivity': 0.4392  # 40% screening rate
        }
        
    def _load_model(self) -> xgb.Booster:
        """Load XGBoost model from JSON file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        model = xgb.Booster()
        model.load_model(str(self.model_path))
        return model
        
    def _load_artifact(self):
        """Load complete model artifact with scaler and metadata."""
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact file not found: {self.artifact_path}")
        
        artifact = joblib.load(self.artifact_path)
        self.scaler = artifact.get('scaler')
        self.feature_names = artifact.get('feature_names')
        self.model_params = artifact.get('params', {})
        
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray], 
                     scale: bool = True) -> np.ndarray:
        """
        Generate risk scores for patients.
        
        Parameters
        ----------
        X : DataFrame or array
            Patient features
        scale : bool
            Whether to apply scaling (requires scaler from artifact)
            
        Returns
        -------
        array
            Risk scores between 0 and 1
        """
        # Validate features if names are available
        if self.feature_names and isinstance(X, pd.DataFrame):
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            X = X[self.feature_names]
        
        # Convert to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Apply scaling if requested
        if scale:
            if self.scaler is None:
                raise ValueError("Scaler not available. Load model artifact or set scale=False")
            X_array = self.scaler.transform(X_array)
        
        # Create DMatrix and predict
        dmatrix = xgb.DMatrix(X_array)
        risk_scores = self.model.predict(dmatrix)
        
        return risk_scores
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], 
                threshold: Union[float, str] = 'standard',
                scale: bool = True) -> np.ndarray:
        """
        Make binary predictions for screening.
        
        Parameters
        ----------
        X : DataFrame or array
            Patient features
        threshold : float or str
            Decision threshold or preset name ('standard', 'high_ppv', 'high_sensitivity')
        scale : bool
            Whether to apply scaling
            
        Returns
        -------
        array
            Binary predictions (1 = needs screening, 0 = no screening needed)
        """
        # Get risk scores
        risk_scores = self.predict_proba(X, scale=scale)
        
        # Get threshold value
        if isinstance(threshold, str):
            if threshold not in self.thresholds:
                raise ValueError(f"Unknown threshold preset: {threshold}")
            threshold_value = self.thresholds[threshold]
        else:
            threshold_value = threshold
        
        # Apply threshold
        predictions = (risk_scores >= threshold_value).astype(int)
        
        return predictions
    
    def evaluate(self, X: Union[pd.DataFrame, np.ndarray], 
                 y_true: Union[pd.Series, np.ndarray],
                 threshold: Union[float, str] = 'standard',
                 scale: bool = True) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.
        
        Parameters
        ----------
        X : DataFrame or array
            Patient features
        y_true : Series or array
            True labels
        threshold : float or str
            Decision threshold
        scale : bool
            Whether to apply scaling
            
        Returns
        -------
        dict
            Performance metrics
        """
        # Get predictions
        y_proba = self.predict_proba(X, scale=scale)
        y_pred = self.predict(X, threshold=threshold, scale=scale)
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'auprc': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'ppv': precision_score(y_true, y_pred) if y_pred.sum() > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'f1_score': f1_score(y_true, y_pred),
            'screening_rate': y_pred.mean(),
            'prevalence': y_true.mean(),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Parameters
        ----------
        importance_type : str
            Type of importance ('gain', 'cover', 'weight')
            
        Returns
        -------
        DataFrame
            Feature importance scores
        """
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        # Map to feature names if available
        if self.feature_names:
            importance_named = {}
            for k, v in importance_dict.items():
                if k.startswith('f'):
                    idx = int(k[1:])
                    if idx < len(self.feature_names):
                        importance_named[self.feature_names[idx]] = v
            importance_dict = importance_named
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', importance_type]
        ).sort_values(importance_type, ascending=False)
        
        return importance_df
    
    def calculate_net_benefit(self, y_true: np.ndarray, y_proba: np.ndarray,
                            threshold_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate net benefit for decision curve analysis.
        
        Parameters
        ----------
        y_true : array
            True labels
        y_proba : array
            Predicted probabilities
        threshold_probs : array
            Threshold probabilities to evaluate
            
        Returns
        -------
        tuple
            (threshold_probs, net_benefits)
        """
        net_benefits = []
        
        for pt in threshold_probs:
            # Find appropriate model threshold
            screening_rate = min(pt * 2, 0.99)  # Approximate
            model_threshold = np.percentile(y_proba, (1 - screening_rate) * 100)
            
            # Calculate net benefit
            y_pred = (y_proba >= model_threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            n = len(y_true)
            tpr = tp / n
            fpr = fp / n
            
            net_benefit = tpr - fpr * (pt / (1 - pt))
            net_benefits.append(net_benefit)
        
        return threshold_probs, np.array(net_benefits)