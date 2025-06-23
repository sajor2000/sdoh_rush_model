"""
SDOH Prediction Model Package
============================

A machine learning model for predicting patients with 2+ unmet SDOH needs.
"""

__version__ = "1.0.0"
__author__ = "Juan C. Rojas, MD, MS and SDOH Research Team"

from .model_evaluation import SDOHPredictor
from .fairness_analysis import FairnessAnalyzer
from .sdoh_risk_screener import SDOHRiskScreener

__all__ = ['SDOHPredictor', 'FairnessAnalyzer', 'SDOHRiskScreener']