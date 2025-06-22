"""
Fairness Analysis Module
=======================

Analyzes model fairness across demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate
)


class FairnessAnalyzer:
    """
    Analyze model fairness across demographic groups.
    
    This class provides comprehensive fairness metrics and visualizations
    to ensure equitable model performance.
    """
    
    def __init__(self):
        """Initialize the fairness analyzer."""
        self.results = {}
        self.metric_frames = {}
        
    def analyze_demographic_parity(self, 
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 sensitive_features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze demographic parity across sensitive features.
        
        Parameters
        ----------
        y_true : array
            True labels
        y_pred : array
            Predictions
        sensitive_features : DataFrame
            DataFrame with sensitive feature columns
            
        Returns
        -------
        dict
            Fairness metrics for each sensitive feature
        """
        results = {}
        
        for column in sensitive_features.columns:
            # Skip if too many missing values
            if sensitive_features[column].isna().sum() > len(sensitive_features) * 0.5:
                continue
                
            # Filter out NaN values
            mask = sensitive_features[column].notna()
            if mask.sum() < 100:  # Skip if too few samples
                continue
                
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            feature_filtered = sensitive_features[column][mask]
            
            # Calculate metrics
            dp_diff = demographic_parity_difference(
                y_true_filtered, y_pred_filtered, 
                sensitive_features=feature_filtered
            )
            
            dp_ratio = demographic_parity_ratio(
                y_true_filtered, y_pred_filtered,
                sensitive_features=feature_filtered
            )
            
            # Selection rates by group
            groups = feature_filtered.unique()
            group_rates = {}
            
            for group in groups:
                group_mask = feature_filtered == group
                if group_mask.sum() > 30:  # Need sufficient samples
                    rate = y_pred_filtered[group_mask].mean()
                    group_rates[str(group)] = {
                        'rate': float(rate),
                        'n': int(group_mask.sum())
                    }
            
            results[column] = {
                'demographic_parity_difference': float(dp_diff),
                'demographic_parity_ratio': float(dp_ratio),
                'selection_rates': group_rates,
                'n_total': int(mask.sum())
            }
        
        return results
    
    def analyze_performance_parity(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_proba: np.ndarray,
                                 sensitive_features: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analyze performance metrics across sensitive features.
        
        Parameters
        ----------
        y_true : array
            True labels
        y_pred : array
            Predictions
        y_proba : array
            Predicted probabilities
        sensitive_features : DataFrame
            DataFrame with sensitive feature columns
            
        Returns
        -------
        dict
            Performance metrics by demographic group
        """
        results = {}
        
        # Define metrics to calculate
        def calculate_metrics(y_true, y_pred):
            if len(np.unique(y_pred)) < 2 or y_pred.sum() == 0:
                return {
                    'sensitivity': 0.0,
                    'specificity': 0.0,
                    'ppv': 0.0,
                    'npv': 0.0,
                    'f1': 0.0
                }
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            return {
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            }
        
        for column in sensitive_features.columns:
            # Filter out NaN values
            mask = sensitive_features[column].notna()
            if mask.sum() < 100:
                continue
                
            y_true_filtered = y_true[mask]
            y_pred_filtered = y_pred[mask]
            y_proba_filtered = y_proba[mask]
            feature_filtered = sensitive_features[column][mask]
            
            # Calculate metrics for each group
            groups = feature_filtered.unique()
            group_metrics = {}
            
            for group in groups:
                group_mask = feature_filtered == group
                if group_mask.sum() > 30:
                    metrics = calculate_metrics(
                        y_true_filtered[group_mask],
                        y_pred_filtered[group_mask]
                    )
                    metrics['n'] = int(group_mask.sum())
                    metrics['prevalence'] = float(y_true_filtered[group_mask].mean())
                    metrics['mean_score'] = float(y_proba_filtered[group_mask].mean())
                    group_metrics[str(group)] = metrics
            
            # Calculate disparities
            if len(group_metrics) > 1:
                metric_values = {
                    metric: [g[metric] for g in group_metrics.values()]
                    for metric in ['sensitivity', 'specificity', 'ppv']
                }
                
                disparities = {
                    f'{metric}_range': max(values) - min(values)
                    for metric, values in metric_values.items()
                }
            else:
                disparities = {}
            
            results[column] = {
                'groups': group_metrics,
                'disparities': disparities
            }
        
        return results
    
    def create_fairness_report(self, 
                             demographic_results: Dict,
                             performance_results: Dict) -> str:
        """
        Create a comprehensive fairness report.
        
        Parameters
        ----------
        demographic_results : dict
            Results from demographic parity analysis
        performance_results : dict
            Results from performance parity analysis
            
        Returns
        -------
        str
            Formatted fairness report
        """
        report = "# Fairness Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall summary
        report += "## Executive Summary\n\n"
        
        fair_features = []
        concerning_features = []
        
        for feature, results in demographic_results.items():
            dp_diff = abs(results['demographic_parity_difference'])
            if dp_diff < 0.1:
                fair_features.append(feature)
            else:
                concerning_features.append(feature)
        
        report += f"**Fair features** (DP difference < 0.1): {', '.join(fair_features)}\n"
        report += f"**Features needing attention**: {', '.join(concerning_features)}\n\n"
        
        # Detailed results by feature
        report += "## Detailed Analysis by Feature\n\n"
        
        for feature in demographic_results.keys():
            report += f"### {feature.upper()}\n\n"
            
            # Demographic parity
            dp_results = demographic_results[feature]
            report += f"**Demographic Parity**\n"
            report += f"- Difference: {dp_results['demographic_parity_difference']:.3f}\n"
            report += f"- Ratio: {dp_results['demographic_parity_ratio']:.3f}\n\n"
            
            # Selection rates
            report += "**Selection Rates by Group**\n"
            for group, info in dp_results['selection_rates'].items():
                report += f"- {group}: {info['rate']*100:.1f}% (n={info['n']:,})\n"
            report += "\n"
            
            # Performance metrics
            if feature in performance_results:
                perf_results = performance_results[feature]
                report += "**Performance Metrics by Group**\n\n"
                
                # Create table
                report += "| Group | N | Sensitivity | Specificity | PPV | F1 |\n"
                report += "|-------|---|-------------|-------------|-----|----|\n"
                
                for group, metrics in perf_results['groups'].items():
                    report += f"| {group} | {metrics['n']:,} | "
                    report += f"{metrics['sensitivity']*100:.1f}% | "
                    report += f"{metrics['specificity']*100:.1f}% | "
                    report += f"{metrics['ppv']*100:.1f}% | "
                    report += f"{metrics['f1']:.3f} |\n"
                
                report += "\n"
                
                # Disparities
                if perf_results['disparities']:
                    report += "**Performance Disparities**\n"
                    for metric, value in perf_results['disparities'].items():
                        report += f"- {metric}: {value:.3f}\n"
                    report += "\n"
        
        return report
    
    def plot_fairness_dashboard(self,
                              demographic_results: Dict,
                              performance_results: Dict,
                              save_path: Optional[str] = None):
        """
        Create a comprehensive fairness visualization dashboard.
        
        Parameters
        ----------
        demographic_results : dict
            Results from demographic parity analysis
        performance_results : dict
            Results from performance parity analysis
        save_path : str, optional
            Path to save the figure
        """
        # Determine number of features
        features = list(demographic_results.keys())
        n_features = len(features)
        
        # Create subplots
        fig, axes = plt.subplots(n_features, 3, figsize=(15, 5*n_features))
        if n_features == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(features):
            # Selection rates
            ax = axes[idx, 0]
            dp_results = demographic_results[feature]
            
            groups = list(dp_results['selection_rates'].keys())
            rates = [dp_results['selection_rates'][g]['rate'] for g in groups]
            ns = [dp_results['selection_rates'][g]['n'] for g in groups]
            
            bars = ax.bar(groups, rates, alpha=0.8)
            ax.set_ylabel('Selection Rate')
            ax.set_title(f'{feature} - Selection Rates')
            ax.set_ylim(0, max(rates) * 1.2)
            
            # Add sample sizes
            for bar, n in zip(bars, ns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={n}', ha='center', va='bottom', fontsize=9)
            
            # PPV comparison
            if feature in performance_results:
                ax = axes[idx, 1]
                perf_results = performance_results[feature]
                
                groups = list(perf_results['groups'].keys())
                ppvs = [perf_results['groups'][g]['ppv'] for g in groups]
                
                ax.bar(groups, ppvs, alpha=0.8, color='green')
                ax.set_ylabel('PPV')
                ax.set_title(f'{feature} - Positive Predictive Value')
                ax.set_ylim(0, max(ppvs) * 1.2)
                
                # Sensitivity comparison
                ax = axes[idx, 2]
                sensitivities = [perf_results['groups'][g]['sensitivity'] for g in groups]
                
                ax.bar(groups, sensitivities, alpha=0.8, color='orange')
                ax.set_ylabel('Sensitivity')
                ax.set_title(f'{feature} - Sensitivity')
                ax.set_ylim(0, 1)
        
        plt.suptitle('Fairness Analysis Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
    
    def calculate_fairness_score(self,
                               demographic_results: Dict,
                               performance_results: Dict) -> float:
        """
        Calculate an overall fairness score.
        
        Parameters
        ----------
        demographic_results : dict
            Results from demographic parity analysis
        performance_results : dict
            Results from performance parity analysis
            
        Returns
        -------
        float
            Overall fairness score (0-1, higher is better)
        """
        scores = []
        
        # Demographic parity scores
        for feature, results in demographic_results.items():
            dp_diff = abs(results['demographic_parity_difference'])
            # Convert to 0-1 score (0.2 difference = 0 score)
            dp_score = max(0, 1 - dp_diff / 0.2)
            scores.append(dp_score)
        
        # Performance parity scores
        for feature, results in performance_results.items():
            if results['disparities']:
                ppv_range = results['disparities'].get('ppv_range', 0)
                sens_range = results['disparities'].get('sensitivity_range', 0)
                
                # Convert to 0-1 scores
                ppv_score = max(0, 1 - ppv_range / 0.2)
                sens_score = max(0, 1 - sens_range / 0.3)
                
                scores.extend([ppv_score, sens_score])
        
        # Return average score
        return np.mean(scores) if scores else 0.5