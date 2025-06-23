#!/usr/bin/env python3
"""
Generate TRIPOD-AI Compliant Figures for JAMA Publication
Creates all required figures at publication quality
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from sklearn.calibration import calibration_curve
import xgboost as xgb
import joblib
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# JAMA publication specifications
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.linewidth'] = 0.5

# JAMA color scheme
JAMA_COLORS = {
    'primary': '#0066CC',
    'secondary': '#CC3300',
    'tertiary': '#009900',
    'quaternary': '#FF6600',
    'neutral': '#666666'
}

def load_model_and_test_data():
    """Load the scientifically validated model and test data"""
    print("Loading model and test data...")
    
    # Load calibrated model
    model = joblib.load('models/xgboost_scientific_calibrated.joblib')
    
    # Load metadata
    with open('models/scientific_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Load base model for feature importance
    base_model = xgb.XGBClassifier()
    base_model.load_model('models/xgboost_scientific_base.json')
    
    # Load full dataset
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    df = pd.read_csv(data_path)
    
    # Use EXACT same split as training script to match test metrics
    feature_cols = [col for col in df.columns if col not in 
                   ['person_id', 'payor_id', 'mbi_id', 'sdoh_two_yes',
                    'race_category', 'ethnicity_category']]
    
    X = df[feature_cols]
    y = df['sdoh_two_yes']
    
    # EXACT same 60/20/20 split as training script
    from sklearn.model_selection import train_test_split
    np.random.seed(2025)  # RANDOM_SEED from config
    
    # First split off 20% for test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=2025
    )
    # Then split remaining 80% into 60% train, 20% validation  
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=2025
    )
    
    print(f"   Test set: {len(X_test):,} samples (matches training script)")
    
    return model, base_model, X_test, y_test, metadata, feature_cols

def create_figure1_model_performance(model, X_test, y_test, threshold, output_dir):
    """Figure 1: Main Model Performance (4 panels)"""
    print("\nCreating Figure 1: Model Performance...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))  # JAMA single column width ~7 inches
    
    # A. ROC Curve
    ax1 = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    ax1.plot(fpr, tpr, color=JAMA_COLORS['primary'], linewidth=2.5)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax1.fill_between(fpr, tpr, alpha=0.1, color=JAMA_COLORS['primary'])
    
    ax1.set_xlabel('False-Positive Rate')
    ax1.set_ylabel('True-Positive Rate')
    ax1.set_title('A', loc='left', fontweight='bold', fontsize=14)
    ax1.text(0.6, 0.3, f'AUC = {auc:.3f}', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.01, 1.01)
    ax1.set_ylim(-0.01, 1.01)
    
    # B. Precision-Recall Curve
    ax2 = axes[0, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    baseline = y_test.mean()
    
    ax2.plot(recall, precision, color=JAMA_COLORS['secondary'], linewidth=2.5)
    ax2.axhline(y=baseline, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.fill_between(recall, precision, alpha=0.1, color=JAMA_COLORS['secondary'])
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('B', loc='left', fontweight='bold', fontsize=14)
    ax2.text(0.3, 0.7, f'AP = {ap:.3f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.3, baseline + 0.02, f'Baseline = {baseline:.3f}', 
             fontsize=9, color='gray')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.01, 1.01)
    ax2.set_ylim(-0.01, 1.01)
    
    # C. Calibration Plot
    ax3 = axes[1, 0]
    fraction_pos, mean_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
    
    # Calculate ECE
    ece = 0
    for i in range(len(fraction_pos)):
        if i < len(mean_pred):
            bin_size = len(y_test) / 10  # Approximate
            ece += abs(fraction_pos[i] - mean_pred[i]) * (bin_size / len(y_test))
    
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax3.plot(mean_pred, fraction_pos, 'o-', color=JAMA_COLORS['tertiary'], 
             linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # Add confidence intervals
    for i in range(len(mean_pred)):
        n_in_bin = int(len(y_test) / 10)
        if n_in_bin > 0:
            se = np.sqrt(fraction_pos[i] * (1 - fraction_pos[i]) / n_in_bin)
            ci = 1.96 * se
            ax3.plot([mean_pred[i], mean_pred[i]], 
                    [max(0, fraction_pos[i] - ci), min(1, fraction_pos[i] + ci)],
                    color=JAMA_COLORS['tertiary'], linewidth=3, alpha=0.5)
    
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Observed Frequency')
    ax3.set_title('C', loc='left', fontweight='bold', fontsize=14)
    ax3.text(0.05, 0.9, f'ECE = {ece:.3f}', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-0.01, 1.01)
    ax3.set_ylim(-0.01, 1.01)
    
    # D. Score Distribution
    ax4 = axes[1, 1]
    
    # Create histograms
    bins = np.linspace(0, 1, 31)
    
    # Negative class
    counts_neg, _, _ = ax4.hist(y_pred_proba[y_test == 0], bins=bins, 
                                alpha=0.7, color=JAMA_COLORS['primary'], 
                                label='No SDOH Need', density=True,
                                edgecolor='white', linewidth=0.5)
    
    # Positive class
    counts_pos, _, _ = ax4.hist(y_pred_proba[y_test == 1], bins=bins, 
                                alpha=0.7, color=JAMA_COLORS['secondary'], 
                                label='SDOH Need', density=True,
                                edgecolor='white', linewidth=0.5)
    
    # Add threshold line
    ax4.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold = {threshold:.3f}')
    
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Density')
    ax4.set_title('D', loc='left', fontweight='bold', fontsize=14)
    ax4.legend(loc='upper right', framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(-0.01, 1.01)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Add main figure title
    fig.suptitle('Figure 1. Model Performance Characteristics', 
                 fontsize=14, fontweight='bold')
    
    # Save at 300 DPI for JAMA
    output_path = os.path.join(output_dir, 'figure1_model_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  Saved Figure 1 to {output_path}")
    plt.close()

def create_figure2_feature_importance(base_model, feature_names, output_dir):
    """Figure 2: Feature Importance Analysis"""
    print("\nCreating Figure 2: Feature Importance...")
    
    # Get feature importance
    importance = base_model.feature_importances_
    
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(20)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(importance_df))
    bars = ax.barh(y_pos, importance_df['importance'], 
                   color=JAMA_COLORS['primary'], edgecolor='white', linewidth=0.5)
    
    # Color code by category
    colors = []
    for feature in importance_df['feature']:
        if 'insurance' in feature.lower() or 'financial' in feature.lower():
            colors.append(JAMA_COLORS['primary'])
        elif 'age' in feature.lower():
            colors.append(JAMA_COLORS['secondary'])
        elif 'rpl' in feature.lower() or 'svi' in feature.lower():
            colors.append(JAMA_COLORS['tertiary'])
        elif 'adi' in feature.lower():
            colors.append(JAMA_COLORS['quaternary'])
        else:
            colors.append(JAMA_COLORS['neutral'])
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Figure 2. Top 20 Most Important Features', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add value labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        ax.text(row['importance'] + 0.001, i, f"{row['importance']:.3f}", 
                va='center', fontsize=9)
    
    # Add legend for categories
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=JAMA_COLORS['primary'], label='Insurance/Financial'),
        Patch(facecolor=JAMA_COLORS['secondary'], label='Demographics'),
        Patch(facecolor=JAMA_COLORS['tertiary'], label='Social Vulnerability'),
        Patch(facecolor=JAMA_COLORS['quaternary'], label='Area Deprivation'),
        Patch(facecolor=JAMA_COLORS['neutral'], label='Other')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, importance_df['importance'].max() * 1.15)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'figure2_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  Saved Figure 2 to {output_path}")
    plt.close()

def create_figure3_subgroup_performance(model, X_test, y_test, df_test, threshold, output_dir):
    """Figure 3: Subgroup Performance Analysis"""
    print("\nCreating Figure 3: Subgroup Performance...")
    
    # Create demographic groups
    df_test['age_group'] = pd.cut(df_test['age_at_survey'], 
                                  bins=[0, 35, 50, 65, 100],
                                  labels=['18-35', '36-50', '51-65', '66+'])
    
    # Convert sex_female to categorical
    df_test['sex'] = df_test['sex_female'].map({1: 'Female', 0: 'Male'})
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for each subgroup
    subgroups = {
        'Age': df_test['age_group'],
        'Sex': df_test['sex'],
        'Race': df_test['race_category'].apply(lambda x: x if x in ['Black or African American', 'White'] else 'Other')
    }
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    for idx, (group_name, groups) in enumerate(subgroups.items()):
        ax = axes[idx]
        
        # Calculate AUC for each subgroup
        auc_values = []
        group_labels = []
        sample_sizes = []
        
        for group in groups.unique():
            if pd.isna(group):
                continue
            
            mask = groups == group
            if mask.sum() < 100:
                continue
            
            y_true_group = y_test[mask]
            y_pred_group = y_pred_proba[mask]
            
            if len(np.unique(y_true_group)) > 1:
                auc = roc_auc_score(y_true_group, y_pred_group)
                auc_values.append(auc)
                group_labels.append(str(group))
                sample_sizes.append(mask.sum())
        
        # Create bar plot
        y_pos = np.arange(len(group_labels))
        bars = ax.bar(y_pos, auc_values, color=JAMA_COLORS['primary'], 
                      edgecolor='white', linewidth=1)
        
        # Color code bars by performance
        for bar, auc in zip(bars, auc_values):
            if auc < 0.7:
                bar.set_color(JAMA_COLORS['secondary'])
            elif auc > 0.8:
                bar.set_color(JAMA_COLORS['tertiary'])
        
        # Add overall AUC line
        overall_auc = roc_auc_score(y_test, y_pred_proba)
        ax.axhline(y=overall_auc, color='black', linestyle='--', 
                  linewidth=1.5, alpha=0.7, label=f'Overall ({overall_auc:.3f})')
        
        # Customize
        ax.set_xticks(y_pos)
        ax.set_xticklabels(group_labels, rotation=45 if len(group_labels) > 3 else 0, ha='right')
        ax.set_ylabel('AUC' if idx == 0 else '')
        ax.set_ylim(0.5, 0.9)
        ax.set_title(f'{["A", "B", "C"][idx]}. {group_name}', 
                    loc='left', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (auc, n) in enumerate(zip(auc_values, sample_sizes)):
            ax.text(i, auc + 0.01, f'{auc:.3f}\n(n={n:,})', 
                   ha='center', va='bottom', fontsize=8)
        
        if idx == 1:
            ax.legend(loc='lower center', framealpha=0.9)
    
    # Main title
    fig.suptitle('Figure 3. Model Performance Across Demographic Subgroups', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    # Save
    output_path = os.path.join(output_dir, 'figure3_subgroup_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  Saved Figure 3 to {output_path}")
    plt.close()

def create_figure4_decision_curve(model, X_test, y_test, threshold, output_dir):
    """Figure 4: Decision Curve Analysis"""
    print("\nCreating Figure 4: Decision Curve Analysis...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate net benefit for different threshold probabilities
    threshold_probs = np.linspace(0.01, 0.50, 100)
    net_benefits_model = []
    net_benefits_all = []
    
    prevalence = y_test.mean()
    
    for thresh_prob in threshold_probs:
        # Model strategy
        y_pred = (y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        n = len(y_test)
        net_benefit_model = (tp / n) - (fp / n) * (thresh_prob / (1 - thresh_prob))
        net_benefits_model.append(net_benefit_model)
        
        # Treat all strategy
        net_benefit_all = prevalence - (1 - prevalence) * (thresh_prob / (1 - thresh_prob))
        net_benefits_all.append(net_benefit_all)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    
    # Plot strategies
    ax.plot(threshold_probs, net_benefits_model, 
           color=JAMA_COLORS['primary'], linewidth=2.5, label='SDOH Model')
    ax.plot(threshold_probs, net_benefits_all, 
           color=JAMA_COLORS['secondary'], linewidth=2.5, linestyle='--', label='Screen All')
    ax.axhline(y=0, color='black', linewidth=1.5, linestyle=':', label='Screen None')
    
    # Add prevalence line
    ax.axvline(x=prevalence, color='gray', linewidth=1, linestyle='-.', alpha=0.7)
    ax.text(prevalence + 0.01, ax.get_ylim()[1] * 0.9, 
           f'Prevalence\n({prevalence:.3f})', fontsize=9, color='gray')
    
    # Shade region where model is beneficial
    model_better = np.array(net_benefits_model) > np.maximum(net_benefits_all, 0)
    ax.fill_between(threshold_probs, ax.get_ylim()[0], ax.get_ylim()[1], 
                   where=model_better, alpha=0.1, color=JAMA_COLORS['primary'])
    
    # Customize
    ax.set_xlabel('Threshold Probability')
    ax.set_ylabel('Net Benefit')
    ax.set_title('Figure 4. Decision Curve Analysis', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-0.05, 0.15)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, 'figure4_decision_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"  Saved Figure 4 to {output_path}")
    plt.close()

def create_supplementary_figures(model, base_model, X_test, y_test, feature_names, threshold, output_dir):
    """Create supplementary figures for online publication"""
    supp_dir = os.path.join(output_dir, 'supplementary')
    os.makedirs(supp_dir, exist_ok=True)
    
    print("\nCreating Supplementary Figures...")
    
    # S1: Extended Feature Importance (top 50)
    print("  Creating Figure S1: Extended Feature Importance...")
    
    importance = base_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(50)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 12))
    
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'], 
           color=JAMA_COLORS['primary'], edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Figure S1. Top 50 Most Important Features', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(supp_dir, 'figureS1_extended_features.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # S2: Threshold Analysis
    print("  Creating Figure S2: Threshold Analysis...")
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.01, 0.50, 100)
    
    metrics = {'sensitivity': [], 'specificity': [], 'ppv': [], 'npv': [], 'f1': []}
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics['sensitivity'].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        metrics['specificity'].append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        metrics['ppv'].append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        metrics['npv'].append(tn / (tn + fn) if (tn + fn) > 0 else 0)
        
        sens = metrics['sensitivity'][-1]
        ppv = metrics['ppv'][-1]
        metrics['f1'].append(2 * sens * ppv / (sens + ppv) if (sens + ppv) > 0 else 0)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))
    
    # Panel A: Sensitivity and Specificity
    ax1.plot(thresholds, metrics['sensitivity'], color=JAMA_COLORS['primary'], 
             linewidth=2.5, label='Sensitivity')
    ax1.plot(thresholds, metrics['specificity'], color=JAMA_COLORS['secondary'], 
             linewidth=2.5, label='Specificity')
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Rate')
    ax1.set_title('A. Sensitivity and Specificity', loc='left', fontweight='bold')
    ax1.legend(loc='center right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1)
    
    # Panel B: PPV and F1
    ax2.plot(thresholds, metrics['ppv'], color=JAMA_COLORS['tertiary'], 
             linewidth=2.5, label='PPV')
    ax2.plot(thresholds, metrics['f1'], color=JAMA_COLORS['quaternary'], 
             linewidth=2.5, label='F1 Score')
    ax2.axvline(x=threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=y_test.mean(), color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax2.text(0.4, y_test.mean() + 0.01, 'Baseline', fontsize=9, color='gray')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('B. PPV and F1 Score', loc='left', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 0.5)
    
    fig.suptitle('Figure S2. Performance Metrics Across Thresholds', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(os.path.join(supp_dir, 'figureS2_threshold_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("  Supplementary figures saved to", supp_dir)

def generate_figure_captions(output_dir):
    """Generate figure captions file for manuscript"""
    captions_path = os.path.join(output_dir, 'figure_captions_jama.txt')
    
    with open(captions_path, 'w') as f:
        f.write("FIGURE CAPTIONS\n")
        f.write("="*50 + "\n\n")
        
        f.write("Figure 1. Model Performance Characteristics\n")
        f.write("Performance evaluation of the social determinants of health (SDOH) ")
        f.write("screening model on the held-out test set (n=78,745). ")
        f.write("(A) Receiver operating characteristic curve showing the trade-off ")
        f.write("between true-positive and false-positive rates. ")
        f.write("(B) Precision-recall curve demonstrating performance for the minority ")
        f.write("positive class, with the dashed line indicating baseline performance. ")
        f.write("(C) Calibration plot comparing predicted probabilities with observed ")
        f.write("frequencies; perfect calibration would follow the diagonal line. ")
        f.write("Error bars represent 95% confidence intervals. ")
        f.write("(D) Distribution of predicted probabilities stratified by actual ")
        f.write("SDOH status, with the vertical dashed line indicating the selected ")
        f.write("threshold (0.05). AUC indicates area under the curve; AP, average ")
        f.write("precision; ECE, expected calibration error.\n\n")
        
        f.write("Figure 2. Feature Importance Analysis\n")
        f.write("Top 20 most important features contributing to SDOH risk prediction, ")
        f.write("measured by gain in the gradient boosting model. Features are ")
        f.write("color-coded by category: insurance/financial factors (blue), ")
        f.write("demographics (red), social vulnerability indices (green), ")
        f.write("area deprivation index (orange), and other factors (gray). ")
        f.write("Higher scores indicate greater contribution to model predictions.\n\n")
        
        f.write("Figure 3. Model Performance Across Demographic Subgroups\n")
        f.write("Area under the curve (AUC) stratified by (A) age groups, ")
        f.write("(B) sex, and (C) race. The dashed horizontal line represents ")
        f.write("overall model performance. Sample sizes are shown below each bar. ")
        f.write("Subgroups with fewer than 100 patients were excluded from analysis.\n\n")
        
        f.write("Figure 4. Decision Curve Analysis\n")
        f.write("Net benefit of the SDOH screening model compared with strategies ")
        f.write("of screening all patients or screening no patients across a range ")
        f.write("of threshold probabilities. The model provides positive net benefit ")
        f.write("(shaded area) when the threshold probability exceeds the population ")
        f.write("prevalence (vertical gray line). The model strategy accounts for ")
        f.write("both the benefits of identifying true positives and the costs of ")
        f.write("false-positive screening.\n\n")
        
        f.write("Online Supplementary Figures\n")
        f.write("-"*30 + "\n")
        f.write("Figure S1. Extended feature importance analysis showing the top 50 ")
        f.write("features.\n\n")
        
        f.write("Figure S2. Performance metrics across the full range of classification ")
        f.write("thresholds. (A) Sensitivity and specificity trade-off. ")
        f.write("(B) Positive predictive value and F1 score. The vertical dashed line ")
        f.write("indicates the selected threshold.\n")
    
    print(f"\nFigure captions saved to {captions_path}")

def main():
    """Generate all TRIPOD-AI compliant figures for JAMA"""
    print("="*80)
    print("GENERATING TRIPOD-AI COMPLIANT FIGURES FOR JAMA")
    print("="*80)
    
    # Create output directory
    output_dir = 'results/figures/jama'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model, base_model, X_test, y_test, metadata, feature_names = load_model_and_test_data()
    
    # Get threshold
    threshold = metadata['threshold_selection']['optimal_threshold']
    print(f"\nUsing validated threshold: {threshold}")
    
    # Load full test dataframe for demographic analysis
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    df_full = pd.read_csv(data_path)
    n_test = int(len(df_full) * 0.2)
    df_test = df_full.iloc[-n_test:].copy()
    
    # Create main figures
    create_figure1_model_performance(model, X_test, y_test, threshold, output_dir)
    create_figure2_feature_importance(base_model, feature_names, output_dir)
    create_figure3_subgroup_performance(model, X_test, y_test, df_test, threshold, output_dir)
    create_figure4_decision_curve(model, X_test, y_test, threshold, output_dir)
    
    # Create supplementary figures
    create_supplementary_figures(model, base_model, X_test, y_test, 
                               feature_names, threshold, output_dir)
    
    # Generate figure captions
    generate_figure_captions(output_dir)
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE")
    print("="*80)
    print("\nMain figures (300 DPI, JAMA specifications):")
    print("1. Figure 1: Model Performance (4 panels)")
    print("2. Figure 2: Feature Importance")
    print("3. Figure 3: Subgroup Performance")
    print("4. Figure 4: Decision Curve Analysis")
    print("\nSupplementary figures:")
    print("- Figure S1: Extended Feature Importance")
    print("- Figure S2: Threshold Analysis")
    print("\nAll figures saved to:", output_dir)
    print("Figure captions saved for manuscript inclusion")

if __name__ == "__main__":
    main()