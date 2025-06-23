#!/usr/bin/env python3
"""
Senior Clinic Threshold Analysis
Analyzes optimal threshold for clinics serving patients aged 65+
Considers higher prevalence and resource constraints
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
import joblib
import json
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# JAMA publication settings
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

def load_senior_data():
    """Load data for patients aged 65+"""
    print("Loading data for senior patients (65+)...")
    
    # Load full dataset
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    df = pd.read_csv(data_path)
    
    # Filter for seniors
    df_seniors = df[df['age_at_survey'] >= 65].copy()
    
    print(f"Total patients: {len(df):,}")
    print(f"Senior patients (65+): {len(df_seniors):,} ({len(df_seniors)/len(df)*100:.1f}%)")
    print(f"SDOH prevalence in all: {df['sdoh_two_yes'].mean():.1%}")
    print(f"SDOH prevalence in seniors: {df_seniors['sdoh_two_yes'].mean():.1%}")
    
    return df_seniors

def analyze_senior_subgroups(df_seniors):
    """Analyze different age subgroups within seniors"""
    subgroups = {
        '65-74': df_seniors[(df_seniors['age_at_survey'] >= 65) & (df_seniors['age_at_survey'] < 75)],
        '75-84': df_seniors[(df_seniors['age_at_survey'] >= 75) & (df_seniors['age_at_survey'] < 85)],
        '85+': df_seniors[df_seniors['age_at_survey'] >= 85]
    }
    
    print("\nSenior Subgroup Analysis:")
    print("-" * 50)
    
    results = []
    for group, data in subgroups.items():
        prevalence = data['sdoh_two_yes'].mean()
        results.append({
            'Age Group': group,
            'N': len(data),
            'Prevalence': prevalence,
            'Percent of Seniors': len(data) / len(df_seniors) * 100
        })
        print(f"{group}: n={len(data):,}, prevalence={prevalence:.1%}")
    
    return pd.DataFrame(results)

def evaluate_thresholds_for_seniors(model, X_test, y_test, feature_names):
    """Evaluate different thresholds specifically for senior population"""
    print("\nEvaluating thresholds for senior clinic...")
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Test a range of thresholds
    thresholds = np.linspace(0.01, 0.50, 100)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        screening_rate = y_pred.mean()
        
        # Number needed to screen
        nns = 1 / ppv if ppv > 0 else np.inf
        
        # Calculate resource metrics
        total_screens = y_pred.sum()
        true_positives_found = tp
        false_alarms = fp
        missed_cases = fn
        
        results.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'screening_rate': screening_rate,
            'nns': nns,
            'total_screens': total_screens,
            'true_positives': true_positives_found,
            'false_positives': false_alarms,
            'false_negatives': missed_cases,
            'f1_score': 2 * (ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal thresholds for different objectives
    optimal_thresholds = {
        'balanced': results_df.iloc[results_df['f1_score'].idxmax()],
        'high_sensitivity': results_df[results_df['sensitivity'] >= 0.80].iloc[0] if any(results_df['sensitivity'] >= 0.80) else None,
        'high_ppv': results_df[results_df['ppv'] >= 0.25].iloc[0] if any(results_df['ppv'] >= 0.25) else None,
        'resource_constrained': results_df[results_df['screening_rate'] <= 0.20].iloc[-1],
        'standard_model': results_df.iloc[np.argmin(np.abs(results_df['threshold'] - 0.05))]
    }
    
    return results_df, optimal_thresholds

def create_senior_threshold_visualizations(results_df, optimal_thresholds, prevalence, output_dir='results/figures'):
    """Create comprehensive visualizations for senior clinic threshold analysis"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Threshold Analysis for Senior Clinic (Ages 65+)', fontsize=16, fontweight='bold')
    
    # 1. Sensitivity vs PPV trade-off
    ax1 = axes[0, 0]
    ax1.plot(results_df['threshold'], results_df['sensitivity'], 'b-', linewidth=2, label='Sensitivity')
    ax1.plot(results_df['threshold'], results_df['ppv'], 'r-', linewidth=2, label='PPV')
    
    # Mark optimal points
    for name, opt in optimal_thresholds.items():
        if opt is not None and name != 'standard_model':
            ax1.scatter(opt['threshold'], opt['sensitivity'], s=100, marker='o', 
                       edgecolors='black', linewidth=2, zorder=5)
            ax1.scatter(opt['threshold'], opt['ppv'], s=100, marker='s', 
                       edgecolors='black', linewidth=2, zorder=5)
    
    ax1.axhline(y=prevalence, color='gray', linestyle='--', alpha=0.5, 
               label=f'Baseline PPV ({prevalence:.1%})')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('A. Sensitivity vs PPV Trade-off', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1)
    
    # 2. Screening Rate and Resource Utilization
    ax2 = axes[0, 1]
    ax2_twin = ax2.twinx()
    
    # Screening rate on primary axis
    line1 = ax2.plot(results_df['threshold'], results_df['screening_rate'] * 100, 
                     'g-', linewidth=2, label='Screening Rate')
    ax2.set_ylabel('Screening Rate (%)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Number needed to screen on secondary axis
    line2 = ax2_twin.plot(results_df['threshold'], results_df['nns'], 
                         'm-', linewidth=2, label='Number Needed to Screen')
    ax2_twin.set_ylabel('Number Needed to Screen', color='m')
    ax2_twin.tick_params(axis='y', labelcolor='m')
    ax2_twin.set_ylim(0, 20)
    
    ax2.set_xlabel('Threshold')
    ax2.set_title('B. Resource Utilization Metrics', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.5)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='center right')
    
    # 3. Clinical Impact Analysis
    ax3 = axes[1, 0]
    
    # Calculate metrics per 1000 seniors
    n_per_1000 = 1000
    prevalence_n = int(prevalence * n_per_1000)
    
    scenarios = []
    for name, opt in optimal_thresholds.items():
        if opt is not None:
            screens = int(opt['screening_rate'] * n_per_1000)
            found = int(opt['sensitivity'] * prevalence_n)
            missed = prevalence_n - found
            false_alarms = screens - found
            
            scenarios.append({
                'Scenario': name.replace('_', ' ').title(),
                'Screens': screens,
                'Found': found,
                'Missed': missed,
                'False Alarms': false_alarms
            })
    
    scenarios_df = pd.DataFrame(scenarios)
    
    # Create grouped bar chart
    x = np.arange(len(scenarios_df))
    width = 0.2
    
    bars1 = ax3.bar(x - 1.5*width, scenarios_df['Screens'], width, label='Total Screened', color='skyblue')
    bars2 = ax3.bar(x - 0.5*width, scenarios_df['Found'], width, label='True Positives', color='green')
    bars3 = ax3.bar(x + 0.5*width, scenarios_df['Missed'], width, label='Missed Cases', color='red')
    bars4 = ax3.bar(x + 1.5*width, scenarios_df['False Alarms'], width, label='False Alarms', color='orange')
    
    # Add value labels
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    ax3.set_ylabel('Number of Patients (per 1,000)')
    ax3.set_title(f'C. Clinical Impact per 1,000 Senior Patients (Prevalence: {prevalence:.1%})', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios_df['Scenario'], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim(0, max(scenarios_df['Screens']) * 1.2)
    
    # 4. Recommendation Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create recommendation text
    rec_text = "RECOMMENDATIONS FOR SENIOR CLINICS\n" + "="*40 + "\n\n"
    
    # Primary recommendation
    balanced = optimal_thresholds['balanced']
    rec_text += f"PRIMARY RECOMMENDATION:\n"
    rec_text += f"Threshold: {balanced['threshold']:.3f}\n"
    rec_text += f"• Screens {balanced['screening_rate']:.1%} of patients\n"
    rec_text += f"• Identifies {balanced['sensitivity']:.1%} of SDOH cases\n"
    rec_text += f"• PPV of {balanced['ppv']:.1%} ({balanced['ppv']/prevalence:.1f}x baseline)\n"
    rec_text += f"• Number needed to screen: {balanced['nns']:.1f}\n\n"
    
    # Alternative scenarios
    rec_text += "ALTERNATIVE SCENARIOS:\n\n"
    
    if optimal_thresholds['resource_constrained'] is not None:
        rc = optimal_thresholds['resource_constrained']
        rec_text += f"Limited Resources (≤20% screening):\n"
        rec_text += f"• Threshold: {rc['threshold']:.3f}\n"
        rec_text += f"• Sensitivity: {rc['sensitivity']:.1%}\n"
        rec_text += f"• PPV: {rc['ppv']:.1%}\n\n"
    
    if optimal_thresholds['high_sensitivity'] is not None:
        hs = optimal_thresholds['high_sensitivity']
        rec_text += f"Maximum Coverage (≥80% sensitivity):\n"
        rec_text += f"• Threshold: {hs['threshold']:.3f}\n"
        rec_text += f"• Screens: {hs['screening_rate']:.1%}\n"
        rec_text += f"• PPV: {hs['ppv']:.1%}\n\n"
    
    # Comparison to general population
    std = optimal_thresholds['standard_model']
    rec_text += f"Standard Model (0.05 threshold):\n"
    rec_text += f"• Screens: {std['screening_rate']:.1%}\n"
    rec_text += f"• Sensitivity: {std['sensitivity']:.1%}\n"
    rec_text += f"• PPV: {std['ppv']:.1%}\n"
    
    ax4.text(0.05, 0.95, rec_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'senior_clinic_threshold_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(output_dir, 'senior_clinic_threshold_analysis.pdf'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Saved senior clinic analysis to {output_dir}")

def create_implementation_flowchart(optimal_threshold, output_dir='results/figures'):
    """Create implementation flowchart for senior clinics"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'SDOH Screening Implementation\nfor Senior Clinics', 
           ha='center', va='top', fontsize=16, fontweight='bold',
           transform=ax.transAxes)
    
    # Define box properties
    box_props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8, edgecolor='black')
    decision_props = dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='black')
    action_props = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8, edgecolor='black')
    
    # Create flowchart
    y_pos = 0.85
    
    # Step 1
    ax.text(0.5, y_pos, 'Senior Patient (Age ≥65)\nPresents to Clinic', 
           ha='center', va='center', fontsize=11, bbox=box_props,
           transform=ax.transAxes)
    
    # Arrow
    ax.annotate('', xy=(0.5, y_pos-0.08), xytext=(0.5, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    y_pos -= 0.10
    
    # Step 2
    ax.text(0.5, y_pos, 'Calculate SDOH Risk Score\nUsing EHR Data', 
           ha='center', va='center', fontsize=11, bbox=box_props,
           transform=ax.transAxes)
    
    # Arrow
    ax.annotate('', xy=(0.5, y_pos-0.08), xytext=(0.5, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    y_pos -= 0.10
    
    # Decision point
    ax.text(0.5, y_pos, f'Risk Score ≥ {optimal_threshold["threshold"]:.3f}?', 
           ha='center', va='center', fontsize=11, bbox=decision_props,
           transform=ax.transAxes)
    
    # Yes branch
    ax.annotate('YES', xy=(0.25, y_pos-0.08), xytext=(0.45, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    # No branch
    ax.annotate('NO', xy=(0.75, y_pos-0.08), xytext=(0.55, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    y_pos -= 0.12
    
    # High risk path
    ax.text(0.25, y_pos, 'Administer Full\nSDOH Screening', 
           ha='center', va='center', fontsize=10, bbox=action_props,
           transform=ax.transAxes)
    
    # Low risk path
    ax.text(0.75, y_pos, 'Standard Care\n(No screening)', 
           ha='center', va='center', fontsize=10, bbox=box_props,
           transform=ax.transAxes)
    
    # Continue high risk path
    ax.annotate('', xy=(0.25, y_pos-0.08), xytext=(0.25, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    y_pos -= 0.10
    
    # SDOH needs assessment
    ax.text(0.25, y_pos, 'SDOH Needs\nIdentified?', 
           ha='center', va='center', fontsize=10, bbox=decision_props,
           transform=ax.transAxes)
    
    # Yes/No branches
    ax.annotate('YES', xy=(0.10, y_pos-0.08), xytext=(0.20, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    ax.annotate('NO', xy=(0.40, y_pos-0.08), xytext=(0.30, y_pos-0.05),
               arrowprops=dict(arrowstyle='->', lw=2),
               transform=ax.transAxes)
    
    y_pos -= 0.10
    
    # Interventions
    ax.text(0.10, y_pos, 'Connect to:\n• Social Services\n• Community Resources\n• Care Coordination', 
           ha='center', va='center', fontsize=9, bbox=action_props,
           transform=ax.transAxes)
    
    # Document and continue
    ax.text(0.40, y_pos, 'Document Screening\nContinue Standard Care', 
           ha='center', va='center', fontsize=9, bbox=box_props,
           transform=ax.transAxes)
    
    # Add statistics box
    stats_text = f"Expected Outcomes (per 100 seniors):\n"
    stats_text += f"• {optimal_threshold['screening_rate']*100:.0f} patients screened\n"
    stats_text += f"• {optimal_threshold['sensitivity']*optimal_threshold['screening_rate']*100*0.066:.0f} true needs identified\n"
    stats_text += f"• {(1-optimal_threshold['ppv'])*optimal_threshold['screening_rate']*100:.0f} false positives\n"
    stats_text += f"• PPV: {optimal_threshold['ppv']:.1%}"
    
    ax.text(0.95, 0.25, stats_text, 
           ha='right', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           transform=ax.transAxes)
    
    # Add notes
    notes_text = "Notes:\n"
    notes_text += f"• Higher prevalence in seniors ({optimal_threshold['screening_rate']:.1%})\n"
    notes_text += "• Consider frailty and cognitive status\n"
    notes_text += "• Integrate with geriatric assessment\n"
    notes_text += "• Monitor for screening fatigue"
    
    ax.text(0.05, 0.25, notes_text, 
           ha='left', va='top', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8),
           transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, 'senior_clinic_implementation_flowchart.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print(f"Saved implementation flowchart to {output_dir}")

def generate_senior_clinic_report(df_seniors, subgroup_analysis, optimal_thresholds, output_dir='results/reports'):
    """Generate comprehensive report for senior clinic implementation"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'senior_clinic_threshold_recommendations.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SDOH SCREENING RECOMMENDATIONS FOR SENIOR CLINICS\n")
        f.write("Analysis for Patients Aged 65 and Older\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Senior patients (65+) have different SDOH risk profiles compared to the general\n")
        f.write(f"population. This analysis provides evidence-based threshold recommendations\n")
        f.write(f"specifically optimized for senior clinics.\n\n")
        
        f.write("Key Findings:\n")
        f.write(f"• SDOH prevalence in seniors: {df_seniors['sdoh_two_yes'].mean():.1%}\n")
        f.write(f"• Recommended threshold: {optimal_thresholds['balanced']['threshold']:.3f}\n")
        f.write(f"• Expected screening rate: {optimal_thresholds['balanced']['screening_rate']:.1%}\n")
        f.write(f"• Expected PPV: {optimal_thresholds['balanced']['ppv']:.1%}\n")
        f.write(f"• Sensitivity: {optimal_thresholds['balanced']['sensitivity']:.1%}\n\n")
        
        # Population Characteristics
        f.write("SENIOR POPULATION CHARACTERISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total senior patients analyzed: {len(df_seniors):,}\n")
        f.write(f"Age distribution:\n")
        for _, row in subgroup_analysis.iterrows():
            f.write(f"  • {row['Age Group']}: {row['N']:,} ({row['Percent of Seniors']:.1f}%)")
            f.write(f" - SDOH prevalence: {row['Prevalence']:.1%}\n")
        
        # Threshold Recommendations
        f.write("\n\nTHRESHOLD RECOMMENDATIONS\n")
        f.write("="*40 + "\n\n")
        
        f.write("1. BALANCED APPROACH (Recommended)\n")
        f.write("-"*30 + "\n")
        balanced = optimal_thresholds['balanced']
        f.write(f"Threshold: {balanced['threshold']:.4f}\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  • Screening rate: {balanced['screening_rate']:.1%}\n")
        f.write(f"  • Sensitivity: {balanced['sensitivity']:.1%}\n")
        f.write(f"  • Specificity: {balanced['specificity']:.1%}\n")
        f.write(f"  • PPV: {balanced['ppv']:.1%}\n")
        f.write(f"  • NPV: {balanced['npv']:.1%}\n")
        f.write(f"  • Number needed to screen: {balanced['nns']:.1f}\n")
        f.write(f"Resource Impact (per 1,000 seniors):\n")
        f.write(f"  • Patients screened: {int(balanced['screening_rate'] * 1000)}\n")
        f.write(f"  • True positives found: {int(balanced['sensitivity'] * 66)}\n")  # Assuming ~6.6% prevalence
        f.write(f"  • False positives: {int((1-balanced['ppv']) * balanced['screening_rate'] * 1000)}\n\n")
        
        f.write("2. RESOURCE-CONSTRAINED SETTING\n")
        f.write("-"*30 + "\n")
        if optimal_thresholds['resource_constrained'] is not None:
            rc = optimal_thresholds['resource_constrained']
            f.write(f"Threshold: {rc['threshold']:.4f}\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  • Screening rate: {rc['screening_rate']:.1%}\n")
            f.write(f"  • Sensitivity: {rc['sensitivity']:.1%}\n")
            f.write(f"  • PPV: {rc['ppv']:.1%}\n")
            f.write(f"  • Number needed to screen: {rc['nns']:.1f}\n\n")
        
        f.write("3. HIGH SENSITIVITY APPROACH\n")
        f.write("-"*30 + "\n")
        if optimal_thresholds['high_sensitivity'] is not None:
            hs = optimal_thresholds['high_sensitivity']
            f.write(f"Threshold: {hs['threshold']:.4f}\n")
            f.write(f"Performance Metrics:\n")
            f.write(f"  • Screening rate: {hs['screening_rate']:.1%}\n")
            f.write(f"  • Sensitivity: {hs['sensitivity']:.1%}\n")
            f.write(f"  • PPV: {hs['ppv']:.1%}\n")
            f.write(f"  • Number needed to screen: {hs['nns']:.1f}\n\n")
        
        # Implementation Considerations
        f.write("\nIMPLEMENTATION CONSIDERATIONS\n")
        f.write("="*40 + "\n\n")
        
        f.write("1. Senior-Specific Factors:\n")
        f.write("   • Higher baseline SDOH prevalence\n")
        f.write("   • Multiple chronic conditions common\n")
        f.write("   • Fixed income considerations\n")
        f.write("   • Transportation barriers more prevalent\n")
        f.write("   • Social isolation risks\n\n")
        
        f.write("2. Workflow Integration:\n")
        f.write("   • Align with Medicare Annual Wellness Visit\n")
        f.write("   • Coordinate with geriatric assessments\n")
        f.write("   • Consider cognitive screening timing\n")
        f.write("   • Integrate with care transitions\n\n")
        
        f.write("3. Resource Requirements:\n")
        f.write(f"   • Estimated {balanced['screening_rate']:.0%} of appointments need screening\n")
        f.write(f"   • Average screening time: 15-20 minutes\n")
        f.write(f"   • Social worker consultations for {balanced['ppv']:.0%} of screened\n")
        f.write(f"   • Community resource connections needed\n\n")
        
        # Quality Metrics
        f.write("QUALITY METRICS TO MONITOR\n")
        f.write("-"*40 + "\n")
        f.write("1. Process Measures:\n")
        f.write("   • Screening completion rate\n")
        f.write("   • Time to intervention after positive screen\n")
        f.write("   • Referral completion rates\n\n")
        
        f.write("2. Outcome Measures:\n")
        f.write("   • Reduction in ED visits\n")
        f.write("   • Improvement in medication adherence\n")
        f.write("   • Patient satisfaction scores\n")
        f.write("   • Caregiver burden assessment\n\n")
        
        # Recommendations
        f.write("FINAL RECOMMENDATIONS\n")
        f.write("="*40 + "\n")
        f.write(f"1. Implement threshold of {balanced['threshold']:.3f} for senior clinics\n")
        f.write("2. Train staff on senior-specific SDOH issues\n")
        f.write("3. Establish partnerships with Area Agencies on Aging\n")
        f.write("4. Monitor performance monthly for first 6 months\n")
        f.write("5. Adjust threshold based on resource availability\n")
        f.write("6. Consider pilot in one clinic before full rollout\n")
    
    print(f"Senior clinic report saved to: {report_path}")

def main():
    """Run comprehensive senior clinic threshold analysis"""
    print("="*80)
    print("SENIOR CLINIC THRESHOLD ANALYSIS")
    print("="*80)
    
    # Load senior data
    df_seniors = load_senior_data()
    
    # Analyze subgroups
    subgroup_analysis = analyze_senior_subgroups(df_seniors)
    
    # Load model
    print("\nLoading calibrated model...")
    model = joblib.load('models/xgboost_scientific_calibrated.joblib')
    
    with open('models/scientific_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Prepare data for model
    feature_cols = [col for col in df_seniors.columns if col not in 
                   ['person_id', 'payor_id', 'mbi_id', 'sdoh_two_yes',
                    'race_category', 'ethnicity_category']]
    
    X_seniors = df_seniors[feature_cols]
    y_seniors = df_seniors['sdoh_two_yes']
    
    # Evaluate thresholds
    results_df, optimal_thresholds = evaluate_thresholds_for_seniors(
        model, X_seniors, y_seniors, feature_cols
    )
    
    # Create visualizations
    print("\nCreating visualizations...")
    prevalence = y_seniors.mean()
    create_senior_threshold_visualizations(results_df, optimal_thresholds, prevalence)
    
    # Create implementation flowchart
    print("\nCreating implementation flowchart...")
    create_implementation_flowchart(optimal_thresholds['balanced'])
    
    # Generate report
    print("\nGenerating comprehensive report...")
    generate_senior_clinic_report(df_seniors, subgroup_analysis, optimal_thresholds)
    
    print("\n" + "="*80)
    print("SENIOR CLINIC ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings:")
    print(f"• SDOH prevalence in seniors: {prevalence:.1%}")
    print(f"• Recommended threshold: {optimal_thresholds['balanced']['threshold']:.3f}")
    print(f"• Expected screening rate: {optimal_thresholds['balanced']['screening_rate']:.1%}")
    print(f"• Expected PPV: {optimal_thresholds['balanced']['ppv']:.1%}")
    print("\nOutputs generated:")
    print("1. Threshold analysis visualization")
    print("2. Implementation flowchart")
    print("3. Comprehensive recommendations report")
    print("\nAll outputs saved to results/figures/ and results/reports/")

if __name__ == "__main__":
    main()