#!/usr/bin/env python3
"""
Create SHAP plots with professional feature labels
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import xgboost as xgb
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set professional plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Professional feature name mapping (same as before)
FEATURE_NAME_MAPPING = {
    # SVI Theme 1: Socioeconomic Status
    'ep_pov150': 'Poverty Rate (% Below 150% Poverty Line) - SVI',
    'ep_unemp': 'Unemployment Rate (%) - SVI',
    'ep_hburd': 'Housing Cost Burden (% Spending >30% Income) - SVI',
    'ep_nohsdp': 'No High School Diploma (% Adults 25+) - SVI',
    'ep_uninsur': 'Uninsured Rate (% Under 65) - SVI',
    'epl_pov150': 'Poverty Percentile Ranking - SVI',
    'epl_unemp': 'Unemployment Percentile Ranking - SVI',
    'epl_hburd': 'Housing Burden Percentile Ranking - SVI',
    'epl_nohsdp': 'Education Percentile Ranking - SVI',
    'epl_uninsur': 'Uninsured Percentile Ranking - SVI',
    'spl_theme1': 'Overall Socioeconomic Vulnerability Score - SVI',
    'rpl_theme1': 'Socioeconomic Status Percentile Ranking - SVI',
    
    # SVI Theme 2: Household Composition & Disability
    'ep_age65': 'Elderly Population (% Age 65+) - SVI',
    'ep_age17': 'Youth Population (% Age 17 & Under) - SVI',
    'ep_disabl': 'Disability Rate (% Civilian Noninstitutionalized) - SVI',
    'ep_sngpnt': 'Single-Parent Households (% With Children) - SVI',
    'ep_limeng': 'Limited English Proficiency (% Age 5+) - SVI',
    'epl_age65': 'Elderly Population Percentile Ranking - SVI',
    'epl_age17': 'Youth Population Percentile Ranking - SVI',
    'epl_disabl': 'Disability Percentile Ranking - SVI',
    'epl_sngpnt': 'Single-Parent Percentile Ranking - SVI',
    'epl_limeng': 'Limited English Percentile Ranking - SVI',
    'spl_theme2': 'Overall Household Composition Score - SVI',
    'rpl_theme2': 'Household Composition Percentile Ranking - SVI',
    
    # SVI Theme 3: Housing Type & Transportation
    'ep_munit': 'Multi-Unit Housing (% 10+ Units) - SVI',
    'ep_mobile': 'Mobile Homes (%) - SVI',
    'ep_crowd': 'Overcrowded Housing (>1 Person/Room) - SVI',
    'ep_noveh': 'No Vehicle Available (% Households) - SVI',
    'ep_groupq': 'Group Quarters Population (%) - SVI',
    'epl_munit': 'Multi-Unit Housing Percentile Ranking - SVI',
    'epl_mobile': 'Mobile Homes Percentile Ranking - SVI',
    'epl_crowd': 'Overcrowding Percentile Ranking - SVI',
    'epl_noveh': 'No Vehicle Percentile Ranking - SVI',
    'epl_groupq': 'Group Quarters Percentile Ranking - SVI',
    'spl_theme3': 'Overall Housing/Transportation Score - SVI',
    'rpl_theme3': 'Housing/Transportation Percentile Ranking - SVI',
    
    # SVI Theme 4: Racial & Ethnic Minority Status
    'ep_minrty': 'Minority Population (% Non-White) - SVI',
    'epl_minrty': 'Minority Population Percentile Ranking - SVI',
    'spl_theme4': 'Overall Minority Status Score - SVI',
    'rpl_theme4': 'Minority Status Percentile Ranking - SVI',
    
    # Overall SVI
    'spl_themes': 'Overall Social Vulnerability Score - SVI',
    'rpl_themes': 'Overall Social Vulnerability Percentile - SVI',
    
    # ADI (Area Deprivation Index)
    'adi_natrank': 'Area Deprivation Index National Ranking - ADI',
    'adi_staterank': 'Area Deprivation Index State Ranking - ADI',
    
    # Demographics
    'age_at_survey': 'Patient Age at Survey',
    'sex_female': 'Sex (Female = 1)',
    
    # Other features
    'e_totpop': 'Total Population in Census Tract - SVI',
    'e_hu': 'Total Housing Units in Census Tract - SVI',
    'e_hh': 'Total Households in Census Tract - SVI',
    'a_totpop': 'Total Area Population - SVI',
    'a_hu': 'Total Area Housing Units - SVI',
    'a_hh': 'Total Area Households - SVI',
}

def get_professional_feature_name(feature):
    """Convert technical feature name to professional label"""
    return FEATURE_NAME_MAPPING.get(feature, feature)

def create_professional_shap_plots():
    """Create SHAP plots with professional labels"""
    print("Loading model and data...")
    
    # Load model
    base_model = xgb.XGBClassifier()
    base_model.load_model('models/xgboost_scientific_base.json')
    
    # Load a sample of data
    df = pd.read_csv('/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv', nrows=1000)
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in 
                   ['person_id', 'payor_id', 'mbi_id', 'sdoh_two_yes', 'race_category', 'ethnicity_category']]
    
    X_sample = df[feature_cols]
    y_sample = df['sdoh_two_yes']
    
    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer(X_sample)
    
    # Create professional feature names
    professional_names = [get_professional_feature_name(f) for f in feature_cols]
    
    # 1. Summary Plot with Professional Labels
    print("Creating SHAP summary plot...")
    plt.figure(figsize=(12, 10))
    
    # Create custom summary plot
    shap_df = pd.DataFrame(shap_values.values, columns=professional_names)
    feature_importance = np.abs(shap_df).mean().sort_values(ascending=False).head(20)
    
    # Create summary plot for top 20 features
    top_features = feature_importance.index.tolist()
    top_indices = [professional_names.index(f) for f in top_features]
    
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values.values[:, top_indices], 
                     X_sample.iloc[:, top_indices], 
                     feature_names=[professional_names[i] for i in top_indices],
                     show=False)
    
    plt.title('SHAP Summary Plot: Impact of Features on SDOH Risk Prediction', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/figures/shap_summary_professional.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Waterfall plot for a high-risk patient
    print("Creating waterfall plots...")
    
    # Find a high-risk patient
    predictions = base_model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(predictions)
    
    plt.figure(figsize=(10, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[high_risk_idx].values,
                                        base_values=shap_values[high_risk_idx].base_values,
                                        data=X_sample.iloc[high_risk_idx],
                                        feature_names=professional_names),
                       max_display=15, show=False)
    
    plt.title(f'SHAP Waterfall Plot: High-Risk Patient (Risk Score: {predictions[high_risk_idx]:.1%})', 
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/figures/shap_waterfall_high_risk_professional.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dependence plots for key features
    print("Creating dependence plots...")
    
    # Create a 2x2 grid of dependence plots for top features
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('SHAP Dependence Plots: How Key Features Impact Risk', 
                 fontsize=16, fontweight='bold')
    
    key_features = ['rpl_themes', 'age_at_survey', 'rpl_theme1', 'adi_natrank']
    
    for idx, (ax, feature) in enumerate(zip(axes.flat, key_features)):
        if feature in feature_cols:
            feature_idx = feature_cols.index(feature)
            professional_name = get_professional_feature_name(feature)
            
            # Get interaction feature (most correlated)
            interaction_idx = None
            
            shap.dependence_plot(feature_idx, shap_values.values, X_sample,
                               feature_names=professional_names,
                               interaction_index=interaction_idx,
                               ax=ax, show=False)
            
            ax.set_xlabel(professional_name, fontsize=11)
            ax.set_ylabel('SHAP Value', fontsize=11)
            ax.set_title(f'{["A", "B", "C", "D"][idx]}. {professional_name.split(" - ")[0]}', 
                        fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/figures/shap_dependence_professional.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Force plot for average patient
    print("Creating force plot visualization...")
    
    # Use median values patient
    median_idx = np.argmin(np.abs(predictions - np.median(predictions)))
    
    # Create force plot (saves as HTML)
    force_plot = shap.force_plot(shap_values[median_idx].base_values,
                                shap_values[median_idx].values,
                                X_sample.iloc[median_idx],
                                feature_names=professional_names,
                                out_names="SDOH Risk Score",
                                matplotlib=True, show=False)
    
    plt.figure(figsize=(20, 5))
    plt.title(f'SHAP Force Plot: Average Risk Patient (Risk Score: {predictions[median_idx]:.1%})', 
             fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('results/figures/shap_force_plot_professional.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Professional SHAP plots created successfully!")
    
    # Create interpretation guide
    create_shap_interpretation_image()

def create_shap_interpretation_image():
    """Create a visual guide for interpreting SHAP plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('How to Interpret SHAP Plots', fontsize=20, fontweight='bold')
    
    # Remove axes for text panels
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axis('off')
    
    # Panel 1: Summary Plot
    ax1.text(0.5, 0.9, 'SHAP Summary Plot', fontsize=16, fontweight='bold', 
            ha='center', transform=ax1.transAxes)
    
    explanation1 = """• Each row is a feature
• Each dot is a patient
• Position shows impact (right = increases risk, left = decreases risk)
• Color shows feature value (red = high, blue = low)
• Features ranked by overall importance

Example: If "Poverty Rate" shows red dots on the right,
it means high poverty areas increase SDOH risk."""
    
    ax1.text(0.05, 0.7, explanation1, fontsize=12, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Panel 2: Waterfall Plot
    ax2.text(0.5, 0.9, 'SHAP Waterfall Plot', fontsize=16, fontweight='bold',
            ha='center', transform=ax2.transAxes)
    
    explanation2 = """• Shows one patient's risk calculation
• Starts from baseline (average risk = 6.6%)
• Each bar shows feature contribution
• Red bars increase risk
• Blue bars decrease risk
• Final value is patient's predicted risk

Example: Patient in high poverty area (+3%),
young age (+2%), results in 11.6% risk."""
    
    ax2.text(0.05, 0.7, explanation2, fontsize=12, transform=ax2.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Panel 3: Dependence Plot
    ax3.text(0.5, 0.9, 'SHAP Dependence Plot', fontsize=16, fontweight='bold',
            ha='center', transform=ax3.transAxes)
    
    explanation3 = """• Shows relationship between feature and impact
• X-axis: Feature value (e.g., age 18-90)
• Y-axis: SHAP value (impact on risk)
• Each dot is a patient
• Color may show interaction with another feature

Example: Risk decreases with age - younger patients
have higher SDOH risk than elderly."""
    
    ax3.text(0.05, 0.7, explanation3, fontsize=12, transform=ax3.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    # Panel 4: Key Insights
    ax4.text(0.5, 0.9, 'Key Insights from SHAP', fontsize=16, fontweight='bold',
            ha='center', transform=ax4.transAxes)
    
    insights = """✓ Transparency: See exactly why each patient is flagged

✓ Fairness: Verify model uses appropriate factors

✓ Actionability: Identify modifiable risk factors

✓ Trust: Clinicians understand AI reasoning

✓ Quality: Monitor for unexpected patterns"""
    
    ax4.text(0.05, 0.7, insights, fontsize=13, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/figures/shap_interpretation_guide.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Create all professional visualizations"""
    print("Creating professional SHAP visualizations...")
    
    # Create output directory if needed
    Path('results/figures').mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    create_professional_shap_plots()
    
    print("\nAll professional visualizations created!")
    print("Files saved in results/figures/")

if __name__ == "__main__":
    main()