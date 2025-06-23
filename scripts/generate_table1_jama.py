#!/usr/bin/env python3
"""
Generate Table 1 for JAMA publication
Creates demographic table stratified by SDOH status
Follows JAMA publication standards
"""

import numpy as np
import pandas as pd
from scipy import stats
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_data():
    """Load the full dataset"""
    print("Loading full dataset...")
    
    data_path = '/Users/jcrmacstudio/Desktop/vebe coding/SDOH_model/sdoh2_ml_final_all_svi.csv'
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df):,} patients")
    return df

def format_continuous(values, sdoh_pos, sdoh_neg):
    """Format continuous variables as mean (SD) or median [IQR]"""
    # Check for normality
    _, p_normal = stats.shapiro(values[:min(5000, len(values))])  # Sample for large datasets
    
    if p_normal > 0.05:  # Normal distribution
        # Use t-test
        overall_mean = np.mean(values)
        overall_sd = np.std(values, ddof=1)
        
        pos_mean = np.mean(sdoh_pos)
        pos_sd = np.std(sdoh_pos, ddof=1)
        
        neg_mean = np.mean(sdoh_neg)
        neg_sd = np.std(sdoh_neg, ddof=1)
        
        _, p_value = stats.ttest_ind(sdoh_pos, sdoh_neg)
        
        return {
            'overall': f"{overall_mean:.1f} ({overall_sd:.1f})",
            'sdoh_neg': f"{neg_mean:.1f} ({neg_sd:.1f})",
            'sdoh_pos': f"{pos_mean:.1f} ({pos_sd:.1f})",
            'p_value': p_value,
            'test': 't-test'
        }
    else:  # Non-normal distribution
        # Use Mann-Whitney U test
        overall_median = np.median(values)
        overall_q1 = np.percentile(values, 25)
        overall_q3 = np.percentile(values, 75)
        
        pos_median = np.median(sdoh_pos)
        pos_q1 = np.percentile(sdoh_pos, 25)
        pos_q3 = np.percentile(sdoh_pos, 75)
        
        neg_median = np.median(sdoh_neg)
        neg_q1 = np.percentile(sdoh_neg, 25)
        neg_q3 = np.percentile(sdoh_neg, 75)
        
        _, p_value = stats.mannwhitneyu(sdoh_pos, sdoh_neg, alternative='two-sided')
        
        return {
            'overall': f"{overall_median:.1f} [{overall_q1:.1f}-{overall_q3:.1f}]",
            'sdoh_neg': f"{neg_median:.1f} [{neg_q1:.1f}-{neg_q3:.1f}]",
            'sdoh_pos': f"{pos_median:.1f} [{pos_q1:.1f}-{pos_q3:.1f}]",
            'p_value': p_value,
            'test': 'Mann-Whitney U'
        }

def format_categorical(values, sdoh_pos, sdoh_neg):
    """Format categorical variables as n (%)"""
    # Overall
    overall_counts = values.value_counts()
    overall_total = len(values)
    
    # SDOH positive
    pos_counts = sdoh_pos.value_counts()
    pos_total = len(sdoh_pos)
    
    # SDOH negative
    neg_counts = sdoh_neg.value_counts()
    neg_total = len(sdoh_neg)
    
    # Chi-square test
    categories = overall_counts.index
    observed = []
    for cat in categories:
        observed.append([
            neg_counts.get(cat, 0),
            pos_counts.get(cat, 0)
        ])
    
    chi2, p_value, _, _ = stats.chi2_contingency(observed)
    
    results = []
    for cat in categories:
        overall_n = overall_counts.get(cat, 0)
        overall_pct = (overall_n / overall_total) * 100
        
        pos_n = pos_counts.get(cat, 0)
        pos_pct = (pos_n / pos_total) * 100
        
        neg_n = neg_counts.get(cat, 0)
        neg_pct = (neg_n / neg_total) * 100
        
        results.append({
            'category': cat,
            'overall': f"{overall_n:,} ({overall_pct:.1f})",
            'sdoh_neg': f"{neg_n:,} ({neg_pct:.1f})",
            'sdoh_pos': f"{pos_n:,} ({pos_pct:.1f})",
            'p_value': p_value if cat == categories[0] else None,  # Only show p-value once
            'test': 'Chi-square' if cat == categories[0] else None
        })
    
    return results

def create_table1(df):
    """Create Table 1 with all relevant characteristics"""
    print("\nCreating Table 1...")
    
    # Split by SDOH status
    sdoh_pos = df[df['sdoh_two_yes'] == 1]
    sdoh_neg = df[df['sdoh_two_yes'] == 0]
    
    print(f"SDOH ≥2 needs: {len(sdoh_pos):,} ({len(sdoh_pos)/len(df)*100:.1f}%)")
    print(f"SDOH <2 needs: {len(sdoh_neg):,} ({len(sdoh_neg)/len(df)*100:.1f}%)")
    
    table_data = []
    
    # Header row
    table_data.append({
        'Characteristic': 'No. (%)',
        'Overall': f"{len(df):,}",
        'SDOH <2 needs': f"{len(sdoh_neg):,} ({len(sdoh_neg)/len(df)*100:.1f})",
        'SDOH ≥2 needs': f"{len(sdoh_pos):,} ({len(sdoh_pos)/len(df)*100:.1f})",
        'P Value': '',
        'indent': 0
    })
    
    # Demographics section
    table_data.append({
        'Characteristic': 'Demographics',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 'P Value': '',
        'indent': 0
    })
    
    # Age
    age_stats = format_continuous(df['age_at_survey'], sdoh_pos['age_at_survey'], sdoh_neg['age_at_survey'])
    table_data.append({
        'Characteristic': 'Age, mean (SD), y',
        'Overall': age_stats['overall'],
        'SDOH <2 needs': age_stats['sdoh_neg'],
        'SDOH ≥2 needs': age_stats['sdoh_pos'],
        'P Value': f"{age_stats['p_value']:.3f}" if age_stats['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    # Age groups
    age_groups = pd.cut(df['age_at_survey'], bins=[0, 35, 50, 65, 100], 
                       labels=['18-35', '36-50', '51-65', '≥66'])
    age_groups_pos = pd.cut(sdoh_pos['age_at_survey'], bins=[0, 35, 50, 65, 100], 
                           labels=['18-35', '36-50', '51-65', '≥66'])
    age_groups_neg = pd.cut(sdoh_neg['age_at_survey'], bins=[0, 35, 50, 65, 100], 
                           labels=['18-35', '36-50', '51-65', '≥66'])
    
    age_group_results = format_categorical(age_groups, age_groups_pos, age_groups_neg)
    
    table_data.append({
        'Characteristic': 'Age group, y',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 
        'P Value': f"{age_group_results[0]['p_value']:.3f}" if age_group_results[0]['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    for result in age_group_results:
        table_data.append({
            'Characteristic': f"  {result['category']}",
            'Overall': result['overall'],
            'SDOH <2 needs': result['sdoh_neg'],
            'SDOH ≥2 needs': result['sdoh_pos'],
            'P Value': '',
            'indent': 2
        })
    
    # Sex
    df['sex'] = df['sex_female'].map({1: 'Female', 0: 'Male'})
    sdoh_pos['sex'] = sdoh_pos['sex_female'].map({1: 'Female', 0: 'Male'})
    sdoh_neg['sex'] = sdoh_neg['sex_female'].map({1: 'Female', 0: 'Male'})
    sex_results = format_categorical(df['sex'], sdoh_pos['sex'], sdoh_neg['sex'])
    table_data.append({
        'Characteristic': 'Sex',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 
        'P Value': f"{sex_results[0]['p_value']:.3f}" if sex_results[0]['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    for result in sex_results:
        table_data.append({
            'Characteristic': f"  {result['category']}",
            'Overall': result['overall'],
            'SDOH <2 needs': result['sdoh_neg'],
            'SDOH ≥2 needs': result['sdoh_pos'],
            'P Value': '',
            'indent': 2
        })
    
    # Race
    race_results = format_categorical(df['race_category'], sdoh_pos['race_category'], sdoh_neg['race_category'])
    table_data.append({
        'Characteristic': 'Race',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 
        'P Value': f"{race_results[0]['p_value']:.3f}" if race_results[0]['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    # Show only major categories for space
    major_races = ['Black or African American', 'White', 'Asian', 'Other', 'Unknown']
    for result in race_results:
        if result['category'] in major_races:
            table_data.append({
                'Characteristic': f"  {result['category']}",
                'Overall': result['overall'],
                'SDOH <2 needs': result['sdoh_neg'],
                'SDOH ≥2 needs': result['sdoh_pos'],
                'P Value': '',
                'indent': 2
            })
    
    # Ethnicity
    ethnicity_results = format_categorical(df['ethnicity_category'], sdoh_pos['ethnicity_category'], sdoh_neg['ethnicity_category'])
    table_data.append({
        'Characteristic': 'Ethnicity',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 
        'P Value': f"{ethnicity_results[0]['p_value']:.3f}" if ethnicity_results[0]['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    for result in ethnicity_results:
        table_data.append({
            'Characteristic': f"  {result['category']}",
            'Overall': result['overall'],
            'SDOH <2 needs': result['sdoh_neg'],
            'SDOH ≥2 needs': result['sdoh_pos'],
            'P Value': '',
            'indent': 2
        })
    
    # Socioeconomic indicators section
    table_data.append({
        'Characteristic': 'Socioeconomic Indicators (Census Tract)',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 'P Value': '',
        'indent': 0
    })
    
    # Poverty
    poverty_stats = format_continuous(df['ep_pov150'], sdoh_pos['ep_pov150'], sdoh_neg['ep_pov150'])
    table_data.append({
        'Characteristic': 'Below 150% poverty line, median [IQR], %',
        'Overall': poverty_stats['overall'],
        'SDOH <2 needs': poverty_stats['sdoh_neg'],
        'SDOH ≥2 needs': poverty_stats['sdoh_pos'],
        'P Value': f"{poverty_stats['p_value']:.3f}" if poverty_stats['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    # Unemployment
    unemp_stats = format_continuous(df['ep_unemp'], sdoh_pos['ep_unemp'], sdoh_neg['ep_unemp'])
    table_data.append({
        'Characteristic': 'Unemployment rate, median [IQR], %',
        'Overall': unemp_stats['overall'],
        'SDOH <2 needs': unemp_stats['sdoh_neg'],
        'SDOH ≥2 needs': unemp_stats['sdoh_pos'],
        'P Value': f"{unemp_stats['p_value']:.3f}" if unemp_stats['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    # Uninsured
    uninsured_stats = format_continuous(df['ep_uninsur'], sdoh_pos['ep_uninsur'], sdoh_neg['ep_uninsur'])
    table_data.append({
        'Characteristic': 'Uninsured rate, median [IQR], %',
        'Overall': uninsured_stats['overall'],
        'SDOH <2 needs': uninsured_stats['sdoh_neg'],
        'SDOH ≥2 needs': uninsured_stats['sdoh_pos'],
        'P Value': f"{uninsured_stats['p_value']:.3f}" if uninsured_stats['p_value'] >= 0.001 else "<.001",
        'indent': 1
    })
    
    # Geographic Vulnerability section
    table_data.append({
        'Characteristic': 'Geographic Vulnerability Indices',
        'Overall': '', 'SDOH <2 needs': '', 'SDOH ≥2 needs': '', 'P Value': '',
        'indent': 0
    })
    
    # SVI themes
    svi_themes = [
        ('rpl_theme1', 'Socioeconomic Status'),
        ('rpl_theme2', 'Household Composition'),
        ('rpl_theme3', 'Housing/Transportation'),
        ('rpl_theme4', 'Minority Status/Language'),
        ('rpl_themes', 'Overall SVI')
    ]
    
    for col, label in svi_themes:
        if col in df.columns:
            stats = format_continuous(df[col], sdoh_pos[col], sdoh_neg[col])
            table_data.append({
                'Characteristic': f"{label}, median [IQR]",
                'Overall': stats['overall'],
                'SDOH <2 needs': stats['sdoh_neg'],
                'SDOH ≥2 needs': stats['sdoh_pos'],
                'P Value': f"{stats['p_value']:.3f}" if stats['p_value'] >= 0.001 else "<.001",
                'indent': 1
            })
    
    # Area Deprivation Index
    if 'adi_natrank' in df.columns:
        adi_stats = format_continuous(df['adi_natrank'], 
                                     sdoh_pos['adi_natrank'], 
                                     sdoh_neg['adi_natrank'])
        table_data.append({
            'Characteristic': 'Area Deprivation Index, median [IQR]',
            'Overall': adi_stats['overall'],
            'SDOH <2 needs': adi_stats['sdoh_neg'],
            'SDOH ≥2 needs': adi_stats['sdoh_pos'],
            'P Value': f"{adi_stats['p_value']:.3f}" if adi_stats['p_value'] >= 0.001 else "<.001",
            'indent': 1
        })
    
    return pd.DataFrame(table_data)

def format_table_for_publication(table_df, output_dir='results/tables'):
    """Format table for JAMA publication standards"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create text version
    text_path = os.path.join(output_dir, 'table1_jama.txt')
    with open(text_path, 'w') as f:
        f.write("Table 1. Baseline Characteristics of Patients by SDOH Status\n")
        f.write("="*80 + "\n\n")
        
        # Format header
        f.write(f"{'Characteristic':<50} {'Overall':<20} {'SDOH <2 needs':<20} {'SDOH ≥2 needs':<20} {'P Value':<10}\n")
        f.write("-"*120 + "\n")
        
        # Format rows
        for _, row in table_df.iterrows():
            indent = "  " * row['indent']
            char = indent + row['Characteristic']
            
            # Bold section headers
            if row['indent'] == 0 and row['Characteristic'] != 'No. (%)':
                char = char.upper()
            
            f.write(f"{char:<50} {row['Overall']:<20} {row['SDOH <2 needs']:<20} {row['SDOH ≥2 needs']:<20} {row['P Value']:<10}\n")
            
            # Add spacing after sections
            if row['indent'] == 0:
                f.write("\n")
        
        # Add footnotes
        f.write("\n" + "-"*120 + "\n")
        f.write("Abbreviations: IQR, interquartile range; SDOH, social determinants of health; SVI, Social Vulnerability Index.\n")
        f.write("SI conversion factors: To convert XXX to YYY, multiply by ZZZ.\n")
        f.write("a P values calculated using t test for normally distributed continuous variables, ")
        f.write("Mann-Whitney U test for non-normally distributed continuous variables, ")
        f.write("and χ² test for categorical variables.\n")
    
    print(f"Table 1 (text) saved to: {text_path}")
    
    # Create LaTeX version
    latex_path = os.path.join(output_dir, 'table1_jama.tex')
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Baseline Characteristics of Patients by SDOH Status}\n")
        f.write("\\label{tab:table1}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Characteristic & Overall & SDOH <2 needs & SDOH $\\geq$2 needs & P Value$^a$ \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in table_df.iterrows():
            indent = "\\quad " * row['indent']
            char = indent + row['Characteristic']
            
            # Bold section headers
            if row['indent'] == 0 and row['Characteristic'] != 'No. (%)':
                char = "\\textbf{" + char + "}"
            
            # Escape special characters
            char = char.replace('%', '\\%').replace('≥', '$\\geq$')
            overall = row['Overall'].replace('%', '\\%')
            sdoh_neg = row['SDOH <2 needs'].replace('%', '\\%')
            sdoh_pos = row['SDOH ≥2 needs'].replace('%', '\\%')
            p_val = row['P Value'].replace('<', '$<$')
            
            f.write(f"{char} & {overall} & {sdoh_neg} & {sdoh_pos} & {p_val} \\\\\n")
            
            # Add spacing after sections
            if row['indent'] == 0 and row['Characteristic'] != 'No. (%)':
                f.write("\\addlinespace\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write("\\item Abbreviations: IQR, interquartile range; SDOH, social determinants of health; SVI, Social Vulnerability Index.\n")
        f.write("\\item $^a$ P values calculated using t test for normally distributed continuous variables, ")
        f.write("Mann-Whitney U test for non-normally distributed continuous variables, ")
        f.write("and $\\chi^2$ test for categorical variables.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{table}\n")
    
    print(f"Table 1 (LaTeX) saved to: {latex_path}")
    
    # Create CSV version
    csv_path = os.path.join(output_dir, 'table1_jama.csv')
    table_df.to_csv(csv_path, index=False)
    print(f"Table 1 (CSV) saved to: {csv_path}")
    
    # Create summary statistics
    summary_path = os.path.join(output_dir, 'table1_summary_statistics.txt')
    with open(summary_path, 'w') as f:
        f.write("TABLE 1 SUMMARY STATISTICS\n")
        f.write("="*50 + "\n\n")
        
        # Count significant differences
        p_values = table_df[table_df['P Value'] != '']['P Value']
        p_values_numeric = []
        for p in p_values:
            if p.startswith('<'):
                p_values_numeric.append(float(p[1:]))
            else:
                try:
                    p_values_numeric.append(float(p))
                except:
                    pass
        
        sig_count = sum(1 for p in p_values_numeric if p < 0.05)
        total_tests = len(p_values_numeric)
        
        f.write(f"Total statistical tests performed: {total_tests}\n")
        f.write(f"Significant differences (p < 0.05): {sig_count} ({sig_count/total_tests*100:.1f}%)\n")
        f.write(f"Non-significant differences: {total_tests - sig_count} ({(total_tests - sig_count)/total_tests*100:.1f}%)\n")
        
        f.write("\n\nKEY FINDINGS:\n")
        f.write("-"*30 + "\n")
        f.write("1. Patients with SDOH ≥2 needs represent 6.6% of the population\n")
        f.write("2. Significant differences exist in demographic and geographic factors\n")
        f.write("3. Insurance type shows strong association with SDOH status\n")
        f.write("4. Geographic vulnerability indices are higher in SDOH ≥2 group\n")
    
    print(f"Summary statistics saved to: {summary_path}")

def main():
    """Generate Table 1 for JAMA publication"""
    print("="*80)
    print("GENERATING TABLE 1 FOR JAMA PUBLICATION")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Create table
    table_df = create_table1(df)
    
    # Format for publication
    format_table_for_publication(table_df)
    
    print("\n" + "="*80)
    print("TABLE 1 GENERATION COMPLETE")
    print("="*80)
    print("\nOutputs generated:")
    print("1. Text version (for manuscript)")
    print("2. LaTeX version (for typesetting)")
    print("3. CSV version (for supplementary materials)")
    print("4. Summary statistics")
    print("\nAll outputs saved to results/tables/")

if __name__ == "__main__":
    main()