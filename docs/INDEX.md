# SDOH Prediction Model - Documentation Index

## Overview
This index provides a comprehensive guide to all documentation for the SDOH Prediction Model project.

## HTML Documentation

### Main Pages
- **[Home](html/index.html)** - Project overview and navigation
- **[Executive Summary](html/executive_summary.html)** - High-level summary of the model and results
- **[Executive Summary (Embedded)](html/executive_summary_embedded.html)** - Version with embedded visualizations

### Reports
Located in `html/reports/`:
- **[Executive Report - SDOH Screening](html/reports/executive_report_sdoh_screening.html)** - Comprehensive screening implementation report
- **[Integrated Executive Report](html/reports/integrated_executive_report_sdoh.html)** - Full integrated analysis
- **[Simplified Executive Report](html/reports/simplified_executive_report.html)** - Streamlined summary for stakeholders

### Clinical Guides
Located in `html/guides/`:
- **[Geriatric Clinical Guide](html/guides/geriatric_clinical_guide.html)** - Implementation guide for geriatric populations
- **[SHAP Explanation Guide](html/guides/shap_explanation_guide.html)** - Understanding model explanations

### Interactive Tools
Located in `html/interactive/`:
- **[SHAP Force Plots](html/interactive/shap_force_plots.html)** - Interactive model explanation visualizations

## Markdown Documentation

### Technical Documentation
- **[Technical Guide](technical_guide.md)** - Detailed technical implementation
- **[Clinical Implementation](clinical_implementation_recommendations.md)** - Clinical deployment recommendations
- **[Clinical Guide](clinical_guide.md)** - Clinical usage guidelines
- **[Threshold Implementation](sdoh_threshold_implementation_guide.md)** - Detailed threshold selection guide

### Policy Documentation
- **[Protected Attributes Policy](PROTECTED_ATTRIBUTES_POLICY.md)** - Bias mitigation and fairness policy

### Compliance Documentation
- **[TRIPOD-AI Compliance Report](../TRIPOD_AI_COMPLIANCE_REPORT.md)** - Model transparency report
- **[Release Notes](../RELEASE_NOTES.md)** - Version history and updates

## Data Documentation
Located in `../data/`:
- **[Data README](../data/README.md)** - Dataset structure and usage
- **[Corrections Log](../data/CORRECTIONS_LOG.md)** - Log of all data and figure corrections
- **[Dataset Metadata](../data/dataset_metadata.json)** - Technical dataset specifications

## Model Information

### Key Metrics (Test Set)
- **Test Set Size**: 78,745 patients
- **AUC**: 0.765
- **Sensitivity**: 72.2%
- **Specificity**: 67.9%
- **PPV**: 13.8%
- **NPV**: 97.1%

### Clinical Thresholds
- **General Screening**: 5.0%
- **Geriatric Screening**: 8.4%

## Project Structure

```
SDOH_Prediction_Model/
├── docs/                    # Documentation
│   ├── html/               # HTML documentation
│   │   ├── reports/        # Executive reports
│   │   ├── guides/         # Clinical guides
│   │   └── interactive/    # Interactive visualizations
│   └── *.md                # Markdown documentation
├── data/                   # Dataset storage (excluded from git)
├── models/                 # Trained models and metadata
├── results/                # Analysis results
│   ├── figures/            # Visualizations
│   │   ├── jama/          # Publication-ready figures
│   │   └── risk_histograms/ # Risk distribution plots
│   ├── reports/            # Text reports
│   └── tables/             # Data tables
├── scripts/                # Analysis scripts
│   ├── training/           # Model training scripts
│   ├── evaluation/         # Model evaluation scripts
│   ├── visualization/      # Figure generation scripts
│   ├── utilities/          # Utility scripts
│   └── archive/            # Archived/one-time scripts
└── src/                    # Source code modules
```

## Quick Links

### For Clinicians
1. [Geriatric Clinical Guide](html/guides/geriatric_clinical_guide.html)
2. [Simplified Executive Report](html/reports/simplified_executive_report.html)
3. [Clinical Implementation Recommendations](clinical_implementation_recommendations.md)

### For Technical Users
1. [Technical Guide](technical_guide.md)
2. [SHAP Explanation Guide](html/guides/shap_explanation_guide.html)
3. [Data Documentation](../data/README.md)

### For Administrators
1. [Executive Summary](html/executive_summary.html)
2. [Protected Attributes Policy](PROTECTED_ATTRIBUTES_POLICY.md)
3. [TRIPOD-AI Compliance](../TRIPOD_AI_COMPLIANCE_REPORT.md)

## Contact
For questions about this documentation, please refer to the project README or contact the development team.