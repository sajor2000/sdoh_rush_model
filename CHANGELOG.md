# Changelog

All notable changes to the SDOH Risk Screening Model project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-06-23 - FINAL RELEASE

### Added
- **Comprehensive SHAP Analysis**: Complete model interpretability with variable importance, waterfall plots, and dependence analysis
- **TRIPOD-AI Compliant Figures**: Dual threshold analysis for general hospital vs geriatric clinic settings
- **Universal Screening Vision**: Reframed documentation to position AI tool as bridge to universal SDOH screening
- **Fairness Verification**: Rigorous bias analysis with race/ethnicity exclusion by design
- **Dual Threshold Strategy**: Optimized cutoffs for general (5.7%) and geriatric (8.4%) populations
- **Container Support**: Docker container for reproducible analysis
- **Build Automation**: Makefile for one-command setup and execution
- **Environment Management**: Conda environment.yml for consistent dependencies

### Changed
- **Improved Calibration**: ECE reduced from 0.028 to 0.0022 (96% improvement)
- **Enhanced Documentation**: Scientifically accurate and readable HTML guides
- **Strategic Messaging**: From efficiency tool to bridge strategy for universal screening
- **Model Performance**: AUC 0.766 with excellent calibration and fairness
- **Project Structure**: Organized scripts, results, and documentation for clarity

### Fixed
- **Metric Consistency**: Aligned all reported values (PPV: 13.7%, Sensitivity: 72.2%)
- **Calibration Error**: Updated to final model ECE value (0.0022)
- **Figure References**: Corrected AUC values in all visualizations
- **Scientific Accuracy**: Verified all statistical interpretations

### Technical Details
- **Model**: XGBoost with isotonic calibration
- **Training Data**: 393,725 patients with 60/20/20 split
- **Performance**: AUC 0.766, ECE 0.0022, excellent fairness
- **Features**: 46 variables (demographics + address-based SDOH)
- **Bias Mitigation**: Race/ethnicity excluded by design

### Documentation
- **HTML Guides**: Comprehensive clinical and executive documentation
- **SHAP Explanations**: Plain language model interpretability guides
- **Implementation Guides**: Step-by-step clinical workflow integration
- **Technical Documentation**: Complete reproducibility instructions

## [1.0.0] - 2025-06-22 - Initial Release

### Added
- Initial SDOH risk prediction model
- Basic training and evaluation scripts
- Core documentation
- Data processing pipeline
- Model validation framework

---

**Full Documentation**: [GitHub Repository](https://github.com/sajor2000/sdoh_rush_model)  
**Author**: Juan C. Rojas, MD, MS  
**License**: MIT