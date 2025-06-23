# SDOH Prediction Model - Project Structure

## Directory Organization

```
SDOH_Prediction_Model/
│
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── fairness_analysis.py  # Fairness evaluation utilities
│   └── model_evaluation.py   # Model evaluation utilities
│
├── scripts/                  # Executable scripts
│   ├── Training scripts:
│   │   ├── train_scientifically_correct.py
│   │   ├── train_final_balanced.py
│   │   ├── train_balanced_calibrated_model.py
│   │   └── train_balanced_fast.py
│   │
│   ├── Evaluation scripts:
│   │   ├── comprehensive_fairness_analysis.py
│   │   ├── senior_clinic_threshold_analysis.py
│   │   └── generate_shap_plots.py
│   │
│   ├── Visualization scripts:
│   │   ├── create_executive_report.py
│   │   ├── create_professional_shap_plots.py
│   │   └── generate_tripod_figures_jama.py
│   │
│   └── Model refinement scripts:
│       ├── apply_platt_scaling.py
│       ├── fix_model_calibration.py
│       └── refit_model_*.py
│
├── models/                   # Trained models and metadata
│   ├── *.joblib             # Serialized model files
│   ├── *.json               # Model configurations
│   └── *_metadata.json      # Model metadata files
│
├── data/                    # Data directory (empty - see README)
│   └── README.md            # Instructions for data setup
│
├── notebooks/               # Jupyter notebooks
│   └── calibration_improvement_demo.ipynb
│
├── results/                 # Output directory
│   ├── figures/            # Generated plots and visualizations
│   │   ├── jama/          # JAMA-style figures
│   │   └── *.png/pdf      # Other figures
│   │
│   ├── reports/           # Analysis reports
│   │   └── *.txt/csv     # Text and CSV reports
│   │
│   └── tables/            # Generated tables
│       └── *.csv/tex     # CSV and LaTeX tables
│
├── docs/                   # Documentation
│   ├── clinical_guide.md   # Clinical usage guide
│   ├── technical_guide.md  # Technical documentation
│   ├── guides/            # How-to guides
│   └── summaries/         # Model summaries
│
├── tests/                 # Test files (to be added)
│
├── config.py              # Configuration file
├── requirements.txt       # Python dependencies
├── LICENSE               # MIT License
├── README.md             # Main documentation
├── .gitignore           # Git ignore rules
└── sdoh_risk_screener.py # Main application entry point
```

## Key Files

### Core Application
- `sdoh_risk_screener.py`: Main production-ready risk screening class
- `config.py`: Central configuration (data paths, parameters)

### Models
- `xgboost_scientific_calibrated.joblib`: Primary calibrated model
- `scientific_model_metadata.json`: Model metadata and performance metrics

### Scripts
All scripts now use relative imports and the central config file.

### Data
**Important**: Data files are not included in the repository for privacy reasons.
See `data/README.md` for setup instructions.

## Usage

1. Set up your data path in `config.py` or via environment variable
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main screener: `python sdoh_risk_screener.py`

## Git Workflow

The `.gitignore` file is configured to exclude:
- Virtual environments (`sdoh_env/`)
- Data files (`*.csv`)
- Python cache files
- OS-specific files (`.DS_Store`)
- Large model files (optional)

## Notes for Contributors

- Always use the `config.py` file for paths and parameters
- Do not commit data files or credentials
- Follow the existing code structure when adding new scripts
- Update this document when adding new directories or major files