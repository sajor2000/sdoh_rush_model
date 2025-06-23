"""
Configuration file for SDOH Prediction Model
Update DATA_PATH to point to your local data file
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.absolute()

# Data configuration
# NOTE: Update this path to point to your local data file
# The data file should not be committed to Git for privacy reasons
DATA_PATH = os.environ.get('SDOH_DATA_PATH', 
                          '/path/to/your/sdoh2_ml_final_all_svi.csv')

# Model directories
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
SCRIPTS_DIR = BASE_DIR / 'scripts'
SRC_DIR = BASE_DIR / 'src'

# Output directories
FIGURES_DIR = RESULTS_DIR / 'figures'
REPORTS_DIR = RESULTS_DIR / 'reports'
TABLES_DIR = RESULTS_DIR / 'tables'

# Ensure directories exist
for dir_path in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR, REPORTS_DIR, TABLES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model configuration
RANDOM_SEED = 2025
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # Of the training set

# Default model parameters
DEFAULT_XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1,
    'scale_pos_weight': 1,
    'random_state': RANDOM_SEED,
    'n_jobs': -1,
    'eval_metric': 'logloss',
    'early_stopping_rounds': 50
}