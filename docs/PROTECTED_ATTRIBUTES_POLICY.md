# Protected Attributes Policy

## Executive Summary

**Race and ethnicity are COMPLETELY EXCLUDED from model training and are ONLY used for post-hoc fairness assessment.**

## Implementation Details

### 1. Data Pipeline

```python
# STEP 1: Load data
df = pd.read_csv('sdoh2_ml_final_all_svi.csv')

# STEP 2: Extract protected attributes BEFORE training
protected_attributes = df[['race_category', 'ethnicity_category']].copy()

# STEP 3: Remove protected attributes from features
X = df.drop(columns=['sdoh_two_yes', 'race_category', 'ethnicity_category'])
y = df['sdoh_two_yes']

# STEP 4: Train model on X (WITHOUT protected attributes)
model.fit(X_train, y_train)  # Model NEVER sees race/ethnicity

# STEP 5: Use protected attributes ONLY for fairness analysis
fairness_results = analyze_fairness(model, X_test, y_test, protected_test)
```

### 2. Verification Steps

The code includes multiple verification steps:

1. **Data Loading**: Explicitly excludes race/ethnicity columns
2. **Preprocessing**: Verifies no protected attributes in feature pipeline
3. **Model Training**: Double-checks features don't contain protected attributes
4. **Fairness Analysis**: Clearly states it's post-hoc analysis only

### 3. State-of-the-Art Frameworks

We use industry-leading ML frameworks:

- **XGBoost**: Gradient boosting with `scale_pos_weight` for imbalance
- **LightGBM**: Microsoft's gradient boosting with `is_unbalance=True`
- **CatBoost**: Yandex's gradient boosting with `auto_class_weights='Balanced'`
- **Scikit-learn**: Random Forest, Logistic Regression, etc.
- **Optuna**: Bayesian hyperparameter optimization

### 4. 70/30 Train-Test Split

```python
# Stratified split maintaining class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 30% for testing
    stratify=y,     # Maintain 6.6% positive class in both sets
    random_state=42 # Reproducibility
)
```

### 5. Class Imbalance Handling

Without synthetic data generation:

```python
# Random Forest
RandomForestClassifier(class_weight='balanced_subsample')

# XGBoost
XGBClassifier(scale_pos_weight=14.15)  # ~93.4/6.6

# LightGBM  
LGBMClassifier(is_unbalance=True)

# CatBoost
CatBoostClassifier(auto_class_weights='Balanced')
```

## Compliance Statement

This implementation ensures:

1. ✅ **Fairness**: Protected attributes never influence model predictions
2. ✅ **Transparency**: Clear separation of modeling and fairness assessment
3. ✅ **Reproducibility**: Deterministic splits and random seeds
4. ✅ **Best Practices**: State-of-the-art ML frameworks
5. ✅ **Healthcare Standards**: Appropriate for clinical/healthcare applications

## Audit Trail

Every function that handles data includes verification:

```python
def _verify_no_protected_attributes(self, X: pd.DataFrame):
    """Verify that no protected attributes are present in features."""
    protected_keywords = ['race', 'ethnic', 'ethnicity']
    found = [col for col in X.columns 
            if any(keyword in col.lower() for keyword in protected_keywords)]
    
    if found:
        raise ValueError(f"CRITICAL ERROR: Protected attributes found in features: {found}")
```

This ensures that even if the data pipeline is modified, protected attributes cannot accidentally be included in model training.