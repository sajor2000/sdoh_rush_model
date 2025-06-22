# Technical Documentation

## Model Architecture

### Algorithm: XGBoost (Gradient Boosting)
- **Framework**: XGBoost 1.7.6
- **Model Type**: Binary classification
- **Objective**: binary:logistic
- **Tree Structure**: Shallow trees (max_depth=3) for interpretability

### Hyperparameters (Optimized Configuration)
```yaml
max_depth: 3
learning_rate: 0.05
n_estimators: 400
subsample: 0.7
colsample_bytree: 0.6
scale_pos_weight: 14
min_child_weight: 20
gamma: 0.3
reg_alpha: 2
reg_lambda: 2
```

### Input Features (46 total)
The model uses demographic, clinical, and geographic features including:

1. **Demographics**: Age, sex
2. **Insurance**: Financial class categories
3. **Geographic**: Social Vulnerability Index (SVI) themes, Area Deprivation Index (ADI)
4. **Clinical**: Utilization patterns, visit history

## Data Pipeline

### 1. Data Preprocessing
```python
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Missing value handling
# Categorical: mode imputation
# Numerical: median imputation
```

### 2. Feature Engineering
- SVI theme calculations from geographic data
- ADI percentile ranking
- Insurance category encoding
- Age binning for certain analyses

### 3. Model Training
```python
import xgboost as xgb

# Create training matrix
dtrain = xgb.DMatrix(X_train, label=y_train)

# Train with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50
)
```

## Performance Metrics

### Classification Performance
- **AUC**: 0.762 (95% CI: 0.757-0.767)
- **AUPRC**: 0.210 (95% CI: 0.195-0.225)
- **Brier Score**: 0.195
- **Expected Calibration Error**: 0.034

### Clinical Metrics (at threshold 0.5644)
- **Sensitivity**: 61.1%
- **Specificity**: 77.6%
- **PPV**: 16.2%
- **NPV**: 96.6%
- **F1-Score**: 0.256

## Model Deployment

### 1. Model Serialization
The model is saved in XGBoost's native JSON format for maximum compatibility:

```python
# Save model
model.save_model('xgboost_best.json')

# Save complete artifact
artifact = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'metadata': {...}
}
joblib.dump(artifact, 'model_artifact.joblib')
```

### 2. Loading and Inference
```python
from src.model_evaluation import SDOHPredictor

# Initialize predictor
predictor = SDOHPredictor(
    model_path='models/xgboost_best.json',
    artifact_path='models/model_artifact.joblib'
)

# Make predictions
risk_scores = predictor.predict_proba(patient_data)
predictions = predictor.predict(patient_data, threshold='standard')
```

### 3. API Integration
Example REST API endpoint:

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
predictor = SDOHPredictor('models/xgboost_best.json', 'models/model_artifact.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data['patients'])
    
    scores = predictor.predict_proba(df)
    predictions = predictor.predict(df, threshold=data.get('threshold', 'standard'))
    
    return jsonify({
        'risk_scores': scores.tolist(),
        'predictions': predictions.tolist()
    })
```

## Model Monitoring

### Performance Monitoring
Track these metrics over time:
- AUC and AUPRC on incoming data
- Calibration metrics
- Feature drift detection
- Prediction distribution changes

### Fairness Monitoring
```python
from src.fairness_analysis import FairnessAnalyzer

analyzer = FairnessAnalyzer()

# Analyze demographic parity
demo_results = analyzer.analyze_demographic_parity(
    y_true, y_pred, sensitive_features
)

# Analyze performance parity
perf_results = analyzer.analyze_performance_parity(
    y_true, y_pred, y_proba, sensitive_features
)
```

### Data Quality Checks
- Missing value rates by feature
- Feature distribution changes
- Outlier detection
- Schema validation

## Retraining Pipeline

### When to Retrain
- Performance degradation (AUC drops > 0.02)
- Significant fairness metric changes
- Scheduled annual retraining
- Major population or process changes

### Retraining Process
1. **Data Collection**: Gather new labeled data
2. **Data Validation**: Check quality and consistency
3. **Model Training**: Use hyperparameter optimization
4. **Evaluation**: Compare against current model
5. **A/B Testing**: Gradual rollout with monitoring
6. **Deployment**: Replace production model

## Security Considerations

### Data Protection
- All patient data encrypted at rest and in transit
- Access controls and audit logging
- HIPAA compliance requirements
- De-identification procedures

### Model Security
- Model files stored in secure environments
- Version control and change management
- Regular security assessments
- Secure deployment pipelines

## Troubleshooting

### Common Issues

1. **Feature Mismatch**
   ```python
   # Check feature alignment
   missing_features = set(expected_features) - set(data.columns)
   if missing_features:
       raise ValueError(f"Missing features: {missing_features}")
   ```

2. **Scaling Issues**
   ```python
   # Ensure proper scaling
   if scaler is None:
       raise ValueError("Scaler not loaded. Check model artifact.")
   ```

3. **Memory Issues**
   ```python
   # Process in batches for large datasets
   batch_size = 10000
   for i in range(0, len(data), batch_size):
       batch = data.iloc[i:i+batch_size]
       predictions.extend(predictor.predict(batch))
   ```

### Performance Optimization

1. **Batch Processing**: Process multiple patients at once
2. **Feature Caching**: Cache computed features when possible
3. **Model Caching**: Keep model in memory for repeated predictions
4. **Parallel Processing**: Use multiple cores for large datasets

## Dependencies

### Core Requirements
```
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
xgboost>=1.7.6
joblib>=1.3.1
```

### Optional Dependencies
```
shap>=0.42.1  # For model interpretation
fairlearn>=0.8.0  # For fairness analysis
matplotlib>=3.7.2  # For visualization
```

## Development Setup

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Testing**
   ```bash
   pytest tests/
   ```

3. **Code Quality**
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```