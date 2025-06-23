#!/usr/bin/env python3
"""
SDOH Risk Screening Model - Production Code
Uses XGBoost with calibration for predicting social determinants of health needs
Model Version: 2.0 (Scientifically Validated)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class SDOHRiskScreener:
    """Production-ready SDOH risk screening model"""
    
    def __init__(self, model_path='models/xgboost_scientific_calibrated.joblib',
                 metadata_path='models/scientific_model_metadata.json'):
        """Initialize the SDOH risk screener
        
        Args:
            model_path: Path to the calibrated model file
            metadata_path: Path to the model metadata JSON
        """
        self.model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.threshold = self.metadata['best_threshold']
        self.feature_names = self.metadata['feature_names']
        
        print(f"Loaded SDOH Risk Screening Model v{self.metadata.get('version', '2.0')}")
        print(f"Threshold: {self.threshold:.3f}")
        print(f"Training AUC: {self.metadata['test_metrics']['auc']:.3f}")
    
    def prepare_features(self, patient_data):
        """Prepare patient data for prediction
        
        Args:
            patient_data: DataFrame with patient demographics and SVI/ADI data
            
        Returns:
            DataFrame ready for prediction
        """
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(patient_data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        X = patient_data[self.feature_names]
        
        # Handle any missing values (should be minimal with SVI/ADI data)
        X = X.fillna(X.median())
        
        return X
    
    def predict_risk(self, patient_data, use_geriatric_threshold=False):
        """Predict SDOH risk for patients
        
        Args:
            patient_data: DataFrame with patient information
            use_geriatric_threshold: Use higher threshold for senior populations
            
        Returns:
            DataFrame with risk scores and recommendations
        """
        # Prepare features
        X = self.prepare_features(patient_data)
        
        # Get risk probabilities
        risk_scores = self.model.predict_proba(X)[:, 1]
        
        # Determine threshold
        threshold = 0.084 if use_geriatric_threshold else self.threshold
        
        # Create results
        results = pd.DataFrame({
            'patient_id': patient_data.index,
            'risk_score': risk_scores,
            'risk_percentage': risk_scores * 100,
            'needs_screening': risk_scores >= threshold,
            'risk_category': pd.cut(risk_scores, 
                                   bins=[0, 0.03, 0.05, 0.10, 1.0],
                                   labels=['Low', 'Moderate', 'High', 'Very High'])
        })
        
        return results
    
    def get_feature_importance(self, top_n=20):
        """Get top feature importances with professional names
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        # Get base model from calibrated model
        base_model = self.model.estimator
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': base_model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Add professional names
        importance_df['description'] = importance_df['feature'].map(
            self._get_feature_descriptions()
        )
        
        return importance_df
    
    def _get_feature_descriptions(self):
        """Map technical feature names to descriptions"""
        return {
            'rpl_themes': 'Overall Social Vulnerability Percentile',
            'adi_natrank': 'Area Deprivation Index National Rank',
            'rpl_theme1': 'Socioeconomic Status Percentile',
            'age_at_survey': 'Patient Age',
            'ep_pov150': 'Poverty Rate (% Below 150% Poverty Line)',
            'ep_unemp': 'Unemployment Rate (%)',
            'ep_hburd': 'Housing Cost Burden (%)',
            'ep_nohsdp': 'No High School Diploma (%)',
            'ep_uninsur': 'Uninsured Rate (%)',
            'sex_female': 'Sex (Female=1)',
            # Add more mappings as needed
        }
    
    def explain_prediction(self, patient_data, patient_id):
        """Provide explanation for a single patient's risk score
        
        Args:
            patient_data: DataFrame with patient information
            patient_id: ID of patient to explain
            
        Returns:
            Dict with risk factors and explanation
        """
        # Get patient's data
        patient = patient_data.loc[patient_id]
        X = self.prepare_features(patient_data.loc[[patient_id]])
        
        # Get risk score
        risk_score = self.model.predict_proba(X)[0, 1]
        
        # Get top contributing features (simplified - would use SHAP in production)
        feature_values = X.iloc[0]
        feature_importance = self.get_feature_importance()
        
        # Identify high-risk factors
        risk_factors = []
        for _, row in feature_importance.head(10).iterrows():
            feature = row['feature']
            value = feature_values[feature]
            
            # Simple logic to identify concerning values
            if 'rpl' in feature and value > 0.8:
                risk_factors.append(f"{row['description']}: High ({value:.1%})")
            elif 'ep_' in feature and value > feature_values[feature].median():
                risk_factors.append(f"{row['description']}: Above average ({value:.1f})")
        
        explanation = {
            'patient_id': patient_id,
            'risk_score': f"{risk_score:.1%}",
            'risk_level': 'High' if risk_score >= self.threshold else 'Low',
            'recommendation': 'Recommend SDOH screening' if risk_score >= self.threshold else 'No screening needed',
            'top_risk_factors': risk_factors[:5],
            'age': patient['age_at_survey'],
            'area_vulnerability': patient.get('rpl_themes', 'Unknown')
        }
        
        return explanation

def main():
    """Example usage of the SDOH Risk Screener"""
    
    # Initialize screener
    screener = SDOHRiskScreener()
    
    # Load sample patient data (in production, this would come from EHR)
    print("\nLoading patient data...")
    patient_data = pd.read_csv('data/sample_patients.csv')
    print(f"Loaded {len(patient_data)} patients")
    
    # Run risk screening
    print("\nRunning SDOH risk screening...")
    results = screener.predict_risk(patient_data)
    
    # Summary statistics
    print("\nScreening Results Summary:")
    print(f"Patients flagged for screening: {results['needs_screening'].sum()} ({results['needs_screening'].mean():.1%})")
    print(f"Average risk score: {results['risk_score'].mean():.1%}")
    print("\nRisk Distribution:")
    print(results['risk_category'].value_counts().sort_index())
    
    # Example: Explain high-risk patient
    high_risk_patients = results[results['needs_screening']].head()
    if len(high_risk_patients) > 0:
        patient_id = high_risk_patients.iloc[0]['patient_id']
        explanation = screener.explain_prediction(patient_data, patient_id)
        
        print(f"\nExample High-Risk Patient Explanation:")
        print(f"Patient ID: {explanation['patient_id']}")
        print(f"Risk Score: {explanation['risk_score']}")
        print(f"Recommendation: {explanation['recommendation']}")
        print("Top Risk Factors:")
        for factor in explanation['top_risk_factors']:
            print(f"  - {factor}")
    
    # Save results
    output_path = 'results/sdoh_screening_results.csv'
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
