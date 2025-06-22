#!/usr/bin/env python3
"""
SDOH Risk Prediction Script
==========================

Command-line interface for making predictions with the SDOH model.
"""

import click
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model_evaluation import SDOHPredictor


@click.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Path to input CSV file with patient data')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Path to output CSV file for predictions')
@click.option('--model', '-m', type=click.Path(exists=True),
              default='models/xgboost_best.json',
              help='Path to model file')
@click.option('--artifact', '-a', type=click.Path(exists=True),
              default='models/model_artifact.joblib',
              help='Path to model artifact file')
@click.option('--threshold', '-t', type=click.Choice(['standard', 'high_ppv', 'high_sensitivity']),
              default='standard',
              help='Threshold preset to use')
@click.option('--include-probabilities', is_flag=True,
              help='Include risk probabilities in output')
@click.option('--batch-size', '-b', type=int, default=10000,
              help='Batch size for processing large files')
def predict(input, output, model, artifact, threshold, include_probabilities, batch_size):
    """
    Make SDOH risk predictions for a dataset.
    
    Example:
        python predict.py -i patients.csv -o predictions.csv
    """
    click.echo(f"🏥 SDOH Risk Prediction Tool")
    click.echo(f"=" * 40)
    
    # Load model
    click.echo(f"📊 Loading model from {model}...")
    try:
        predictor = SDOHPredictor(model, artifact)
    except Exception as e:
        click.echo(f"❌ Error loading model: {e}", err=True)
        return 1
    
    # Load data
    click.echo(f"📁 Loading data from {input}...")
    try:
        data = pd.read_csv(input)
        click.echo(f"✅ Loaded {len(data):,} records")
    except Exception as e:
        click.echo(f"❌ Error loading data: {e}", err=True)
        return 1
    
    # Check for required features
    if predictor.feature_names:
        missing_features = set(predictor.feature_names) - set(data.columns)
        if missing_features:
            click.echo(f"❌ Missing required features: {missing_features}", err=True)
            return 1
    
    # Process in batches
    all_predictions = []
    all_probabilities = []
    
    n_batches = (len(data) + batch_size - 1) // batch_size
    
    with click.progressbar(range(n_batches), label='Processing batches') as bar:
        for i in bar:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(data))
            batch = data.iloc[start_idx:end_idx]
            
            # Get features
            if predictor.feature_names:
                X_batch = batch[predictor.feature_names]
            else:
                # Assume all columns except common ID columns are features
                id_cols = ['patient_id', 'id', 'ID', 'PatientID']
                feature_cols = [col for col in batch.columns if col not in id_cols]
                X_batch = batch[feature_cols]
            
            # Make predictions
            predictions = predictor.predict(X_batch, threshold=threshold)
            all_predictions.extend(predictions)
            
            if include_probabilities:
                probabilities = predictor.predict_proba(X_batch)
                all_probabilities.extend(probabilities)
    
    # Create output dataframe
    output_data = data.copy()
    output_data['needs_screening'] = all_predictions
    
    if include_probabilities:
        output_data['risk_score'] = all_probabilities
    
    # Add threshold info
    threshold_value = predictor.thresholds[threshold]
    output_data['threshold_used'] = threshold_value
    output_data['threshold_type'] = threshold
    
    # Save results
    click.echo(f"💾 Saving predictions to {output}...")
    output_data.to_csv(output, index=False)
    
    # Summary statistics
    n_screened = sum(all_predictions)
    screening_rate = n_screened / len(data) * 100
    
    click.echo(f"\n📊 Prediction Summary")
    click.echo(f"-" * 40)
    click.echo(f"Total patients: {len(data):,}")
    click.echo(f"Flagged for screening: {n_screened:,} ({screening_rate:.1f}%)")
    click.echo(f"Threshold used: {threshold_value:.4f} ({threshold})")
    
    if include_probabilities:
        risk_scores = np.array(all_probabilities)
        click.echo(f"\nRisk Score Distribution:")
        click.echo(f"  Min: {risk_scores.min():.4f}")
        click.echo(f"  25%: {np.percentile(risk_scores, 25):.4f}")
        click.echo(f"  50%: {np.percentile(risk_scores, 50):.4f}")
        click.echo(f"  75%: {np.percentile(risk_scores, 75):.4f}")
        click.echo(f"  Max: {risk_scores.max():.4f}")
    
    click.echo(f"\n✅ Predictions saved successfully!")


if __name__ == '__main__':
    predict()