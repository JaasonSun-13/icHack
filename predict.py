# predict.py
"""
Generate predictions using trained models.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_master_data
from utils import setup_logging, load_model, load_json, ensure_dir

import logging

logger = logging.getLogger(__name__)


def predict(model_dir: str, input_file: str, output_file: str) -> None:
    """
    Generate predictions using trained models.
    
    Args:
        model_dir: Directory containing trained models
        input_file: Path to input data file
        output_file: Path to save predictions
    """
    model_path = Path(model_dir)
    
    # Load configuration
    logger.info("Loading model configuration...")
    config = load_json(str(model_path / 'config.json'))
    feature_cols = load_json(str(model_path / 'feature_columns.json'))['feature_columns']
    
    # Load data
    logger.info(f"Loading data from {input_file}...")
    df = load_master_data(input_file, date_column=config['data']['date_column'])
    
    # Check feature columns exist
    missing_features = set(feature_cols) - set(df.columns)
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    X = df[feature_cols]
    dates = df[config['data']['date_column']]
    
    # Load models
    logger.info("Loading models...")
    models = {}
    calibrators = {}
    
    for target_name in ['risk_7d', 'risk_30d', 'event_type_7d', 'impact_7d']:
        model_file = model_path / f'model_{target_name}.pkl'
        if model_file.exists():
            models[target_name] = load_model(str(model_file))
        else:
            logger.warning(f"Model file not found: {model_file}")
    
    # Load calibrators
    for target_name in ['risk_7d', 'risk_30d']:
        calibrator_file = model_path / f'calibrator_{target_name}.pkl'
        if calibrator_file.exists():
            calibrators[target_name] = load_model(str(calibrator_file))
    
    # Generate predictions
    logger.info("Generating predictions...")
    predictions = {'date': dates.values}
    
    for target_name, model in models.items():
        logger.info(f"  Predicting {target_name}...")
        
        if target_name in ['risk_7d', 'risk_30d']:
            # Binary classification
            y_pred_proba = model.predict_proba(X)
            
            # Apply calibration if available
            if target_name in calibrators:
                y_pred_proba = calibrators[target_name].transform(y_pred_proba)
            
            predictions[f'{target_name}_proba'] = y_pred_proba
            predictions[f'{target_name}_pred'] = (y_pred_proba >= 0.5).astype(int)
            
        elif target_name == 'event_type_7d':
            # Multi-class classification
            y_pred = model.predict(X)
            y_pred_proba = model.predict_proba(X)
            
            predictions[f'{target_name}_pred'] = y_pred
            # Store max probability
            predictions[f'{target_name}_max_proba'] = y_pred_proba.max(axis=1)
            
        elif target_name == 'impact_7d':
            # Regression
            y_pred = model.predict(X)
            predictions[f'{target_name}_pred'] = y_pred
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame(predictions)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    
    # Save predictions
    output_path = Path(output_file)
    ensure_dir(output_path.parent)
    
    if output_path.suffix == '.parquet':
        pred_df.to_parquet(output_path, index=False)
    else:
        pred_df.to_csv(output_path, index=False)
    
    logger.info(f"\nSaved {len(pred_df)} predictions to {output_path}")
    logger.info(f"Date range: {pred_df['date'].min()} to {pred_df['date'].max()}")
    
    # Print summary statistics
    logger.info("\nPrediction summary:")
    for col in pred_df.columns:
        if col != 'date' and '_proba' in col:
            logger.info(f"  {col}: mean={pred_df[col].mean():.4f}, std={pred_df[col].std():.4f}")
        elif col != 'date' and '_pred' in col and pred_df[col].dtype in [np.float64, np.float32]:
            logger.info(f"  {col}: mean={pred_df[col].mean():.4f}, std={pred_df[col].std():.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate predictions with trained models')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='artifacts/latest',
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/master.parquet',
        help='Path to input data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/predictions.parquet',
        help='Path to save predictions'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Generate predictions
    predict(args.model_dir, args.input, args.output)


if __name__ == '__main__':
    main()
