# train.py
"""
Main training script for multi-task event risk forecasting.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data import load_master_data, get_feature_columns, validate_data, split_features_targets
from splits import TimeSeriesSplitter
from models import RiskBinaryModel, EventTypeModel, ImpactRegressionModel
from calibration import ProbabilityCalibrator, compute_calibration_curve
from metrics import (
    evaluate_binary_classification,
    evaluate_multiclass_classification,
    evaluate_regression,
    MetricsTracker
)
from interpret import (
    plot_feature_importance,
    compute_shap_values,
    plot_shap_summary,
    get_top_shap_features
)
from utils import (
    setup_logging,
    load_config,
    save_json,
    save_model,
    ensure_dir
)

import logging

logger = logging.getLogger(__name__)


def train_all_models(config: dict) -> None:
    """
    Train all models using walk-forward validation.
    
    Args:
        config: Configuration dictionary
    """
    # Setup directories
    artifacts_dir = ensure_dir(config['output']['artifacts_dir'])
    models_dir = ensure_dir(artifacts_dir / config['output']['models_subdir'])
    reports_dir = ensure_dir(config['output']['reports_dir'])
    
    # Load data
    logger.info("Loading data...")
    df = load_master_data(
        config['data']['master_file'],
        config['data'].get('fallback_csv'),
        config['data']['date_column']
    )
    
    # Get target and feature columns
    targets = config['data']['targets']
    target_cols = list(targets.values())
    feature_cols = get_feature_columns(df, target_cols, config['data']['date_column'])
    
    # Validate data
    validate_data(df, target_cols + [config['data']['date_column']])
    
    # Split features and targets
    X, y, dates = split_features_targets(df, feature_cols, target_cols, config['data']['date_column'])
    
    logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
    
    # Initialize splitter
    splitter = TimeSeriesSplitter(config['splits']['folds'])
    
    # Initialize models
    models = {
        'risk_7d': RiskBinaryModel(config['models']['risk_binary'], 'risk_7d'),
        'risk_30d': RiskBinaryModel(config['models']['risk_binary'], 'risk_30d'),
        'event_type_7d': EventTypeModel(config['models']['event_type']),
        'impact_7d': ImpactRegressionModel(config['models']['impact_regression'])
    }
    
    # Initialize calibrators for risk models
    calibrators = {
        'risk_7d': ProbabilityCalibrator(config['calibration']['method']),
        'risk_30d': ProbabilityCalibrator(config['calibration']['method'])
    }
    
    # Track metrics across folds
    metrics_trackers = {
        'risk_7d': MetricsTracker(),
        'risk_30d': MetricsTracker(),
        'event_type_7d': MetricsTracker(),
        'impact_7d': MetricsTracker()
    }
    
    # Store all predictions and feature importances
    all_predictions = []
    all_importances = {}
    shap_results = {}
    
    # Walk-forward training
    for fold_idx in range(splitter.n_splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"Training fold {fold_idx + 1}/{splitter.n_splits}")
        logger.info(f"{'='*60}")
        
        # Get fold indices
        train_idx, valid_idx = splitter.get_fold_indices(dates, fold_idx)
        
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        dates_valid = dates.iloc[valid_idx]
        
        fold_name = f"fold_{fold_idx + 1}"
        fold_predictions = {'date': dates_valid.values}
        
        # Train each model
        for target_name, model in models.items():
            logger.info(f"\nTraining {target_name}...")
            
            y_train_target = y_train[targets[target_name]]
            y_valid_target = y_valid[targets[target_name]]
            
            # Train model
            train_info = model.train(X_train, y_train_target, X_valid, y_valid_target)
            logger.info(f"Training info: {train_info}")
            
            # Get predictions
            if target_name in ['risk_7d', 'risk_30d']:
                # Binary classification - get probabilities
                y_pred_proba = model.predict_proba(X_valid)
                
                # Calibrate probabilities
                calibrators[target_name].fit(y_valid_target.values, y_pred_proba)
                y_pred_proba_cal = calibrators[target_name].transform(y_pred_proba)
                
                # Evaluate
                metrics_uncal = evaluate_binary_classification(y_valid_target.values, y_pred_proba)
                metrics_cal = evaluate_binary_classification(y_valid_target.values, y_pred_proba_cal)
                
                # Compute calibration curve
                bin_centers, empirical_prob, counts = compute_calibration_curve(
                    y_valid_target.values, y_pred_proba_cal
                )
                
                metrics = {
                    **{f'uncal_{k}': v for k, v in metrics_uncal.items()},
                    **{f'cal_{k}': v for k, v in metrics_cal.items()},
                    'calibration_bins': bin_centers.tolist(),
                    'calibration_empirical': empirical_prob.tolist()
                }
                
                logger.info(f"  Uncalibrated - AUROC: {metrics_uncal['auroc']:.4f}, Brier: {metrics_uncal['brier']:.4f}")
                logger.info(f"  Calibrated   - AUROC: {metrics_cal['auroc']:.4f}, Brier: {metrics_cal['brier']:.4f}")
                
                # Store predictions
                fold_predictions[f'{target_name}_proba'] = y_pred_proba_cal
                fold_predictions[f'{target_name}_true'] = y_valid_target.values
                
            elif target_name == 'event_type_7d':
                # Multi-class classification
                y_pred_proba = model.predict_proba(X_valid)
                y_pred = model.predict(X_valid)
                
                # Evaluate
                metrics = evaluate_multiclass_classification(y_valid_target.values, y_pred)
                
                logger.info(f"  Macro-F1: {metrics['macro_f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
                
                # Store predictions
                fold_predictions[f'{target_name}_pred'] = y_pred
                fold_predictions[f'{target_name}_true'] = y_valid_target.values
                
            elif target_name == 'impact_7d':
                # Regression
                y_pred = model.predict(X_valid)
                
                # Evaluate
                metrics = evaluate_regression(y_valid_target.values, y_pred)
                
                logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, Spearman: {metrics['spearman_corr']:.4f}")
                
                # Store predictions
                fold_predictions[f'{target_name}_pred'] = y_pred
                fold_predictions[f'{target_name}_true'] = y_valid_target.values
            
            # Track metrics
            metrics_trackers[target_name].add_fold(fold_name, metrics)
            
            # Get feature importance
            importance_df = model.get_feature_importance(feature_cols)
            if fold_idx not in all_importances:
                all_importances[fold_idx] = {}
            all_importances[fold_idx][target_name] = importance_df
            
            # Plot feature importance for last fold
            if fold_idx == splitter.n_splits - 1:
                plot_feature_importance(
                    importance_df,
                    f"{target_name} - Feature Importance",
                    top_n=config['interpretation']['top_n_features'],
                    save_path=str(reports_dir / f'feature_importance_{target_name}.png')
                )
        
        # Compute SHAP values for risk_7d on last fold
        if fold_idx == splitter.n_splits - 1 and config['interpretation'].get('compute_shap', False):
            logger.info("\nComputing SHAP values for risk_7d...")
            shap_values = compute_shap_values(
                models['risk_7d'],
                X_valid,
                sample_size=config['interpretation'].get('shap_sample_size', 1000)
            )
            
            if shap_values is not None:
                shap_results['risk_7d'] = {
                    'shap_values': shap_values,
                    'feature_names': feature_cols
                }
                
                # Plot SHAP summary
                plot_shap_summary(
                    shap_values,
                    "risk_7d - SHAP Summary",
                    save_path=str(reports_dir / 'shap_summary_risk_7d.png'),
                    max_display=20
                )
                
                # Get top SHAP features
                top_shap = get_top_shap_features(shap_values, feature_cols, top_n=20)
                top_shap.to_csv(reports_dir / 'top_shap_features_risk_7d.csv', index=False)
        
        # Store fold predictions
        all_predictions.append(pd.DataFrame(fold_predictions))
    
    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    # Save predictions
    predictions_path = Path(config['output']['predictions_dir']) / 'predictions.parquet'
    ensure_dir(predictions_path.parent)
    predictions_df.to_parquet(predictions_path, index=False)
    logger.info(f"\nSaved predictions to {predictions_path}")
    
    # Save metrics summary
    all_metrics = {}
    for target_name, tracker in metrics_trackers.items():
        logger.info(f"\n{target_name} - Metrics across folds:")
        summary_df = tracker.get_summary()
        logger.info(f"\n{summary_df.to_string()}")
        
        # Save to CSV
        summary_df.to_csv(reports_dir / f'metrics_{target_name}.csv', index=False)
        
        # Get aggregated metrics
        agg_metrics = tracker.get_aggregated_metrics()
        all_metrics[target_name] = agg_metrics
    
    # Save all metrics to JSON
    save_json(all_metrics, str(reports_dir / 'metrics.json'))
    
    # Save final models (trained on last fold)
    logger.info("\nSaving final models...")
    for target_name, model in models.items():
        save_model(model, str(models_dir / f'model_{target_name}.pkl'))
    
    # Save calibrators
    for target_name, calibrator in calibrators.items():
        save_model(calibrator, str(models_dir / f'calibrator_{target_name}.pkl'))
    
    # Save feature columns
    save_json({'feature_columns': feature_cols}, str(models_dir / 'feature_columns.json'))
    
    # Save config
    save_json(config, str(models_dir / 'config.json'))
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Predictions saved to: {predictions_path}")
    logger.info(f"Reports saved to: {reports_dir}")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train event risk forecasting models')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
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
    
    # Load config
    config = load_config(args.config)
    
    # Train models
    train_all_models(config)


if __name__ == '__main__':
    main()
