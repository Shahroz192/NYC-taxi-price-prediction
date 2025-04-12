"""
Module for automated model testing before promotion.
"""
import pandas as pd
import numpy as np
import logging
import mlflow
from mlflow.tracking import MlflowClient
from typing import Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
TEST_DATA_PATH = "data/processed/test.csv"
TEST_METRIC_THRESHOLDS = {
    'rmse': 10.0,  # Maximum allowed RMSE
    'mae': 8.0,    # Maximum allowed MAE
    'r2': 0.6      # Minimum allowed R2
}

def load_test_data() -> pd.DataFrame:
    """Load and return the test dataset."""
    try:
        test_df = pd.read_csv(TEST_DATA_PATH)
        logger.info(f"Loaded test data from {TEST_DATA_PATH}")
        return test_df
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def run_model_tests(model_uri: str, test_data: pd.DataFrame) -> Tuple[dict, bool]:
    """
    Run automated tests on the model before promotion.
    
    Returns:
        Tuple[dict, bool]: (test_metrics, all_passed)
    """
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Prepare test data
        X_test = test_data.drop(columns=['fare_amount'], errors='ignore')
        y_test = test_data['fare_amount'] if 'fare_amount' in test_data else None
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_test - y_pred)),
            'r2': 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        }
        
        # Check against thresholds
        all_passed = True
        for metric, threshold in TEST_METRIC_THRESHOLDS.items():
            if (metric == 'r2' and metrics[metric] < threshold) or \
               (metric != 'r2' and metrics[metric] > threshold):
                logger.warning(f"Test failed: {metric} = {metrics[metric]:.2f} (threshold: {threshold})")
                all_passed = False
            else:
                logger.info(f"Test passed: {metric} = {metrics[metric]:.2f} (threshold: {threshold})")
        
        return metrics, all_passed
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise
