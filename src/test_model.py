import yaml
import logging
import mlflow
import pandas as pd
from typing import Tuple
from utils import calculate_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    experiment_config = config["experiment"]
    model_config = config["model"]

    TEST_METRIC_THRESHOLDS = experiment_config["test_metric_thresholds"]
    FEATURES_TO_DROP = model_config["features_to_drop"]
except (FileNotFoundError, KeyError) as e:
    logger.error(f"Configuration error: {e}")
    raise


def run_model_tests(model_uri: str, test_data: pd.DataFrame) -> Tuple[dict, bool]:
    """
    Run automated tests on the model before promotion.

    Returns:
        Tuple[dict, bool]: (test_metrics, all_passed)
    """
    try:
        model = mlflow.pyfunc.load_model(model_uri)

        X_test = test_data.drop(
            columns=["fare_amount"] + FEATURES_TO_DROP, errors="ignore"
        )
        y_test = test_data["fare_amount"]

        y_pred = model.predict(X_test)

        metrics = calculate_metrics(y_test, y_pred)
        logger.info(f"Calculated test metrics: {metrics}")

        all_passed = True
        for metric, threshold in TEST_METRIC_THRESHOLDS.items():
            if metric == "r2":
                if metrics[metric] < threshold:
                    logger.warning(
                        f"Test FAILED for {metric}: {metrics[metric]:.4f} < {threshold}"
                    )
                    all_passed = False
                else:
                    logger.info(
                        f"Test PASSED for {metric}: {metrics[metric]:.4f} >= {threshold}"
                    )
            else:
                if metrics[metric] > threshold:
                    logger.warning(
                        f"Test FAILED for {metric}: {metrics[metric]:.4f} > {threshold}"
                    )
                    all_passed = False
                else:
                    logger.info(
                        f"Test PASSED for {metric}: {metrics[metric]:.4f} <= {threshold}"
                    )

        return metrics, all_passed

    except Exception as e:
        logger.error(f"Error during model testing: {e}", exc_info=True)
        raise
