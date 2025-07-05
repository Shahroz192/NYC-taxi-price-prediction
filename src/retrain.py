import os
import yaml
import mlflow
import shutil 
import logging
import pandas as pd
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from evidently.presets import DataDriftPreset
from evidently import Report
from utils import calculate_metrics
from test_model import run_model_tests

# --- Configuration Loading ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logging.error("config.yaml not found")
    raise
except KeyError:
    logging.error("Invalid config.yaml format. Missing 'mlflow' or 'data' keys.")
    raise

mlflow_config = config["mlflow"]
data_config = config["data"]
model_config = config["model"]
retrain_config = config["retrain"]

# MLflow settings
MLFLOW_TRACKING_URI = mlflow_config["tracking_uri"]
MLFLOW_EXPERIMENT_NAME = mlflow_config["experiment_name"]
REGISTERED_MODEL_NAME = mlflow_config["registered_model_name"]

# Data paths
NEW_DATA_DIR = data_config["source_for_new_dir"]
PROCESSED_NEW_DATA_DIR = data_config["processed_new_data_dir"]
HISTORICAL_DATA_PATH = data_config["historical_data_path"]
REPORT_DIR = data_config["report_dir"]

# Model settings
MODEL_PARAMS = model_config["params"]
FEATURES_TO_DROP = model_config["features_to_drop"]

# Retraining thresholds
DRIFT_DETECTION_FEATURES = retrain_config["drift_detection_features"]
PERFORMANCE_DEGRADATION_THRESHOLD = retrain_config["performance_degradation_threshold"]
PROMOTION_THRESHOLD_FACTOR = retrain_config["promotion_threshold_factor"]

# --- Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.makedirs(PROCESSED_NEW_DATA_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def get_production_model_metrics(client: MlflowClient, model_name: str):
    """Fetches metrics for the current model in the Production stage."""
    try:
        production_versions = client.get_latest_versions(
            model_name, stages=["Production"]
        )
        if not production_versions:
            logging.warning(
                f"No model found in Production stage for '{model_name}'. This is expected for the first run."
            )
            return None, float("inf")

        prod_model_version = production_versions[0]
        prod_run_id = prod_model_version.run_id
        prod_run = client.get_run(prod_run_id)
        prod_rmse = prod_run.data.metrics.get("test_rmse")

        if prod_rmse is None:
            logging.warning(
                "Production model exists but has no test_rmse metric. Treating as infinity."
            )
            return prod_model_version, float("inf")

        logging.info(
            f"Current Production Model: Version={prod_model_version.version}, RunID={prod_run_id}, Test RMSE={prod_rmse:.4f}"
        )
        return prod_model_version, prod_rmse

    except Exception as e:
        logging.error(f"Error fetching production model metrics: {e}", exc_info=True)
        return None, float("inf")


def detect_data_drift(historical_data: pd.DataFrame, new_data: pd.DataFrame) -> dict:
    """
    Detect data drift using Evidently AI.
    Generates both JSON metrics and HTML reports.
    """
    report = Report([DataDriftPreset(columns=DRIFT_DETECTION_FEATURES)])
    report.run(reference_data=historical_data, current_data=new_data)
    report.save_html(os.path.join(REPORT_DIR, "data_drift_report.html"))
    return report.as_dict()


def run_retraining_pipeline():
    """
    Checks for new data, retrains the model, compares performance,
    and promotes the new model if it's better.
    """
    logging.info("--- Starting Retraining Pipeline ---")

    # 1. Check for new data files
    try:
        new_files = [
            os.path.join(NEW_DATA_DIR, f)
            for f in os.listdir(NEW_DATA_DIR)
            if f.endswith(".csv")
        ]
    except FileNotFoundError:
        logging.error(f"New data directory not found: {NEW_DATA_DIR}")
        return

    if not new_files:
        logging.info("No new data files found. Exiting pipeline.")
        return

    logging.info(f"Found {len(new_files)} new data file(s): {', '.join(new_files)}")

    # 2. Load and combine new data
    df_list = [pd.read_csv(f) for f in new_files]
    new_data_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined new data shape: {new_data_df.shape}")

    # 3. Preprocess the new data
    logging.info("Preprocessing new data...")
    X_new = new_data_df.drop(
        columns=["fare_amount"] + FEATURES_TO_DROP, errors="ignore"
    )
    y_new = new_data_df["fare_amount"]
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(
        X_new, y_new, test_size=0.2, random_state=42
    )

    # 4. Set up MLflow and Client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 5. Load historical data and check for drift
    try:
        historical_data = pd.read_csv(HISTORICAL_DATA_PATH)
        logging.info(f"Loaded historical data from {HISTORICAL_DATA_PATH}")
        drift_results = detect_data_drift(historical_data, new_data_df)
        drift_detected = drift_results["data_drift"]["data"]["metrics"]["dataset_drift"]
        logging.info(f"Drift detection complete. Drift detected: {drift_detected}")
    except Exception as e:
        logging.error(f"Error loading historical data or detecting drift: {e}")
        logging.warning("Proceeding with caution due to drift detection failure")
        drift_detected = None
        mlflow.set_tag("drift_status", "detection_failed")
    logging.info("Fetching current production model metrics...")
    prod_model_version_obj, current_prod_rmse = get_production_model_metrics(
        client, REGISTERED_MODEL_NAME
    )

    logging.info("Starting MLflow run for retraining...")
    with mlflow.start_run(run_name="retraining_run") as run:
        new_run_id = run.info.run_id
        logging.info(f"MLflow Run ID for retraining: {new_run_id}")
        mlflow.log_param("retraining_data_source", ", ".join(new_files))
        mlflow.log_param("retraining_data_shape", new_data_df.shape)
        mlflow.log_metric("drift_detected", int(drift_detected))
        mlflow.set_tag("drift_status", "detected" if drift_detected else "ok")

        logging.info("Training new model...")
        model_new = RandomForestRegressor(**MODEL_PARAMS)
        model_new.fit(X_train_new, y_train_new)
        logging.info("New model training complete.")

        # 8. Evaluate the new model
        logging.info("Evaluating new model...")
        y_pred_test_new = model_new.predict(X_test_new)
        new_metrics = calculate_metrics(y_test_new, y_pred_test_new)
        new_rmse = new_metrics.get("rmse", float("inf"))
        logging.info(f"New Model Test Metrics: {new_metrics}")

        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metrics({f"test_{k}": v for k, v in new_metrics.items()})

        new_model_uri = f"runs:/{new_run_id}/model"
        mlflow.sklearn.log_model(
            sk_model=model_new,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # 9. Run automated tests
        logging.info("Running automated tests on the new model...")
        try:
            # The test data for `run_model_tests` should be the new data
            test_metrics, tests_passed = run_model_tests(
                new_model_uri, test_data=new_data_df
            )
            mlflow.log_metrics({f"post_test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("tests_passed", int(tests_passed))

            if not tests_passed:
                logging.error("Automated tests failed! Aborting promotion.")
                mlflow.set_tag("status", "tests_failed")
                return
        except Exception as e:
            logging.error(f"Error during automated testing: {e}", exc_info=True)
            mlflow.set_tag("status", "testing_error")
            return

        # 10. Compare and Promote
        logging.info(
            f"Comparing New Model RMSE ({new_rmse:.4f}) vs Production RMSE ({current_prod_rmse:.4f})"
        )

        is_better = new_rmse < (current_prod_rmse * PROMOTION_THRESHOLD_FACTOR)
        is_not_degraded = new_rmse < (
            current_prod_rmse * PERFORMANCE_DEGRADATION_THRESHOLD
        )

        if is_better and is_not_degraded:
            logging.info("New model performance is better. Promoting to Production.")
            try:
                # The model is already registered, just get the latest version
                latest_version = client.get_latest_versions(
                    REGISTERED_MODEL_NAME, stages=["None"]
                )[0]
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logging.info(
                    f"Successfully transitioned model version {latest_version.version} to Production."
                )
                mlflow.log_metric("promoted_to_production", 1)
                mlflow.set_tag("status", "promoted")
            except Exception as e:
                logging.error(f"Error during model promotion: {e}", exc_info=True)
                mlflow.set_tag("status", "promotion_failed")
        elif not is_not_degraded:
            logging.error(
                f"Performance degradation detected! New RMSE ({new_rmse:.4f}) is worse than threshold. Aborting promotion."
            )
            mlflow.set_tag("status", "performance_degradation")
        else:
            logging.info(
                "New model performance is not significantly better. Keeping current Production model."
            )
            mlflow.log_metric("promoted_to_production", 0)
            mlflow.set_tag("status", "candidate_rejected")

    # 11. Move processed files
    logging.info(f"Moving processed data files to {PROCESSED_NEW_DATA_DIR}...")
    for f_path in new_files:
        try:
            base_name = os.path.basename(f_path)
            shutil.move(f_path, os.path.join(PROCESSED_NEW_DATA_DIR, base_name))
        except Exception as e:
            logging.error(f"Error moving file {f_path}: {e}")

    logging.info("--- Retraining Pipeline Finished ---")


if __name__ == "__main__":
    run_retraining_pipeline()
