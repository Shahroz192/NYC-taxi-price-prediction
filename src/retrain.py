import os
import yaml
import mlflow
import shutil 
import logging
import pandas as pd
import mlflow.sklearn 
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from train import calculate_metrics, MODEL_PARAMS
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split
from evidently import Report


NEW_DATA_DIR = os.getenv("NEW_DATA_DIR", "data/source_for_new")
DRIFT_DETECTION_FEATURES = ['distance_km', 'hour']  # Key features to monitor for drift
DRIFT_THRESHOLD = 0.05  # KS test p-value threshold
PERFORMANCE_DEGRADATION_THRESHOLD = 1.2  # RMSE increase factor to trigger alert
REPORT_DIR = os.getenv('REPORT_DIR', 'reports')  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = "NYC Taxi Fare Prediction"
REGISTERED_MODEL_NAME = "nyc-taxi-fare-regressor"

PROCESSED_NEW_DATA_DIR = os.getenv("PROCESSED_NEW_DATA_DIR", "data/source_for_new") # Where to move files after processing

os.makedirs(PROCESSED_NEW_DATA_DIR, exist_ok=True)

PROMOTION_THRESHOLD_FACTOR = 0.98 


def get_production_model_metrics(client: MlflowClient, model_name: str):
    """Fetches metrics for the current model in the Production stage."""
    try:
        production_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not production_versions:
            logging.warning(f"No model found in Production stage for '{model_name}'. This is expected for the first run.")
            return None, None  # Return None for both values to indicate no baseline

        prod_model_version = production_versions[0]
        prod_run_id = prod_model_version.run_id
        prod_run = client.get_run(prod_run_id)
        prod_rmse = prod_run.data.metrics.get("test_rmse")
        
        if prod_rmse is None:
            logging.warning("Production model exists but has no test_rmse metric")
            return None, None

        logging.info(f"Current Production Model: Version={prod_model_version.version}, RunID={prod_run_id}, Test RMSE={prod_rmse:.4f}")
        return prod_model_version, prod_rmse

    except Exception as e:
        logging.error(f"Error fetching production model metrics: {e}", exc_info=True)
        return None, None


def detect_data_drift(historical_data: pd.DataFrame, new_data: pd.DataFrame) -> dict:
    """
    Detect data drift using Evidently AI.
    Generates both JSON metrics and HTML reports.
    """
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    report = Report([DataDriftPreset()])
    report.run(reference_data=historical_data, current_data=new_data)
    report.save_html(os.path.join(REPORT_DIR, "data_drift_report.html"))
    
    return report.as_dict()

def load_and_preprocess_new_data(data_path: str):
    """
    Loads and preprocesses new data for retraining.
    Returns X_train, X_test, y_train, y_test
    """
    try:
        df = pd.read_csv(data_path)
        
        X = df.drop(["fare_amount","key","pickup_latitude","pickup_longitude",
                    "dropoff_latitude","dropoff_longitude","pickup_datetime"], axis=1)
        y = df["fare_amount"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logging.error(f"Error loading and preprocessing new data: {e}")
        return None, None, None, None


def run_retraining_pipeline():
    """
    Checks for new data, retrains the model, compares performance,
    and promotes the new model if it's better.
    """
    logging.info("--- Starting Retraining Pipeline ---")

    # 1. Check for new data files
    try:
        new_files = [os.path.join(NEW_DATA_DIR, f) for f in os.listdir(NEW_DATA_DIR) if f.endswith('.csv')]
    except FileNotFoundError:
        logging.error(f"New data directory not found: {NEW_DATA_DIR}")
        return # Exit if the directory doesn't exist

    if not new_files:
        logging.info("No new data files found. Exiting pipeline.")
        return

    logging.info(f"Found {len(new_files)} new data file(s): {new_files}")

   
    df_list = []
    for f in new_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            logging.error(f"Error reading new data file {f}: {e}")
            continue 

    if not df_list:
        logging.error("No valid new data could be read. Exiting pipeline.")
        return

    new_data_df = pd.concat(df_list, ignore_index=True)
    logging.info(f"Combined new data shape: {new_data_df.shape}")

    # Save the combined DataFrame to a temporary file
    temp_combined_path = os.path.join(NEW_DATA_DIR, "temp_combined_new_data.csv")
    new_data_df.to_csv(temp_combined_path, index=False)
    logging.info(f"Temporarily saved combined new data to {temp_combined_path}")

    # 2. Preprocess the new data
    logging.info("Preprocessing new data...")
    X_train_new, X_test_new, y_train_new, y_test_new = load_and_preprocess_new_data(temp_combined_path)
        
    # Clean up temporary file
    os.remove(temp_combined_path)
    logging.info(f"Removed temporary file {temp_combined_path}")

    if X_train_new is None:
        logging.error("Preprocessing of new data failed. Exiting pipeline.")
        return

    # 3. Set up MLflow and Client
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 4. Load historical data and check for drift
    try:
        historical_data_path = os.getenv("HISTORICAL_DATA_PATH", "data/processed/historical_data.csv")
        historical_data = pd.read_csv(historical_data_path)
        logging.info(f"Loaded historical data from {historical_data_path}")
        
        # Enhanced drift detection with Evidently
        drift_results = detect_data_drift(historical_data, new_data_df)
        
        # Log drift metrics
        drift_detected = any(result['drift_detected'] for result in drift_results.values())
        mlflow.log_metric("drift_detected", int(drift_detected))
        
        if drift_detected:
            logging.warning("Significant data drift detected!")
            mlflow.set_tag("drift_status", "detected")
        else:
            mlflow.set_tag("drift_status", "ok")
    except Exception as e:
        logging.error(f"Error loading historical data or detecting drift: {e}")

    # 4. Get current production model metrics
    logging.info("Fetching current production model metrics...")
    prod_model_version_obj, current_prod_rmse = get_production_model_metrics(client, REGISTERED_MODEL_NAME)

    # 5. Train a new model on the new data
    logging.info("Starting MLflow run for retraining...")
    with mlflow.start_run(run_name="retraining_run") as run:
        new_run_id = run.info.run_id
        logging.info(f"MLflow Run ID for retraining: {new_run_id}")
        mlflow.log_param("retraining_data_source", ", ".join(new_files)) # Log source files
        mlflow.log_param("retraining_data_shape", new_data_df.shape)

        logging.info("Training new model...")
        # model_new = xgb.XGBRegressor(**MODEL_PARAMS) # XGBoost
        model_new = RandomForestRegressor(**MODEL_PARAMS) # Scikit-learn RandomForest
        model_new.fit(X_train_new, y_train_new)
        logging.info("New model training complete.")

        # 6. Evaluate the new model (use the *test* split from the new data)
        logging.info("Evaluating new model...")
        y_pred_test_new = model_new.predict(X_test_new)
        new_metrics = calculate_metrics(y_test_new, y_pred_test_new)
        new_rmse = new_metrics.get("rmse", float('inf'))
        logging.info(f"New Model Test Metrics: {new_metrics}")

        # Log parameters and metrics for the new run
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metrics({f"test_{k}": v for k, v in new_metrics.items()}) # Use 'test_' prefix

        # Log the newly trained model artifact
        logging.info("Logging new model artifact to MLflow...")
        # mlflow.xgboost.log_model(xgb_model=model_new, artifact_path="model")
        mlflow.sklearn.log_model(sk_model=model_new, artifact_path="model")
        new_model_uri = f"runs:/{new_run_id}/model"

        # 7. Compare and Promote (or Archive)
        logging.info(f"Comparing New Model RMSE ({new_rmse}) vs Production RMSE ({current_prod_rmse})")
        
        from test_model import run_model_tests
        try:
            test_metrics, tests_passed = run_model_tests(new_model_uri, test_data)
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
            mlflow.log_metric("tests_passed", int(tests_passed))
            
            if not tests_passed:
                logging.error("Automated tests failed! Aborting promotion.")
                mlflow.log_metric("promoted_to_production", 0)
                mlflow.set_tag("status", "tests_failed")
                return
                
        except Exception as e:
            logging.error(f"Error during automated testing: {e}")
            mlflow.log_metric("promoted_to_production", 0)
            mlflow.set_tag("status", "testing_error")
            return
        
        # Enhanced promotion logic with performance degradation check
        performance_ratio = new_rmse / current_prod_rmse
        
        if (performance_ratio > PERFORMANCE_DEGRADATION_THRESHOLD):
            logging.error(f"Performance degradation detected! New RMSE is {performance_ratio:.2f}x worse than production. Aborting promotion.")
            mlflow.log_metric("promoted_to_production", 0)
            mlflow.set_tag("status", "performance_degradation")
        elif new_rmse < (current_prod_rmse * PROMOTION_THRESHOLD_FACTOR):
            logging.info("New model performance is better. Promoting to Production.")
            try:
                # Register the new model version first
                registered_model = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=new_model_uri,
                    run_id=new_run_id,
                    description=f"Retrained model from run {new_run_id} with data {', '.join(new_files)}"
                )
                logging.info(f"Registered new model version: {registered_model.version}")

                # Transition the new version to Production
                client.transition_model_version_stage(
                    name=REGISTERED_MODEL_NAME,
                    version=registered_model.version,
                    stage="Production",
                    archive_existing_versions=True # Archive the old Production version
                )
                logging.info(f"Successfully transitioned model version {registered_model.version} to Production stage.")
                mlflow.log_metric("promoted_to_production", 1) # Log success
                mlflow.set_tag("status", "promoted")

            except Exception as e:
                logging.error(f"Error during model registration or promotion: {e}", exc_info=True)
                mlflow.log_metric("promoted_to_production", 0) # Log failure
                mlflow.set_tag("status", "promotion_failed")
        else:
            logging.info("New model performance is not significantly better. Keeping current Production model.")
            # Optionally register the model but keep it in 'Staging' or 'None' stage
            try:
                registered_model = client.create_model_version(
                    name=REGISTERED_MODEL_NAME,
                    source=new_model_uri,
                    run_id=new_run_id,
                    description=f"Retrained model (candidate) from run {new_run_id}. Not promoted."
                )
                logging.info(f"Registered candidate model version: {registered_model.version} (Stage: None)")
                client.transition_model_version_stage( 
                    name=REGISTERED_MODEL_NAME,
                    version=registered_model.version,
                    stage="None"
                )
            except Exception as e:
                logging.error(f"Error registering candidate model version: {e}", exc_info=True)

            mlflow.log_metric("promoted_to_production", 0)
            mlflow.set_tag("status", "candidate_rejected")

    logging.info(f"Moving processed data files from {NEW_DATA_DIR} to {PROCESSED_NEW_DATA_DIR}...")
    for f_path in new_files:
        try:
            base_name = os.path.basename(f_path)
            shutil.move(f_path, os.path.join(PROCESSED_NEW_DATA_DIR, base_name))
        except Exception as e:
            logging.error(f"Error moving file {f_path}: {e}")

    logging.info("--- Retraining Pipeline Finished ---")


if __name__ == "__main__":
    run_retraining_pipeline()


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow_config = config["mlflow"]
data_config = config["data"]
model_config = config["model"]
retrain_config = config["retrain"]

MLFLOW_TRACKING_URI = mlflow_config["tracking_uri"]
MLFLOW_EXPERIMENT_NAME = mlflow_config["experiment_name"]
REGISTERED_MODEL_NAME = mlflow_config["registered_model_name"]

NEW_DATA_DIR = data_config["source_for_new_dir"]
REPORT_DIR = data_config["report_dir"]
HISTORICAL_DATA_PATH = data_config["historical_data_path"]

MODEL_PARAMS = model_config["params"]

DRIFT_DETECTION_FEATURES = retrain_config["drift_detection_features"]
DRIFT_THRESHOLD = retrain_config["drift_threshold"]
PERFORMANCE_DEGRADATION_THRESHOLD = retrain_config["performance_degradation_threshold"]
PROMOTION_THRESHOLD_FACTOR = retrain_config["promotion_threshold_factor"]
