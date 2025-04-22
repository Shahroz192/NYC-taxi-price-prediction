import mlflow
import mlflow.sklearn 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import os
import joblib 
import yaml

# Load config at module level so variables are importable
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

mlflow_config = config["mlflow"]
data_config = config["data"]
model_config = config["model"]

MLFLOW_TRACKING_URI = mlflow_config["tracking_uri"]
MLFLOW_EXPERIMENT_NAME = mlflow_config["experiment_name"]
REGISTERED_MODEL_NAME = mlflow_config["registered_model_name"]

MODEL_PARAMS = model_config["params"]

LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "model.joblib")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

def calculate_metrics(actual, predicted):
    """Calculates regression evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return {"rmse": rmse, "mae": mae, "r2": r2}

def train_model(X_train, X_test, y_train, y_test):
    """Loads data, trains model, evaluates, and logs to MLflow."""
    print("Starting model training process...")

    print(f"Setting MLflow tracking URI to: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Setting MLflow experiment to: {MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Start MLflow run
    print("Starting MLflow run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log dataset parameters
        mlflow.log_param("train_data_shape", X_train.shape)
        mlflow.log_param("test_data_shape", X_test.shape)

        # Initialize and Train Model
        print("Initializing and training the model...")
        model = RandomForestRegressor(**MODEL_PARAMS) 
        model.fit(X_train, y_train)
        print("Model training complete.")

        # 3. Evaluate Model
        print("Evaluating the model...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_metrics = calculate_metrics(y_train, y_pred_train)
        test_metrics = calculate_metrics(y_test, y_pred_test)

        print(f"Train Metrics: {train_metrics}")
        print(f"Test Metrics: {test_metrics}")

        # 4. Log to MLflow
        print("Logging parameters, metrics, and model to MLflow...")
        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        # Create signature and input example
        input_example = X_train[:1]
        signature = mlflow.models.infer_signature(X_train, y_pred_train)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", 
            registered_model_name=REGISTERED_MODEL_NAME, 
            signature=signature,
            input_example=input_example
        )

        client = mlflow.tracking.MlflowClient()
        production_versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])
        if production_versions:
            model_version = production_versions[0]
            print(f"Model is in production stage: Version {model_version.version}")
        else:
            print("No model found in production stage - this is expected for first run")

        print("MLflow logging complete.")
        print(f"View run in MLflow UI: {MLFLOW_TRACKING_URI}/#/experiments/{mlflow.active_run().info.experiment_id}/runs/{run_id}")

        # 5. Save model locally (optional)
        print(f"Saving model locally to {LOCAL_MODEL_PATH}...")
        joblib.dump(model, LOCAL_MODEL_PATH)
        print("Local model saving complete.")

    print("Model training process finished.")

if __name__ == "__main__":
    df = pd.read_csv(data_config["historical_data_path"])
    X = df.drop(["fare_amount","key","pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude","pickup_datetime"], axis=1)
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_model(X_train, X_test, y_train, y_test)
    print("Training completed and model saved.")
