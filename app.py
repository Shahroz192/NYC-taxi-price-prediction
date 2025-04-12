import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from haversine import haversine, Unit
import os
import logging
from datetime import datetime

# --- Configuration ---
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow settings from environment variables
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "nyc-taxi-fare-regressor")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production") # Stage to load from Registry

# Define expected features based on training
# IMPORTANT: Must match the features used in train.py and preprocess.py
EXPECTED_FEATURES = [
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude', 'passenger_count', 'hour',
    'day_of_week', 'month', 'year', 'distance_km'
]
# --- End Configuration ---


# --- Model Loading ---
model = None
model_version = None

def load_model():
    """Loads the specified model stage from MLflow Model Registry."""
    global model, model_version
    logger.info(f"Attempting to load model '{MODEL_NAME}' stage '{MODEL_STAGE}' from {MLFLOW_TRACKING_URI}")
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        if latest_versions:
            model_version = latest_versions[0].version
            logger.info(f"Successfully loaded model '{MODEL_NAME}' version '{model_version}' (Stage: {MODEL_STAGE})")
        else:
             logger.warning(f"Model '{MODEL_NAME}' stage '{MODEL_STAGE}' loaded, but couldn't retrieve version details.")

    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}", exc_info=True)
        # Depending on the use case, you might want the app to fail startup
        # or handle predictions differently if the model doesn't load.
        # For now, we'll let predictions fail if the model is None.
        model = None

# Load the model when the application starts
load_model()
# --- End Model Loading ---


# --- API Definition ---
app = FastAPI(title="NYC Taxi Fare Predictor", version="1.0.0")

# Define input data schema using Pydantic
# Ensures type validation and generates OpenAPI docs
class TaxiRideFeatures(BaseModel):
    pickup_datetime: datetime = Field(..., description="Timestamp of the pickup (e.g., '2023-10-26T10:00:00Z')")
    pickup_longitude: float = Field(..., example=-73.985428)
    pickup_latitude: float = Field(..., example=40.748817)
    dropoff_longitude: float = Field(..., example=-73.987885)
    dropoff_latitude: float = Field(..., example=40.744015)
    passenger_count: int = Field(..., ge=1, le=6, example=1) # Add validation

    # Pydantic model configuration
    class Config:
        # Example for OpenAPI documentation
        json_schema_extra = {
            "example": {
                "pickup_datetime": "2024-01-15T09:30:00Z",
                "pickup_longitude": -73.9854,
                "pickup_latitude": 40.7488,
                "dropoff_longitude": -73.9780,
                "dropoff_latitude": 40.7648,
                "passenger_count": 1
            }
        }


# Define output data schema
class PredictionOut(BaseModel):
    predicted_fare: float = Field(..., example=15.5)
    model_version: str | None = Field(None, example="5") # Optionally return model version


def preprocess_input(ride: TaxiRideFeatures) -> pd.DataFrame | None:
    """Preprocesses raw input features into a DataFrame suitable for the model."""
    try:
        # Calculate distance
        distance_km = haversine(
            (ride.pickup_latitude, ride.pickup_longitude),
            (ride.dropoff_latitude, ride.dropoff_longitude),
            unit=Unit.KILOMETERS
        )

        dt = ride.pickup_datetime
        hour = dt.hour
        day_of_week = dt.dayofweek
        month = dt.month
        year = dt.year

        data = {
            'pickup_longitude': ride.pickup_longitude,
            'pickup_latitude': ride.pickup_latitude,
            'dropoff_longitude': ride.dropoff_longitude,
            'dropoff_latitude': ride.dropoff_latitude,
            'passenger_count': ride.passenger_count,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'year': year,
            'distance_km': distance_km
        }

        # Create DataFrame and ensure column order matches training
        input_df = pd.DataFrame([data])
        input_df = input_df[EXPECTED_FEATURES] # Reorder columns to match model expectation
        return input_df

    except Exception as e:
        logger.error(f"Error during input preprocessing: {e}", exc_info=True)
        return None


@app.post("/predict", response_model=PredictionOut)
async def predict_fare(ride_features: TaxiRideFeatures):
    """
    Predicts the taxi fare based on input ride features.
    Requires loading a model from MLflow registry during startup.
    """
    if model is None:
        logger.error("Model is not loaded. Cannot make predictions.")
        raise HTTPException(status_code=503, detail="Model is not available. Service temporarily unavailable.")

    logger.info(f"Received prediction request: {ride_features.model_dump()}")

    # Preprocess the input data
    input_df = preprocess_input(ride_features)
    if input_df is None:
         raise HTTPException(status_code=400, detail="Invalid input data or preprocessing failed.")

    logger.info(f"Preprocessed input DataFrame shape: {input_df.shape}")

    try:
        prediction = model.predict(input_df)
        predicted_fare = float(prediction[0]) # Ensure result is a standard float
        logger.info(f"Prediction successful. Predicted fare: {predicted_fare}")

        return PredictionOut(predicted_fare=predicted_fare, model_version=model_version)

    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during prediction.")


@app.get("/health", summary="Health Check")
async def health_check():
    """Returns 'ok' if the service is running, checks if model is loaded."""
    status = "ok" if model is not None else "error: model not loaded"
    return {"status": status, "model_name": MODEL_NAME, "model_stage": MODEL_STAGE, "model_version": model_version}

