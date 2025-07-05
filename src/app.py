import yaml
import joblib
import logging
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from haversine import haversine, Unit
from datetime import datetime

# --- Configuration Loading ---
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_config = config["model"]
    LOCAL_MODEL_PATH = model_config["local_model_path"]
except FileNotFoundError:
    print("Error: config.yaml not found. Please ensure the file exists.")
    exit()
except KeyError:
    print(
        "Error: Invalid config.yaml format. Missing 'model' or 'local_model_path' keys."
    )
    exit()


# --- Model Loading ---
model = None
try:
    model = joblib.load(LOCAL_MODEL_PATH)
    logging.info(f"Model loaded successfully from {LOCAL_MODEL_PATH}")
except FileNotFoundError:
    logging.error(f"Error: Model file not found at {LOCAL_MODEL_PATH}")
    model = None
except Exception as e:
    logging.error(f"An unexpected error occurred while loading the model: {e}")
    model = None


app = FastAPI(title="NYC Taxi Fare Predictor", version="1.0.0")


class TaxiRideFeatures(BaseModel):
    pickup_datetime: str = Field(
        ...,
        example="2024-01-15T09:30:00",
        description="ISO format timestamp for the pickup",
    )
    pickup_longitude: float = Field(..., example=-73.9854)
    pickup_latitude: float = Field(..., example=40.7488)
    dropoff_longitude: float = Field(..., example=-73.9780)
    dropoff_latitude: float = Field(..., example=40.7648)
    passenger_count: int = Field(..., ge=1, le=6, example=1)


class PredictionOut(BaseModel):
    predicted_fare: float


def preprocess_input(ride: TaxiRideFeatures) -> pd.DataFrame:
    """Preprocesses raw input features into a DataFrame suitable for the model."""
    distance_km = haversine(
        (ride.pickup_latitude, ride.pickup_longitude),
        (ride.dropoff_latitude, ride.dropoff_longitude),
        unit=Unit.KILOMETERS,
    )
    try:
        dt = datetime.fromisoformat(ride.pickup_datetime)
    except ValueError as e:
        raise ValueError(f"Invalid datetime format: {e}")

    input_data = {
        "passenger_count": ride.passenger_count,
        "hour": dt.hour,
        "day_of_week": dt.dayofweek,
        "month": dt.month,
        "year": dt.year,
        "distance_km": distance_km,
    }

    return pd.DataFrame([input_data])


@app.post("/predict", response_model=PredictionOut)
async def predict_fare(ride_features: TaxiRideFeatures):
    """
    Predicts the taxi fare based on input ride features.
    """
    if model is None:
        logging.error("Model is not available.")
        raise HTTPException(status_code=503, detail="Model is not available.")

    try:
        input_df = preprocess_input(ride_features)

        prediction = model.predict(input_df)
        predicted_fare = float(prediction[0])

        return PredictionOut(predicted_fare=predicted_fare)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service and model status."""
    status = "ok" if model is not None else "error: model not loaded"
    return {"status": status}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
