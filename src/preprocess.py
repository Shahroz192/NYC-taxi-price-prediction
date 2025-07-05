import pandas as pd
from haversine import haversine, Unit
import numpy as np
import os
import yaml

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_config = config["data"]
preprocessing_config = config["preprocessing"]

RAW_DATA_PATH = data_config["raw_data_path"]
PROCESSED_DATA_DIR = data_config["processed_data_dir"]
SOURCE_FOR_NEW_DIR = data_config["source_for_new_dir"]
HISTORICAL_DATA_PATH = data_config["historical_data_path"]
FUTURE_DATA_PATH = data_config["future_data_path"]

N_ROWS_TO_LOAD = preprocessing_config["n_rows_to_load"]
FUTURE_DATA_FRACTION = preprocessing_config["future_data_fraction"]


os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SOURCE_FOR_NEW_DIR, exist_ok=True)


def calculate_distance(df):
    """Calculates the Haversine distance between pickup and dropoff points with input validation."""
    required_cols = [
        "pickup_latitude",
        "pickup_longitude",
        "dropoff_latitude",
        "dropoff_longitude",
    ]

    # Validate required columns exist
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one or more required columns: {required_cols}")

    # Vectorized haversine calculation
    distances = np.vectorize(haversine)(
        (df["pickup_latitude"], df["pickup_longitude"]),
        (df["dropoff_latitude"], df["dropoff_longitude"]),
        unit=Unit.KILOMETERS,
    )
    return distances


def extract_time_features(df):
    """Extracts time-based features from the pickup_datetime column."""
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df.dropna(subset=["pickup_datetime"], inplace=True)
    df["hour"] = df["pickup_datetime"].dt.hour
    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
    df["month"] = df["pickup_datetime"].dt.month
    df["year"] = df["pickup_datetime"].dt.year
    return df


def clean_data(df):
    """Cleans the dataframe by handling missing values and outliers."""
    df.dropna(
        subset=[
            "fare_amount",
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
        ],
        inplace=True,
    )

    df = df[(df["passenger_count"] > 0) & (df["passenger_count"] <= 6)]

    df = df[df["fare_amount"] > 0]

    df = df[(df["pickup_longitude"] >= -75) & (df["pickup_longitude"] <= -73)]
    df = df[(df["pickup_latitude"] >= 40) & (df["pickup_latitude"] <= 42)]
    df = df[(df["dropoff_longitude"] >= -75) & (df["dropoff_longitude"] <= -73)]
    df = df[(df["dropoff_latitude"] >= 40) & (df["dropoff_latitude"] <= 42)]

    if "distance_km" in df.columns:
        df = df[df["distance_km"] > 0]

    df = df[(df["fare_amount"] >= 2.5) & (df["fare_amount"] <= 200)]

    return df


def create_dataset():
    """
    Loads sample, processes, splits into historical/future by time, and saves them.
    This function should be run ONCE initially.
    """
    print("--- Creating Dataset ---")
    print(f"Loading {N_ROWS_TO_LOAD} rows from {RAW_DATA_PATH}...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, nrows=N_ROWS_TO_LOAD)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        return False

    print("Initial sample shape:", df.shape)

    print("Cleaning data...")
    df = clean_data(df)

    print("Shape after cleaning:", df.shape)

    print("Extracting time features...")
    df = extract_time_features(df)
    if "pickup_datetime" not in df.columns:
        print(
            "Error: 'pickup_datetime' column not found after time feature extraction."
        )
        return False

    print("Shape after time features:", df.shape)

    print("Calculating distance...")
    df["distance_km"] = calculate_distance(df)
    df = clean_data(df)
    print("Shape after distance calculation:", df.shape)

    print("Sorting data by pickup_datetime...")
    df.sort_values(by="pickup_datetime", inplace=True)

    split_index = int(len(df) * (1 - FUTURE_DATA_FRACTION))
    historical_df = df.iloc[:split_index]
    future_df = df.iloc[split_index:]

    print(f"Historical data shape: {historical_df.shape}")
    print(f"Future data shape: {future_df.shape}")

    print(f"Saving historical data to {HISTORICAL_DATA_PATH}...")
    historical_df.to_csv(HISTORICAL_DATA_PATH, index=False)

    print(f"Saving future simulation data to {FUTURE_DATA_PATH}...")
    future_df.to_csv(FUTURE_DATA_PATH, index=False)

    print("--- Dataset Creation Complete ---")
    return True


if __name__ == "__main__":
    create_dataset()
