import pandas as pd
from haversine import haversine, Unit
from sklearn.model_selection import train_test_split
import numpy as np
import os


RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "data/train.csv")
PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR", "data/processed")
SOURCE_FOR_NEW_DIR = os.getenv("SOURCE_FOR_NEW_DIR", "data/source_for_new")
HISTORICAL_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "historical_data.csv")
FUTURE_DATA_PATH = os.path.join(SOURCE_FOR_NEW_DIR, "future_data.csv")

N_ROWS_TO_LOAD = 1_000_000 
FUTURE_DATA_FRACTION = 0.2 


os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(SOURCE_FOR_NEW_DIR, exist_ok=True)

def calculate_distance(df):
    """Calculates the Haversine distance between pickup and dropoff points with input validation."""
    required_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    
    # Validate required columns exist
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing one or more required columns: {required_cols}")
    

    distances = []
    for _, row in df.iterrows():
        start = (row['pickup_latitude'], row['pickup_longitude'])
        end = (row['dropoff_latitude'], row['dropoff_longitude'])
        try:
            distance = haversine(start, end, unit=Unit.KILOMETERS)
        except ValueError:
            distance = np.nan
        distances.append(distance)
    
    return np.array(distances)


def extract_time_features(df):
    """Extracts time-based features from the pickup_datetime column."""
    # Ensure the column is in datetime format, handling potential errors
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    # Drop rows where conversion failed
    df.dropna(subset=['pickup_datetime'], inplace=True)

    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    return df

def clean_data(df):
    """Cleans the dataframe by handling missing values and outliers."""
    df.dropna(subset=['fare_amount', 'pickup_longitude', 'pickup_latitude',
                      'dropoff_longitude', 'dropoff_latitude'], inplace=True)

    df = df[(df['passenger_count'] > 0) & (df['passenger_count'] <= 6)]

    df = df[df['fare_amount'] > 0]

    df = df[(df['pickup_longitude'] >= -75) & (df['pickup_longitude'] <= -73)]
    df = df[(df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 42)]
    df = df[(df['dropoff_longitude'] >= -75) & (df['dropoff_longitude'] <= -73)]
    df = df[(df['dropoff_latitude'] >= 40) & (df['dropoff_latitude'] <= 42)]

    if 'distance_km' in df.columns:
        df = df[df['distance_km'] > 0]


    df = df[(df['fare_amount'] >= 2.5) & (df['fare_amount'] <= 200)]

    return df


def create_simulation_data(raw_data_path=RAW_DATA_PATH,
                           n_rows=N_ROWS_TO_LOAD,
                           future_fraction=FUTURE_DATA_FRACTION):
    """
    Loads sample, processes, splits into historical/future by time, and saves them.
    This function should be run ONCE initially.
    """
    print(f"--- Creating Simulation Data ---")
    print(f"Loading {n_rows} rows from {raw_data_path}...")
    try:
        df = pd.read_csv(raw_data_path, nrows=n_rows)
    except FileNotFoundError:
        print(f"Error: Raw data file not found at {raw_data_path}")
        return False

    print("Initial sample shape:", df.shape)

    # Apply cleaning and feature engineering
    print("Cleaning data...")
    df = clean_data(df)
    print("Shape after cleaning:", df.shape)
    print("Extracting time features...")
    df = extract_time_features(df)
     # Ensure pickup_datetime is present after extraction
    if 'pickup_datetime' not in df.columns:
        print("Error: 'pickup_datetime' column not found after time feature extraction.")
        return False
    print("Shape after time features:", df.shape)
    print("Calculating distance...")
    df['distance_km'] = calculate_distance(df)
    df = clean_data(df) 
    print("Shape after distance calculation:", df.shape)


    print("Sorting data by pickup_datetime...")
    df.sort_values(by='pickup_datetime', inplace=True)

    # Split into historical and future
    split_index = int(len(df) * (1 - future_fraction))
    historical_df = df.iloc[:split_index]
    future_df = df.iloc[split_index:]

    print(f"Historical data shape: {historical_df.shape}")
    print(f"Future data shape: {future_df.shape}")

    # Save the datasets
    print(f"Saving historical data to {HISTORICAL_DATA_PATH}...")
    historical_df.to_csv(HISTORICAL_DATA_PATH, index=False)

    print(f"Saving future simulation data to {FUTURE_DATA_PATH}...")
    future_df.to_csv(FUTURE_DATA_PATH, index=False)

    print("--- Simulation Data Creation Complete ---")
    return True


if __name__ == "__main__":
    create_simulation_data()
