import requests
import pandas as pd
import logging
from pathlib import Path
from typing import Optional
import io # For S3
import boto3 # For S3
from botocore.exceptions import ClientError # For S3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__) # Use __name__

# API configuration
API_BASE_URL = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
STATION_SHORTNAME = "edertalsperre"

# S3 Configuration
S3_BUCKET_NAME = "edersee-water-level"
S3_DATA_KEY = "water_levels_daily.csv"

s3_client = boto3.client("s3")

def fetch_recent_data(days: int = 15) -> pd.DataFrame:
    """
    Fetch daily water level measurements for the last `days` days from the Edersee API.
    (Keep this function as is)
    """
    logger.info(f"Fetching station data from {API_BASE_URL}")
    stations_url = f"{API_BASE_URL}/stations.json"
    try:
        response = requests.get(stations_url)
        response.raise_for_status()
        stations = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch stations: {e}")
        raise

    edertalsperre = next(
        (s for s in stations if s.get("shortname", "").lower() == STATION_SHORTNAME),
        None,
    )
    if not edertalsperre:
        error_msg = f"Station '{STATION_SHORTNAME}' not found in API response"
        logger.error(error_msg)
        raise ValueError(error_msg)

    uuid = edertalsperre["uuid"]
    logger.info(f"Found station with UUID: {uuid}")

    measurements_url = (
        f"{API_BASE_URL}/stations/{uuid}/W/measurements.json?start=P{days}D"
    )
    try:
        logger.info(f"Fetching water level measurements for past {days} days")
        resp2 = requests.get(measurements_url)
        resp2.raise_for_status()
        measurements = resp2.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch measurements: {e}")
        raise

    df = pd.DataFrame(measurements)
    if df.empty:
        logger.warning("No measurements received from API")
        return pd.DataFrame(columns=["value"]) # Ensure it has a 'value' column for consistency

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.resample("D").mean() # Ensure 'value' column is present after resample
    df.index = df.index.tz_localize(None)
    logger.info(f"Processed {len(df)} daily measurements")
    return df[["value"]] # Select only the value column

def load_historical_data_from_s3() -> Optional[pd.DataFrame]:
    """
    Load existing historical data from S3.
    Returns:
        pd.DataFrame or None: The loaded DataFrame or None if file doesn't exist in S3.
    """
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set.")
        return None
    try:
        logger.info(f"Loading historical data from S3: s3://{S3_BUCKET_NAME}/{S3_DATA_KEY}")
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=S3_DATA_KEY)
        csv_content = response["Body"].read().decode("utf-8")
        df = pd.read_csv(io.StringIO(csv_content), parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        # Ensure daily frequency and mean aggregation if multiple entries for a day exist
        df = df.resample("D").mean()
        logger.info(f"Loaded {len(df)} historical records from S3")
        return df[["value"]] # Select only the value column
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning(f"Historical data file not found in S3: s3://{S3_BUCKET_NAME}/{S3_DATA_KEY}")
            return None
        else:
            logger.error(f"Error loading historical data from S3: {e}")
            raise
    except Exception as e:
        logger.error(f"Error processing historical data from S3: {e}")
        raise

def save_data_to_s3(df: pd.DataFrame):
    """Saves the DataFrame to S3."""
    if not S3_BUCKET_NAME:
        logger.error("S3_BUCKET_NAME environment variable not set. Cannot save to S3.")
        return
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=S3_DATA_KEY, Body=csv_buffer.getvalue())
        logger.info(f"Successfully saved updated dataset to s3://{S3_BUCKET_NAME}/{S3_DATA_KEY}")
    except Exception as e:
        logger.error(f"Failed to save data to S3: {e}")
        raise

def update_historical_data(days: int = 15) -> pd.DataFrame:
    """
    Load existing historical data from S3, fetch recent measurements, append new dates,
    and save the updated dataset back to S3.
    Args:
        days (int): Lookback window (in days) for fetching recent data from API.
    Returns:
        pd.DataFrame: The combined historical dataset.
    """
    old_df = load_historical_data_from_s3()

    # If old_df is None (not found or error), start with an empty DataFrame
    if old_df is None:
        old_df = pd.DataFrame(columns=["value"])
        old_df.index = pd.to_datetime(old_df.index) # Ensure DatetimeIndex

    fetch_days = days
    # If we have historical data, we only need to fetch data since the last record
    if not old_df.empty:
        last_date = old_df.index.max()
        days_since_last_update = (pd.Timestamp.now(tz='UTC').tz_localize(None) - last_date).days
        # Fetch a bit more to be safe, max of `days` or `days_since_last_update + buffer`
        fetch_days = max(days, days_since_last_update + 2 if days_since_last_update > 0 else days)
        logger.info(f"Last historical data: {last_date}. Fetching data for the last {fetch_days} days.")


    new_df = fetch_recent_data(days=fetch_days)

    if new_df.empty and old_df.empty:
        logger.warning("No data available - both historical and new data are empty")
        return pd.DataFrame(columns=["value"])

    # Combine and remove duplicates
    combined = pd.concat([old_df, new_df])
    if not combined.empty:
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        save_data_to_s3(combined) # Save the updated data back to S3
    else:
        logger.warning("Combined dataframe is empty. Not saving to S3.")
        return pd.DataFrame(columns=["value"]) # Return empty DataFrame

    return combined

# The __main__ block is useful for local testing or a separate update script,
# but the Streamlit app will call update_historical_data() directly.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update historical water level data for the Edersee reservoir from API and S3."
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        # default=os.environ.get("S3_BUCKET_NAME", "your-default-bucket-for-local-testing"), # Local default
        help="S3 bucket name for historical data.",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        # default=os.environ.get("S3_DATA_KEY", "processed/water_levels_daily.csv"), # Local default
        help="S3 key for the historical data CSV file.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=35, # Fetch a bit more than a month to ensure overlap
        help="Number of past days to fetch from the API.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    args = parser.parse_args()

    # Set S3 vars from CLI for local testing if provided
    if args.s3_bucket: S3_BUCKET_NAME = args.s3_bucket
    if args.s3_key: S3_DATA_KEY = args.s3_key

    if not S3_BUCKET_NAME or not S3_DATA_KEY:
        logger.error("S3 bucket name and key must be provided via arguments or environment variables (S3_BUCKET_NAME, S3_DATA_KEY) for local execution.")
        exit(1)

    logger.setLevel(getattr(logging, args.log_level))

    try:
        updated_df = update_historical_data(args.days)
        if not updated_df.empty:
             logger.info(f"Dataset updated from S3 and API: {len(updated_df)} total rows.")
        else:
             logger.warning("Dataset update resulted in an empty dataframe.")
    except Exception as e:
        logger.critical(f"Update failed: {e}")
        exit(1)