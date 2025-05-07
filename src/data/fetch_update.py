import requests
import pandas as pd
import logging
from pathlib import Path
from typing import Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# API configuration
API_BASE_URL = "https://www.pegelonline.wsv.de/webservices/rest-api/v2"
STATION_SHORTNAME = "edertalsperre"


def fetch_recent_data(days: int = 15) -> pd.DataFrame:
    """
    Fetch daily water level measurements for the last `days` days from the Edersee API.

    Args:
        days (int): Number of past days to fetch data for.

    Returns:
        pd.DataFrame: Daily-aggregated water level data indexed by timestamp.
        
    Raises:
        ValueError: If the station is not found.
        requests.RequestException: If API requests fail.
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

    # Find the Edersee station by shortname
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

    # Fetch measurements for the specified days
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

    # Convert to DataFrame and resample to daily
    df = pd.DataFrame(measurements)
    if df.empty:
        logger.warning("No measurements received from API")
        return pd.DataFrame(columns=["value"])
        
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    df = df.resample("D").mean()
    df.index = df.index.tz_localize(None)
    
    logger.info(f"Processed {len(df)} daily measurements")
    return df


def load_historical_data(csv_path: Path) -> Optional[pd.DataFrame]:
    """
    Load existing historical data from CSV.
    
    Args:
        csv_path (Path): Path to the historical CSV file.
        
    Returns:
        pd.DataFrame or None: The loaded DataFrame or None if file doesn't exist.
    """
    if not csv_path.exists():
        logger.warning(f"Historical data file not found: {csv_path}")
        return None
        
    try:
        logger.info(f"Loading historical data from {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df.set_index("timestamp", inplace=True)
        df = df.resample("D").mean()
        logger.info(f"Loaded {len(df)} historical records")
        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        raise


def update_historical_data(csv_path: Path | str, days: int = 15) -> pd.DataFrame:
    """
    Load existing historical data from CSV, fetch recent measurements, append new dates,
    and save the updated dataset back to CSV.

    Args:
        csv_path (Path | str): Path to the historical CSV file.
        days (int): Lookback window (in days) for fetching recent data.

    Returns:
        pd.DataFrame: The combined historical dataset.
    """
    csv_path = Path(csv_path)
    
    # Ensure parent directory exists
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data or create empty DataFrame
    old_df = load_historical_data(csv_path) if csv_path.exists() else pd.DataFrame(columns=["value"])

    # Fetch new data
    new_df = fetch_recent_data(days)
    if new_df.empty and (old_df is None or old_df.empty):
        logger.warning("No data available - both historical and new data are empty")
        return pd.DataFrame(columns=["value"])

    # Combine and remove duplicates
    if old_df is None:
        combined = new_df
    else:
        combined = pd.concat([old_df, new_df])
        
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Save back to CSV
    try:
        logger.info(f"Saving updated dataset to {csv_path}")
        combined.to_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise
        
    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Update historical water level data for the Edersee reservoir."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/processed/water_levels_daily.csv",
        help="Path to the historical CSV file.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=15,
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
    
    # Update log level if specified
    logger.setLevel(getattr(logging, args.log_level))
    
    try:
        updated_df = update_historical_data(args.csv, args.days)
        logger.info(f"Dataset updated: {len(updated_df)} total rows.")
    except Exception as e:
        logger.critical(f"Update failed: {e}")
        exit(1)