"""
Feature engineering for Edersee water level forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """
    Load daily water level data from CSV and ensure a clean index.

    Args:
        csv_path (Path): Path to the processed CSV file.

    Returns:
        pd.DataFrame: Daily water level DataFrame with a DateTimeIndex.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        ValueError: If the required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
        
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    
    # Validate required columns
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"CSV must contain 'timestamp' and 'value' columns")
        
    df.set_index("timestamp", inplace=True)
    df = df.resample("D").mean()
    
    # Handle missing values
    missing_count = df["value"].isna().sum()
    if missing_count > 0:
        df = df.interpolate(method="time")
        
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic calendar features for yearly and weekly seasonality.

    Args:
        df (pd.DataFrame): DataFrame with a DateTimeIndex and at least one column 'value'.

    Returns:
        pd.DataFrame: DataFrame with added sine/cosine features.
        
    Raises:
        ValueError: If the DataFrame doesn't have a DateTimeIndex or is missing the 'value' column.
    """
    # Validate input
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DateTimeIndex")
    if "value" not in df.columns:
        raise ValueError("DataFrame must contain a 'value' column")
    
    daily_df = df.copy()
    
    # Yearly seasonality
    doy = daily_df.index.dayofyear
    ang_yr = 2 * np.pi * doy / 365.25
    daily_df["sin365"] = np.sin(ang_yr)
    daily_df["cos365"] = np.cos(ang_yr)

    # Weekly seasonality
    dow = daily_df.index.dayofweek
    ang_wk = 2 * np.pi * dow / 7
    daily_df["sin7"] = np.sin(ang_wk)
    daily_df["cos7"] = np.cos(ang_wk)

    # Ensure column order: target first
    cols = ["value", "sin365", "cos365", "sin7", "cos7"]
    return daily_df[cols]


def create_forecast_window(processed: pd.DataFrame,
                           scaler,
                           lookback: int = 365) -> np.ndarray:
    """
    Prepare the most recent window for forecasting.

    Args:
        processed (pd.DataFrame): Feature-engineered DataFrame.
        scaler: Fitted scaler (e.g., StandardScaler).
        lookback (int): Number of days to look back.

    Returns:
        np.ndarray: Array of shape (1, lookback, n_features) ready for model input.
        
    Raises:
        ValueError: If there's insufficient historical data for the lookback window.
    """
    if len(processed) < lookback:
        raise ValueError(f"Insufficient data: need at least {lookback} days, but got {len(processed)}")
    
    # Transform the entire dataset
    arr = scaler.transform(processed)

    # Use the most recent data
    X = arr[-lookback:]
    
    return X.reshape(1, lookback, arr.shape[1])