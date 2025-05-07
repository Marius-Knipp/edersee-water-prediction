"""
Load the trained LSTM and scaler to generate forecasts.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
from tensorflow.keras.models import load_model
import logging
from src.features.preprocessing import create_forecast_window

# Configure logging
logger = logging.getLogger(__name__)


def load_artifacts(model_path: Path, scaler_path: Path) -> Tuple:
    """
    Load the trained LSTM model and the fitted scaler.
    
    Args:
        model_path (Path): Path to the saved Keras model
        scaler_path (Path): Path to the saved scaler object
    
    Returns:
        Tuple: (model, scaler) - The loaded model and scaler objects
        
    Raises:
        FileNotFoundError: If either file doesn't exist
        ValueError: If loading fails
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    try:
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise ValueError(f"Error loading model: {str(e)}")
        
    try:
        logger.info(f"Loading scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        raise ValueError(f"Error loading scaler: {str(e)}")
        
    return model, scaler


def forecast_next_days(
    processed_df: pd.DataFrame,
    model,
    scaler,
    lookback: int = 365,
) -> Union[pd.Series, Dict[str, pd.Series]]:
    """
    Forecast water levels for the next 30 days.
    
    Note: The model is specifically trained to output exactly 30 days of predictions at once.

    Args:
        processed_df (pd.DataFrame): Data with calendar features.
        model: Trained LSTM model.
        scaler: Fitted scaler.
        lookback (int): Look-back window length.

    Returns:
        Union[pd.Series, Dict[str, pd.Series]]: Forecasted values indexed by date.
    """
    # Fixed forecast horizon - model is trained for exactly 30 days
    horizon = 30
    
    logger.info(f"Generating {horizon}-day forecast with {lookback}-day lookback window")
    
    # Validate input data
    if len(processed_df) < lookback:
        raise ValueError(f"Insufficient historical data: {len(processed_df)} days available, {lookback} days required")
    
    # Prepare input window from historical data
    X = create_forecast_window(processed_df, scaler, lookback)
    
    # Make prediction (outputs all 30 days at once)
    predictions_scaled = model.predict(X, verbose=0)[0]
    
    # Reverse scaling for the target only
    mean, scale = scaler.mean_[0], scaler.scale_[0]
    predictions = predictions_scaled * scale + mean
    
    # Build date index for the forecast
    future_idx = pd.date_range(processed_df.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
    
    forecast = pd.Series(predictions, index=future_idx, name="forecast")
    
    return forecast