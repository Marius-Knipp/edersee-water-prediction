import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import logging

# --- Setup Project Root Path ---
# This allows the app to find the 'src' module
# Assumes streamlit_app.py is in 'app/' and 'src/' is a sibling directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# --- Import your custom modules ---
from src.data.fetch_update import update_historical_data
from src.features.preprocessing import load_processed_data, add_calendar_features
from src.models.predict_lstm import load_artifacts, forecast_next_days

# --- Configure Logging (optional for Streamlit, but good practice) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration ---
HISTORICAL_DATA_CSV = PROJECT_ROOT / "data" / "processed" / "water_levels_daily.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "water_level_lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "LSTM_standard_scaler.joblib"

LOOKBACK_PERIOD = 365  # Model's lookback period
DAYS_TO_FETCH_API = 35 # How many recent days to fetch to ensure data freshness (>= horizon)
FORECAST_HORIZON = 30 # LSTM model predicts 30 days

# --- Caching Functions ---

@st.cache_resource # For objects that are expensive to create (like models, scalers)
def load_model_and_scaler():
    """Loads the Keras model and the scaler."""
    logger.info("Loading model and scaler...")
    try:
        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model/scaler: {e}")
        st.error(f"Could not load model and scaler: {e}")
        return None, None

@st.cache_data # For data transformations that are pure functions of their inputs
def get_prepared_data(csv_path: Path, _update_trigger):
    """
    Loads processed data, adds calendar features.
    _update_trigger is a dummy argument to bust cache when data is updated.
    """
    logger.info(f"Loading and preprocessing data from {csv_path}...")
    try:
        df_raw = load_processed_data(csv_path)
        df_processed = add_calendar_features(df_raw)
        logger.info(f"Data loaded and preprocessed. Shape: {df_processed.shape}")
        return df_processed
    except FileNotFoundError:
        st.error(f"Historical data file not found at {csv_path}. Please run data fetching script first.")
        return None
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        st.error(f"Error processing data: {e}")
        return None

@st.cache_data
def generate_forecast(_model, _scaler, processed_df_for_forecast: pd.DataFrame):
    """
    Generates the 30-day forecast.
    The '_' prefix for model and scaler indicates they come from a cached resource.
    """
    if processed_df_for_forecast is None or _model is None or _scaler is None:
        return None
    logger.info("Generating forecast...")
    try:
        predictions = forecast_next_days(
            processed_df=processed_df_for_forecast,
            model=_model,
            scaler=_scaler,
            lookback=LOOKBACK_PERIOD,
        )
        logger.info("Forecast generated successfully.")
        return predictions
    except ValueError as ve:
        logger.error(f"ValueError during forecast generation: {ve}")
        st.warning(f"Could not generate forecast: {ve}. Ensure enough historical data is available.")
        return None
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        st.error(f"Error during forecast generation: {e}")
        return None

# --- Main App Logic ---
def main():
    st.set_page_config(page_title="Edersee Water Level Forecast", layout="wide")
    st.title("🌊 Edersee Water Level Forecast")
    st.markdown("Predicting the Edersee water level for the next 30 days using an LSTM model.")

    # 1. Update historical data (runs every time, not cached here intentionally to get fresh data)
    # We use a simple button to trigger an update or do it on load.
    # For automatic update on load, we can just call it.
    # For more control, a button is better. Let's start with automatic on load.
    update_triggered_flag = False
    with st.spinner(f"Updating historical data (fetching last {DAYS_TO_FETCH_API} days from API)..."):
        try:
            logger.info("Attempting to update historical data...")
            update_historical_data(HISTORICAL_DATA_CSV, days=DAYS_TO_FETCH_API)
            logger.info("Historical data update process completed.")
            update_triggered_flag = True # This will be used to bust the get_prepared_data cache
            st.sidebar.success("Data updated successfully!")
        except Exception as e:
            logger.error(f"Failed to update historical data: {e}")
            st.sidebar.error(f"Data update failed: {e}")
            # Proceed with potentially stale data if file exists

    # 2. Load Model and Scaler (cached)
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        st.stop() # Stop execution if model/scaler can't be loaded

    # 3. Load and Preprocess Data (cached, cache busted by update_triggered_flag)
    # Passing update_triggered_flag ensures that if data was updated, this function re-runs.
    processed_df = get_prepared_data(HISTORICAL_DATA_CSV, update_triggered_flag)
    if processed_df is None or processed_df.empty:
        st.warning("No processed data available to display or make predictions.")
        st.stop()

    # 4. Generate Forecast (cached)
    # We pass the dataframe to ensure that if the data changes, forecast re-runs.
    forecast_series = generate_forecast(model, scaler, processed_df)

    # 5. Display Data and Forecast
    st.subheader("Water Level History and Forecast")

    fig = go.Figure()

    # Plot historical data (e.g., last 2 years + forecast period for context)
    # You might want to adjust how much historical data to show
    display_start_date_historical = processed_df.index[-1] - pd.Timedelta(days=365 * 2)
    historical_to_plot = processed_df[processed_df.index >= display_start_date_historical]['value']

    fig.add_trace(
        go.Scatter(
            x=historical_to_plot.index,
            y=historical_to_plot,
            mode="lines",
            name="Historical Water Level",
            line=dict(color="blue"),
        )
    )

    if forecast_series is not None and not forecast_series.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_series.index,
                y=forecast_series,
                mode="lines",
                name="Forecasted Water Level (Next 30 Days)",
                line=dict(color="orange", dash="dash"),
            )
        )
        st.sidebar.metric(
            label="Forecast: Day 1 Level",
            value=f"{forecast_series.iloc[0]:.2f} m",
            delta=f"{(forecast_series.iloc[0] - historical_to_plot.iloc[-1]):.2f} m vs yesterday",
            delta_color="normal" # or "inverse" or "off"
        )
        st.sidebar.metric(
            label=f"Forecast: Day {FORECAST_HORIZON} Level",
            value=f"{forecast_series.iloc[-1]:.2f} m"
        )

    else:
        st.warning("Forecast could not be generated.")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Water Level (m)",
        legend_title_text="Legend",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optionally display recent data table
    st.subheader("Recent Data")
    if not processed_df.empty:
        st.dataframe(processed_df[['value']].tail(10).sort_index(ascending=False))
    
    if forecast_series is not None and not forecast_series.empty:
        st.subheader("Forecasted Values (Next 30 Days)")
        st.dataframe(forecast_series.to_frame(name="Forecasted Water Level"))

if __name__ == "__main__":
    main()