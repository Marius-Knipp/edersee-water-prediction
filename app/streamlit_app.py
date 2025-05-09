import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go

# Ensure project root is correctly identified for module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import your custom modules
from src.data.fetch_update import update_historical_data # This now handles S3
from src.features.preprocessing import add_calendar_features # load_processed_data is not directly needed here anymore
from src.models.predict_lstm import load_artifacts, forecast_next_days

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Edersee Wasserstandsprognose",
    page_icon="💧",
    layout="wide",
)

# Paths to model artifacts (these are relative to the project root inside the container)
MODEL_PATH = PROJECT_ROOT / "models" / "water_level_lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "LSTM_standard_scaler.joblib"

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 10px;
        height: 130px; 
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .metric-value {
        font-size: 28px; 
        font-weight: bold;
    }
    .metric-label {
        font-size: 14px; 
        color: #555;
    }
    .change-positive { color: #28a745; }
    .change-negative { color: #dc3545; }
    .change-neutral { color: #6c757d; }
    .stPlotlyChart {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 5px;
        background-color: white;
    }
    .app-header { text-align: center; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Cached function to load and update data from S3/API
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_data_from_s3_and_api():
    try:
        logger.info("Attempting to update and load data from S3 and API...")
        # update_historical_data now handles S3 read/write and returns the full df
        # Using days=35 to align with fetch_update.py's __main__ default for sufficient overlap
        df = update_historical_data(days=35) 

        if df is None or df.empty:
            st.error("Kritischer Fehler: Keine Daten von S3 oder API geladen.")
            logger.error("update_historical_data returned None or empty DataFrame.")
            return None

        # Ensure 'value' column exists (update_historical_data should provide this)
        if "value" not in df.columns:
             logger.error("DataFrame from update_historical_data is missing 'value' column.")
             st.error("Datenfehler: 'value'-Spalte fehlt nach dem Update-Prozess.")
             return None
        
        # Interpolate missing values. update_historical_data already does resample('D').mean().
        missing_count = df["value"].isna().sum()
        if missing_count > 0:
            logger.info(f"Interpolating {missing_count} missing values in the data.")
            df["value"] = df["value"].interpolate(method="time")
            # Fill any remaining NaNs at the beginning or end
            df["value"] = df["value"].fillna(method='ffill').fillna(method='bfill')

        # Final check for NaNs after interpolation
        if df["value"].isna().any():
            logger.warning("Data still contains NaN values after all interpolation attempts. This might affect forecasting.")
            # Depending on strictness, you might want to st.error here and return None

        logger.info(f"Data loaded and preprocessed successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        st.error(f"Fehler beim Laden und Aktualisieren der Daten: {str(e)}")
        logger.error(f"Data loading/update error: {e}", exc_info=True)
        return None

# Cached function to load model and scaler
@st.cache_resource
def load_model_and_scaler_wrapper():
    try:
        logger.info(f"Loading model from {MODEL_PATH} and scaler from {SCALER_PATH}")
        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Kritischer Fehler: Modell- oder Skaliererdatei nicht gefunden. Gesucht unter {MODEL_PATH} und {SCALER_PATH}.")
        logger.error(f"Model or scaler file not found. Model: {MODEL_PATH}, Scaler: {SCALER_PATH}", exc_info=True)
        return None, None
    except Exception as e:
        st.error(f"Fehler beim Laden von Modell/Skalierer: {str(e)}")
        logger.error(f"Model/scaler loading error: {e}", exc_info=True)
        return None, None

# Helper function to format change values for display
def format_change(value, abs_value=False):
    value_str = f"{abs(value) if abs_value else value:.2f} m"
    if value > 0.005: return f'<span class="change-positive">▲ {value_str}</span>'
    elif value < -0.005: return f'<span class="change-negative">▼ {value_str}</span>'
    else: return f'<span class="change-neutral">― {value_str}</span>'

# Plotting function using Plotly
def plot_water_levels_plotly(filtered_historical_data, forecast_data, actual_latest_historical_date):
    fig = go.Figure()
    if not filtered_historical_data.empty:
        fig.add_trace(
            go.Scatter(
                x=filtered_historical_data.index,
                y=filtered_historical_data.values,
                name="Historisch",
                line=dict(color="#1f77b4", width=2),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Wasserstand: %{y:.2f} m<extra></extra>'
            )
        )
    if not forecast_data.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data.values,
                name="Prognose",
                line=dict(color="#ff7f0e", width=2),
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Prognose: %{y:.2f} m<extra></extra>'
            )
        )
    
    y_values = []
    if not filtered_historical_data.empty: y_values.extend(list(filtered_historical_data.values))
    if not forecast_data.empty: y_values.extend(list(forecast_data.values))
    
    y_plot_min, y_plot_max = 0, 1 # Defaults
    if y_values:
        y_min_val, y_max_val = min(y_values), max(y_values)
        y_range_val = y_max_val - y_min_val
        y_plot_min = y_min_val - (y_range_val * 0.05) if y_range_val > 0 else y_min_val - 0.5
        y_plot_max = y_max_val + (y_range_val * 0.05) if y_range_val > 0 else y_max_val + 0.5
    
    if actual_latest_historical_date:
        plot_latest_date = pd.Timestamp(actual_latest_historical_date)
        # Determine if the vertical line for "latest data point" is within the plot's x-axis range
        current_plot_min_x, current_plot_max_x = pd.Timestamp.max, pd.Timestamp.min
        if not filtered_historical_data.empty:
            current_plot_min_x = min(current_plot_min_x, filtered_historical_data.index.min())
            current_plot_max_x = max(current_plot_max_x, filtered_historical_data.index.max())
        if not forecast_data.empty:
            current_plot_min_x = min(current_plot_min_x, forecast_data.index.min())
            current_plot_max_x = max(current_plot_max_x, forecast_data.index.max())
        
        if current_plot_min_x == pd.Timestamp.max: current_plot_min_x = plot_latest_date - timedelta(days=1)
        if current_plot_max_x == pd.Timestamp.min: current_plot_max_x = plot_latest_date + timedelta(days=1)

        if current_plot_min_x <= plot_latest_date <= current_plot_max_x:
            fig.add_shape(
                type="line", x0=plot_latest_date, y0=y_plot_min, x1=plot_latest_date, y1=y_plot_max,
                line=dict(color="green", width=1.5, dash="dot"), name="Letzter Datenpunkt"
            )
            fig.add_annotation(
                x=plot_latest_date, y=y_plot_max, text="Aktuellster historischer Messwert",
                showarrow=False, yshift=10, font=dict(color="green", size=10), xanchor="center"
            )
    
    fig.update_layout(
        xaxis_title="Datum", 
        yaxis_title="Wasserstand (m)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified", margin=dict(l=40, r=40, t=20, b=40), height=500,
        xaxis_type='date'
    )
    return fig

# Main application logic
def main():
    st.markdown('<div class="app-header"><h1>Edersee Wasserstandsprognose</h1></div>', unsafe_allow_html=True)
    
    with st.spinner("Lade und aktualisiere Daten, lade Modell..."):
        # This now calls the S3-aware data loading function
        df_historical_raw = load_data_from_s3_and_api() 
        model, scaler = load_model_and_scaler_wrapper()
    
    if df_historical_raw is None or model is None or scaler is None:
        st.warning("Daten oder Modell konnten nicht geladen werden. Die Anwendung kann nicht fortgesetzt werden.")
        if df_historical_raw is None: logger.error("df_historical_raw is None after loading attempt.")
        if model is None or scaler is None: logger.error("model or scaler is None after loading attempt.")
        return
    if df_historical_raw.empty:
        st.warning("Keine historischen Daten vorhanden. Prognose oder Anzeige nicht möglich.")
        logger.warning("df_historical_raw is empty after loading attempt.")
        return

    # Ensure 'value' column exists before proceeding (should be handled by load_data_from_s3_and_api)
    if 'value' not in df_historical_raw.columns or df_historical_raw['value'].isna().all():
        st.error("Die geladenen Daten enthalten keine gültigen 'value'-Einträge. Prognose nicht möglich.")
        logger.error("df_historical_raw is missing 'value' column or all values are NaN.")
        return

    # Prepare data for forecasting
    processed_df_for_forecast = add_calendar_features(df_historical_raw.copy())
    
    forecast_series = pd.Series(dtype='float64')
    try:
        # Ensure enough data for lookback period
        lookback_period = 365 # Default, adjust if your model's lookback is different
        if len(processed_df_for_forecast) < lookback_period:
            raise ValueError(f"Nicht genügend historische Daten ({len(processed_df_for_forecast)} Tage). Benötigt werden {lookback_period} Tage für die Prognose.")
        forecast_series = forecast_next_days(processed_df_for_forecast, model, scaler, lookback=lookback_period)
    except ValueError as ve:
        st.error(f"Fehler bei der Prognoseerstellung: {str(ve)}")
        logger.error(f"Forecast ValueError: {ve}", exc_info=True)
    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist bei der Prognoseerstellung aufgetreten: {str(e)}")
        logger.error(f"Forecast error: {e}", exc_info=True)

    st.markdown("## Aktueller Status & Prognostizierte Änderungen")
    col1, col2, col3 = st.columns(3) 
    
    # Use 'value' column directly from df_historical_raw
    historical_values_only = df_historical_raw["value"].dropna()

    if not historical_values_only.empty:
        current_level = historical_values_only.iloc[-1]
        current_date_str = historical_values_only.index[-1].strftime("%Y-%m-%d")
        latest_historical_date_ts = historical_values_only.index[-1]

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Aktueller Stand ({current_date_str})</div><div class="metric-value">{current_level:.2f} m</div><div class="metric-label" style="font-size:12px;">Letzte Messung</div></div>', unsafe_allow_html=True)
        
        def display_forecast_metric(column, days_ahead_label, days_ahead_num):
            with column:
                if not forecast_series.empty:
                    target_date = latest_historical_date_ts + timedelta(days=days_ahead_num)
                    predicted_level = np.nan
                    try: 
                        predicted_level = forecast_series.loc[target_date]
                    except KeyError: # If exact date not in index, try iloc
                        if len(forecast_series) >= days_ahead_num: 
                            predicted_level = forecast_series.iloc[days_ahead_num-1] # 0-indexed

                    if not pd.isna(predicted_level):
                        pred_change = predicted_level - current_level
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: {days_ahead_label}</div><div class="metric-value">{format_change(pred_change)}</div><div class="metric-label" style="font-size:12px;">ggü. aktuellem Stand</div></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: {days_ahead_label}</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: {days_ahead_label}</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)

        display_forecast_metric(col2, "Nächste 7 Tage", 7)
        display_forecast_metric(col3, "Nächste 30 Tage", 30)
    else:
        st.info("Nicht genügend historische Daten zur Anzeige der aktuellen Messwerte vorhanden.")
        logger.info("historical_values_only is empty, cannot display current metrics.")

    st.markdown("## Visualisierung der Wasserstandsprognose")
    plot_config = {'displayModeBar': False}

    if not historical_values_only.empty:
        actual_min_hist_date = historical_values_only.index.min().date() 
        max_hist_date_val = historical_values_only.index.max().date()
        
        # Slider range: max 10 years back from latest data, but not before actual earliest data
        ten_years_ago_from_max = (pd.Timestamp(max_hist_date_val) - pd.DateOffset(years=10)).date()
        slider_min_date_val = max(ten_years_ago_from_max, actual_min_hist_date)
        
        # Default display range: last 6 months, but not before slider_min_date_val
        six_months_ago_from_max = (pd.Timestamp(max_hist_date_val) - pd.DateOffset(months=6)).date()
        default_display_start_date_val = max(six_months_ago_from_max, slider_min_date_val)

        selected_start_date, selected_end_date = st.slider(
            "Zeitraum für historische Daten auswählen (max. 10 Jahre zurückliegend):",
            min_value=slider_min_date_val,
            max_value=max_hist_date_val, # Slider ends at the last historical data point
            value=(default_display_start_date_val, max_hist_date_val),
            format="DD.MM.YYYY", key="date_slider"
        )
        
        display_historical_subset = historical_values_only[
            (historical_values_only.index.normalize() >= pd.to_datetime(selected_start_date)) &
            (historical_values_only.index.normalize() <= pd.to_datetime(selected_end_date))
        ]
        
        fig = plot_water_levels_plotly(display_historical_subset, forecast_series, max_hist_date_val) 
        st.plotly_chart(fig, use_container_width=True, config=plot_config)
    
    else: # Only if historical_values_only is empty
        st.info("Keine historischen Daten zum Plotten verfügbar.")
        if not forecast_series.empty:
            st.markdown("Nur Prognosedaten verfügbar:")
            fig_forecast_only = plot_water_levels_plotly(pd.Series(dtype='float64'), forecast_series, None)
            st.plotly_chart(fig_forecast_only, use_container_width=True, config=plot_config)
    
    st.markdown("---")
    st.markdown(f'<div style="text-align: center; color: #666; font-size: 0.9em; padding-top: 20px;"><p>App aktualisiert: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p><p>Datenquelle: PEGELONLINE WSV API | LSTM Prognosemodell</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()