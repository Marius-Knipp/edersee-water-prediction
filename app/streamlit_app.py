import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging
from datetime import datetime, timedelta
import plotly.graph_objects as go

# ... (rest of your imports and setup code from before) ...
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.fetch_update import update_historical_data
from src.features.preprocessing import load_processed_data, add_calendar_features
from src.models.predict_lstm import load_artifacts, forecast_next_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Edersee Wasserstandsprognose",
    page_icon="💧",
    layout="wide",
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "water_levels_daily.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "water_level_lstm.keras"
SCALER_PATH = PROJECT_ROOT / "models" / "LSTM_standard_scaler.joblib"

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

@st.cache_data(ttl=3600)
def load_data_wrapper():
    try:
        logger.info(f"Attempting to update data at {DATA_PATH}")
        update_historical_data(DATA_PATH, days=15)
        df = load_processed_data(DATA_PATH)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        st.error(f"Kritischer Fehler: Verarbeitete Datendatei nicht gefunden unter {DATA_PATH}. Stellen Sie sicher, dass die initialen Skripte zur Datenbeschaffung ausgeführt wurden.")
        return None
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {str(e)}")
        logger.error(f"Data loading error: {e}", exc_info=True)
        return None

@st.cache_resource
def load_model_and_scaler_wrapper():
    try:
        logger.info(f"Loading model from {MODEL_PATH} and scaler from {SCALER_PATH}")
        model, scaler = load_artifacts(MODEL_PATH, SCALER_PATH)
        logger.info("Model and scaler loaded successfully.")
        return model, scaler
    except FileNotFoundError:
        st.error(f"Kritischer Fehler: Modell- oder Skaliererdatei nicht gefunden. Gesucht unter {MODEL_PATH} und {SCALER_PATH}.")
        return None, None
    except Exception as e:
        st.error(f"Fehler beim Laden von Modell/Skalierer: {str(e)}")
        logger.error(f"Model loading error: {e}", exc_info=True)
        return None, None

def format_change(value, abs_value=False):
    value_str = f"{abs(value) if abs_value else value:.2f} m"
    if value > 0.005: return f'<span class="change-positive">▲ {value_str}</span>'
    elif value < -0.005: return f'<span class="change-negative">▼ {value_str}</span>'
    else: return f'<span class="change-neutral">― {value_str}</span>'


def plot_water_levels_plotly(filtered_historical_data, forecast_data, actual_latest_historical_date):
    fig = go.Figure()
    if not filtered_historical_data.empty:
        fig.add_trace(
            go.Scatter(
                x=filtered_historical_data.index,
                y=filtered_historical_data.values,
                name="Historisch",
                line=dict(color="#1f77b4", width=2), # Solid line by default
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Wasserstand: %{y:.2f} m<extra></extra>'
            )
        )
    if not forecast_data.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast_data.index,
                y=forecast_data.values,
                name="Prognose",
                line=dict(color="#ff7f0e", width=2), # Removed dash='dash' for a solid line
                hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Prognose: %{y:.2f} m<extra></extra>'
            )
        )
    
    y_values = []
    if not filtered_historical_data.empty: y_values.extend(list(filtered_historical_data.values))
    if not forecast_data.empty: y_values.extend(list(forecast_data.values))
    
    if y_values:
        y_min_val, y_max_val = min(y_values), max(y_values)
        y_range_val = y_max_val - y_min_val
        y_plot_min = y_min_val - (y_range_val * 0.05) if y_range_val > 0 else y_min_val - 0.5
        y_plot_max = y_max_val + (y_range_val * 0.05) if y_range_val > 0 else y_max_val + 0.5
    else: y_plot_min, y_plot_max = 0, 1 

    if actual_latest_historical_date:
        plot_latest_date = pd.Timestamp(actual_latest_historical_date)
        current_plot_min_x = pd.Timestamp.max
        if not filtered_historical_data.empty: current_plot_min_x = min(current_plot_min_x, filtered_historical_data.index.min())
        if not forecast_data.empty: current_plot_min_x = min(current_plot_min_x, forecast_data.index.min())
        if current_plot_min_x == pd.Timestamp.max: current_plot_min_x = plot_latest_date - timedelta(days=1)

        current_plot_max_x = pd.Timestamp.min
        if not filtered_historical_data.empty: current_plot_max_x = max(current_plot_max_x, filtered_historical_data.index.max())
        if not forecast_data.empty: current_plot_max_x = max(current_plot_max_x, forecast_data.index.max())
        if current_plot_max_x == pd.Timestamp.min: current_plot_max_x = plot_latest_date + timedelta(days=1)

        if current_plot_min_x <= plot_latest_date <= current_plot_max_x or \
           (not forecast_data.empty and plot_latest_date < forecast_data.index.min() and \
            not filtered_historical_data.empty and plot_latest_date >= filtered_historical_data.index.min()):
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

def main():
    st.markdown('<div class="app-header"><h1>Edersee Wasserstandsprognose</h1></div>', unsafe_allow_html=True)
    
    with st.spinner("Lade historische Daten und Modell..."):
        df_historical_raw = load_data_wrapper() 
        model, scaler = load_model_and_scaler_wrapper()
    
    if df_historical_raw is None or model is None or scaler is None:
        st.warning("Daten oder Modell konnten nicht geladen werden. Die Anwendung kann nicht fortgesetzt werden.")
        return
    if df_historical_raw.empty:
        st.warning("Keine historischen Daten vorhanden. Prognose oder Anzeige nicht möglich.")
        return

    processed_df_for_forecast = add_calendar_features(df_historical_raw.copy())
    forecast_series = pd.Series(dtype='float64')
    try:
        forecast_series = forecast_next_days(processed_df_for_forecast, model, scaler)
    except ValueError as ve:
        st.error(f"Fehler bei der Prognoseerstellung: {str(ve)}. Nicht genügend historische Daten für das Lookback-Fenster vorhanden.")
        logger.error(f"Forecast ValueError: {ve}", exc_info=True)
    except Exception as e:
        st.error(f"Ein unerwarteter Fehler ist bei der Prognoseerstellung aufgetreten: {str(e)}")
        logger.error(f"Forecast error: {e}", exc_info=True)

    st.markdown("## Aktueller Status & Prognostizierte Änderungen")
    col1, col2, col3 = st.columns(3) 
    historical_values_only = df_historical_raw["value"].dropna()

    if not historical_values_only.empty:
        current_level = historical_values_only.iloc[-1]
        current_date_str = historical_values_only.index[-1].strftime("%Y-%m-%d")
        latest_historical_date_ts = historical_values_only.index[-1]

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Aktueller Stand ({current_date_str})</div><div class="metric-value">{current_level:.2f} m</div><div class="metric-label" style="font-size:12px;">Letzte Messung</div></div>', unsafe_allow_html=True)
        
        if not forecast_series.empty:
            target_date_7_days = latest_historical_date_ts + timedelta(days=7)
            predicted_level_7_days = np.nan
            try: predicted_level_7_days = forecast_series.loc[target_date_7_days]
            except KeyError:
                if len(forecast_series) >= 7: predicted_level_7_days = forecast_series.iloc[6]

            if not pd.isna(predicted_level_7_days):
                pred_change_7_days = predicted_level_7_days - current_level
                with col2: 
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 7 Tage</div><div class="metric-value">{format_change(pred_change_7_days)}</div><div class="metric-label" style="font-size:12px;">ggü. aktuellem Stand</div></div>', unsafe_allow_html=True)
            else:
                with col2: st.markdown('<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 7 Tage</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)
        else:
            with col2: st.markdown('<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 7 Tage</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)

        if not forecast_series.empty:
            target_date_30_days = latest_historical_date_ts + timedelta(days=30)
            predicted_level_30_days = np.nan
            try: predicted_level_30_days = forecast_series.loc[target_date_30_days]
            except KeyError:
                if len(forecast_series) >= 30: predicted_level_30_days = forecast_series.iloc[29]
            
            if not pd.isna(predicted_level_30_days):
                pred_change_30_days = predicted_level_30_days - current_level
                with col3: 
                    st.markdown(f'<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 30 Tage</div><div class="metric-value">{format_change(pred_change_30_days)}</div><div class="metric-label" style="font-size:12px;">ggü. aktuellem Stand</div></div>', unsafe_allow_html=True)
            else:
                with col3: st.markdown('<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 30 Tage</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)
        else:
            with col3: st.markdown('<div class="metric-card"><div class="metric-label">Prognostizierte Änderung: Nächste 30 Tage</div><div class="metric-value">N/V</div><div class="metric-label" style="font-size:12px;">Prognose nicht verfügbar</div></div>', unsafe_allow_html=True)
    else:
        st.info("Nicht genügend historische Daten zur Anzeige der aktuellen Messwerte vorhanden.")

    st.markdown("## Visualisierung der Wasserstandsprognose")
    
    plot_config = {'displayModeBar': False} 

    if not historical_values_only.empty:
        actual_min_hist_date = historical_values_only.index.min().date() 
        max_hist_date_val = historical_values_only.index.max().date()
        ten_years_ago_from_max = (pd.Timestamp(max_hist_date_val) - pd.DateOffset(years=10)).date()
        slider_min_date_val = max(ten_years_ago_from_max, actual_min_hist_date)
        six_months_ago_from_max = (pd.Timestamp(max_hist_date_val) - pd.DateOffset(months=6)).date()
        default_display_start_date_val = max(six_months_ago_from_max, slider_min_date_val)

        selected_start_date, selected_end_date = st.slider(
            "Zeitraum für historische Daten auswählen (max. 10 Jahre):",
            min_value=slider_min_date_val,
            max_value=max_hist_date_val,
            value=(default_display_start_date_val, max_hist_date_val),
            format="YYYY-MM-DD", key="date_slider"
        )
        
        display_historical_subset = historical_values_only[
            (historical_values_only.index.normalize() >= pd.to_datetime(selected_start_date)) &
            (historical_values_only.index.normalize() <= pd.to_datetime(selected_end_date))
        ]
        
        fig = plot_water_levels_plotly(display_historical_subset, forecast_series, max_hist_date_val) 
        st.plotly_chart(fig, use_container_width=True, config=plot_config)
    
    else:
        st.info("Keine historischen Daten zum Plotten verfügbar.")
        if not forecast_series.empty:
            st.markdown("Nur Prognosedaten verfügbar:")
            fig_forecast_only = plot_water_levels_plotly(pd.Series(dtype='float64'), forecast_series, None)
            st.plotly_chart(fig_forecast_only, use_container_width=True, config=plot_config)
    
    st.markdown("---")
    st.markdown(f'<div style="text-align: center; color: #666; font-size: 0.9em; padding-top: 20px;"><p>App aktualisiert: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p><p>Datenquelle: PEGELONLINE WSV API | LSTM Prognosemodell</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()