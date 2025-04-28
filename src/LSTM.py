import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Set style for the plots
plt.style.use('seaborn-v0_8')



df = pd.read_csv('./data/processed/water_levels_daily.csv', parse_dates=['timestamp'])
weekly = (
    df.set_index("timestamp")["value"]
      .resample("W-SUN").mean()
      .interpolate()
)


periods = [52]
t = np.arange(len(weekly))
fourier_terms = pd.DataFrame(index=weekly.index)
for P in periods:
    fourier_terms[f"sin_{P}w"] = np.sin(2 * np.pi * t / P)
    fourier_terms[f"cos_{P}w"] = np.cos(2 * np.pi * t / P)
    
    
anchor_template  = [
    ("11-01", 237.30), ("12-15", 237.30), ("01-01", 239.61),
    ("02-01", 241.71), ("03-01", 242.70), ("04-01", 244.10),
    ("05-01", 244.95),
]

def compute_max_level_series(dates):
    def max_level_on(date):
        y0 = date.year if date.month >= 11 else date.year - 1
        pts = []
        for md, lvl in anchor_template:
            mon, day = map(int, md.split("-"))
            year = y0 if mon >= 11 else y0 + 1
            # Handle potential errors creating timestamps (e.g., Feb 29)
            try:
                pts.append((pd.Timestamp(year, mon, day), lvl))
            except ValueError: # Skip invalid dates like Feb 29 in non-leap years if needed
                 if mon == 2 and day == 29:
                     pts.append((pd.Timestamp(year, 2, 28), lvl)) # Use Feb 28 instead
                 else:
                     raise # Re-raise other errors
        pts.sort(key=lambda x: x[0])
        if date < pts[0][0] or date >= pts[-1][0]:
            return pts[-1][1]
        for (d0, l0), (d1, l1) in zip(pts, pts[1:]):
            if d0 <= date <= d1:
                frac = (date - d0) / (d1 - d0)
                return l0 + frac * (l1 - l0)
        return pts[-1][1] # Fallback

    return pd.Series([max_level_on(d) for d in dates], index=dates, name="max_level_m")

max_level = compute_max_level_series(weekly.index)

# Combine features
data_full = pd.concat([weekly.rename('value'), fourier_terms], axis=1)


validation_start_date = '2018-01-01'
validation_end_date = '2020-12-31'

train_data = data_full[:validation_start_date]
valid_data = data_full[validation_start_date:validation_end_date]
test_data = data_full[validation_end_date:]

# Also get the validation/test target values and max_levels for later evaluation/capping
actual_valid_values = weekly[validation_start_date:]
max_level_valid = max_level[validation_start_date:]
actual_test_values = weekly[validation_end_date:]
max_level_test = max_level[validation_end_date:]

# Print time ranges of the datasets
print("-" * 20)
print(f"Train data time range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Validation data time range: {valid_data.index.min()} to {valid_data.index.max()}")
print(f"Test data time range: {test_data.index.min()} to {test_data.index.max()}")

# Print the sizes of the datasets
print("-" * 20)
print(f"Train data size: {len(train_data) / len(data_full):.2%}")
print(f"Validation data size: {len(valid_data) / len(data_full):.2%}")
print(f"Test data size: {len(test_data) / len(data_full):.2%}")



# Scaler for input features (X) - includes 'value' column for lagged inputs
scaler_X = MinMaxScaler()
# Fit on the entire training data (all columns that will form the input sequences)
scaler_X.fit(train_data)

# Scaler for the target variable (y) - only the 'value' column
scaler_y = MinMaxScaler()
# # Fit requires a 2D array, so reshape the 'value' column
scaler_y.fit(train_data[['value']]) # Use double brackets to keep it as DataFrame -> 2D

# Transform the data using the appropriate scalers
train_scaled = scaler_X.transform(train_data)
valid_scaled = scaler_X.transform(valid_data)

# Convert back to DataFrames for creating sequences
# Use the original columns for clarity
train_scaled_df = pd.DataFrame(train_scaled, index=train_data.index, columns=train_data.columns)
valid_scaled_df = pd.DataFrame(valid_scaled, index=valid_data.index, columns=valid_data.columns)



def create_sequences_separate(input_data_scaled_X, # Data scaled by scaler_X
                             target_data_original, # Original target data Series
                             scaler_y_obj,       # The fitted scaler_y
                             n_steps_in,
                             n_steps_out):
    X, y = [], []
    # Scale the target data separately using scaler_y
    target_scaled_y = scaler_y_obj.transform(
        target_data_original.to_frame()     
    ).flatten()

    input_values_X = input_data_scaled_X.values # Use the X-scaled data for input features

    # Create sequences
    for i in range(len(input_values_X) - n_steps_in - n_steps_out + 1):
        seq_in = input_values_X[i : i + n_steps_in]
         # Use the y-scaled data for the target sequence
        seq_out = target_scaled_y[i + n_steps_in : i + n_steps_in + n_steps_out]
        X.append(seq_in)
        y.append(seq_out)
    return np.array(X), np.array(y)

# Define sequence parameters (same as before)
N_STEPS_IN = 52
N_STEPS_OUT = 4
N_FEATURES = train_scaled_df.shape[1] # Still 3 features in the input

# Create sequences for training: Use scaled training data for X, original training target for y scaling
X_train, y_train = create_sequences_separate(train_scaled_df, train_data['value'], scaler_y, N_STEPS_IN, N_STEPS_OUT)

# Create sequences for validation: Use scaled validation data for X, original validation target for y scaling
# We need the original 'value' from valid_data for y scaling
X_valid, y_valid = create_sequences_separate(valid_scaled_df, valid_data['value'], scaler_y, N_STEPS_IN, N_STEPS_OUT)


# Align actual validation values (same logic as before)
first_pred_target_idx_valid = N_STEPS_IN + N_STEPS_OUT -1
actual_valid_sequences_target = actual_valid_values[first_pred_target_idx_valid : first_pred_target_idx_valid + len(y_valid)]
max_level_valid_sequences = max_level_valid[first_pred_target_idx_valid : first_pred_target_idx_valid + len(y_valid)]


print("-" * 20)
print("Number of features in X:", N_FEATURES)
print("Training X shape:", X_train.shape)
print("Training y shape:", y_train.shape) # y is now scaled by scaler_y
print("Validation X shape:", X_valid.shape)
print("Validation y shape:", y_valid.shape)

# Ensure alignment (Check dates)
first_pred_target_date_in_valid = valid_data.index[first_pred_target_idx_valid]
print("-" * 20)
print("First target date for validation prediction:", first_pred_target_date_in_valid)
print("First date in aligned actuals:", actual_valid_sequences_target.index[0])

assert first_pred_target_date_in_valid == actual_valid_sequences_target.index[0]
assert len(y_valid) == len(actual_valid_sequences_target)


# Define the LSTM model architecture
model = Sequential([
    LSTM(units=50, activation='relu', input_shape=(N_STEPS_IN, N_FEATURES), return_sequences=True),
    Dropout(0.15), 
    LSTM(units=50, activation='relu'), 
    Dense(units=N_STEPS_OUT) # Output layer predicts N_STEPS_OUT values
])

# Compile the model
model.compile(optimizer='adam', loss='mse') # Mean Squared Error is common for regression


# Train the model

early = keras.callbacks.EarlyStopping(
    patience=500, restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,  # Start with a reasonable number, maybe use EarlyStopping later
    batch_size=32,
    validation_data=(X_valid, y_valid),
    callbacks=[early],
    shuffle=False,  # Important for time series data
    verbose=1
)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()



# Predict on validation data - Shape: (n_samples, n_steps_out)
predictions_scaled = model.predict(X_valid)

# We still want the prediction for the 4th week ahead (index 3)
predictions_scaled_h4 = predictions_scaled[:, N_STEPS_OUT - 1] # Shape: (n_samples,)

# Inverse transform using scaler_y - requires 2D input
# Reshape predictions_scaled_h4 to (n_samples, 1)
predictions_raw = scaler_y.inverse_transform(predictions_scaled_h4.reshape(-1, 1))

# Flatten the result back to 1D if needed, or keep as column vector
predictions_raw = predictions_raw.flatten() # Shape: (n_samples,)

# Create a Pandas Series for easier handling and alignment
predictions_raw_series = pd.Series(predictions_raw, index=actual_valid_sequences_target.index, name='LSTM_Raw')

# --- Capping ---
MIN_VAL = 205
predictions_capped = np.maximum(MIN_VAL, predictions_raw_series)
predictions_final = np.minimum(predictions_capped, max_level_valid_sequences)
predictions_final = pd.Series(predictions_final, index=actual_valid_sequences_target.index, name='LSTM_Final')



mae = mean_absolute_error(actual_valid_sequences_target, predictions_final)
rmse = np.sqrt(mean_squared_error(actual_valid_sequences_target, predictions_final))

print(f"Validation MAE (LSTM, h=4, capped): {mae:.4f}")
print(f"Validation RMSE (LSTM, h=4, capped): {rmse:.4f}")

# Optional: Plot actual vs predicted
plt.figure(figsize=(15, 7))
plt.plot(actual_valid_sequences_target, label='Actual', alpha=0.7)
plt.plot(predictions_final, label='LSTM Forecast (h=4, Capped)', alpha=0.7)
# plt.plot(predictions_raw_series, label='LSTM Raw', alpha=0.5, linestyle='--') # Optional: plot raw forecast
plt.plot(max_level_valid_sequences, label='Max Level Cap', color='red', linestyle=':', alpha=0.6) # Plot the cap
plt.title('Edersee Water Level: Actual vs LSTM Forecast (4 Weeks Ahead)')
plt.xlabel('Date')
plt.ylabel('Water Level (m)')
plt.legend()
plt.grid(True)
plt.show()