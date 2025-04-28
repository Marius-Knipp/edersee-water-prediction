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
from tensorflow.keras.callbacks import EarlyStopping

# Set style for the plots
plt.style.use('seaborn-v0_8')



# Load and preprocess the data
df = pd.read_csv('./data/processed/water_levels_daily.csv', parse_dates=['timestamp'])
data = (
    df.set_index("timestamp")["value"]
      .resample("W-SUN").mean()
      .interpolate()
)

# Convert series to dataframe
data = data.to_frame()

# Split the data into train, validation, and test sets
train_data = data[:'2017-12-31']
validation_data = data['2018-01-01':'2020-12-31']
test_data = data['2021-01-01':]

print(f"Train data: from {train_data.index.min()} to {train_data.index.max()}")
print(f"Validation data: from {validation_data.index.min()} to {validation_data.index.max()}")
print(f"Test data: from {test_data.index.min()} to {test_data.index.max()}")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
validation_scaled = scaler.transform(validation_data)
test_scaled = scaler.transform(test_data)

# Function to create datasets with lookback and forecast_horizon
def create_dataset_with_horizon(dataset, lookback=52, forecast_horizon=4):
    X, y = [], []
    for i in range(lookback, len(dataset) - forecast_horizon + 1):
        X.append(dataset[i-lookback:i, 0])
        # Target is 4 weeks ahead
        y.append(dataset[i + forecast_horizon - 1, 0])
    return np.array(X), np.array(y)

# Parameters
lookback = 52  # 1 year of weekly data
forecast_horizon = 4  # 4 weeks ahead prediction

# Create train dataset with 4-week ahead target
X_train, y_train = create_dataset_with_horizon(train_scaled, lookback, forecast_horizon)

# For validation, we need to handle the overlap between training and validation periods
# First, prepare a combined array for creating the validation dataset
train_val_combined = np.vstack((train_scaled, validation_scaled))

# We'll create validation samples starting from the beginning of validation period
# but we need lookback data from the training period
val_start_idx = len(train_scaled) - lookback

# Create validation dataset
X_val, y_val = [], []
for i in range(val_start_idx, len(train_scaled) + len(validation_scaled) - forecast_horizon + 1):
    if i >= len(train_scaled):  # Only include samples where target is in validation period
        X_val.append(train_val_combined[i-lookback:i, 0])
        y_val.append(train_val_combined[i + forecast_horizon - 1, 0])

X_val, y_val = np.array(X_val), np.array(y_val)

# Reshape inputs to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (4-Week Ahead Prediction)')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Make predictions for validation period
# For each week in validation, predict its water level 4 weeks ahead
validation_predictions = []
validation_actual = []
prediction_dates = []

# We need to account for the forecast horizon in our predictions
for i in range(len(X_val)):
    # Get input sequence
    input_seq = X_val[i:i+1]
    
    # Make prediction (4 weeks ahead)
    pred = model.predict(input_seq, verbose=0)[0][0]
    validation_predictions.append(pred)
    
    # Store actual value for the corresponding date
    validation_actual.append(y_val[i])
    
    # Calculate the date for this prediction (4 weeks ahead from the last date in input sequence)
    last_date_in_sequence = train_data.index[-1] + pd.Timedelta(days=7 * (i + 1))
    target_date = last_date_in_sequence + pd.Timedelta(days=7 * (forecast_horizon - 1))
    prediction_dates.append(target_date)

# Convert predictions back to original scale
validation_predictions = np.array(validation_predictions).reshape(-1, 1)
validation_actual = np.array(validation_actual).reshape(-1, 1)
validation_predictions = scaler.inverse_transform(validation_predictions)
validation_actual = scaler.inverse_transform(validation_actual)

# Create a DataFrame for validation results
validation_results = pd.DataFrame({
    'Date': prediction_dates,
    'Actual': validation_actual.flatten(),
    'Predicted': validation_predictions.flatten()
})
validation_results.set_index('Date', inplace=True)

# Plot actual vs predicted for validation period
plt.figure(figsize=(15, 8))
plt.plot(validation_results.index, validation_results['Actual'], label='Actual')
plt.plot(validation_results.index, validation_results['Predicted'], label='Predicted (4-weeks ahead)')
plt.title('Edersee Water Level: 4-Week Ahead Prediction (Validation Period)')
plt.xlabel('Date')
plt.ylabel('Water Level')
plt.legend()
plt.grid(True)
plt.show()

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(validation_results['Actual'], validation_results['Predicted']))
mae = mean_absolute_error(validation_results['Actual'], validation_results['Predicted'])

print(f"Validation RMSE (4-week ahead): {rmse:.4f}")
print(f"Validation MAE (4-week ahead): {mae:.4f}")


