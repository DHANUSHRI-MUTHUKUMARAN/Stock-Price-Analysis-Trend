import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

ticker = "AAPL"

df = pd.read_csv(
    f"data/indicators/{ticker}.csv"
)

data = df["Close"].values.reshape(-1, 1)

# Scale data
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(data)

# Create sequences
sequence_length = 60

X = []
y = []

for i in range(
    sequence_length,
    len(scaled_data)
):
    X.append(
        scaled_data[
            i-sequence_length:i
        ]
    )

    y.append(
        scaled_data[i]
    )

X = np.array(X)
y = np.array(y)

# Split
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Model
model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=True,
        input_shape=(
            X_train.shape[1],
            1
        )
    )
)

model.add(
    Dropout(0.2)
)

model.add(
    LSTM(50)
)

model.add(
    Dropout(0.2)
)

model.add(
    Dense(1)
)

model.compile(
    optimizer="adam",
    loss="mse"
)

print("Training LSTM...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Predictions
predictions = model.predict(
    X_test
)

# Convert back to real prices
predictions = scaler.inverse_transform(
    predictions
)

actual = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

# Metrics
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print("\nRESULTS")
print("-" * 30)

print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot
plt.figure(
    figsize=(12,6)
)

plt.plot(
    actual,
    label="Actual"
)

plt.plot(
    predictions,
    label="Predicted"
)

plt.legend()

plt.title(
    "LSTM Stock Price Forecast"
)

plt.savefig(
    "images/lstm_prediction.png"
)

plt.close()

# Save model
model.save(
    "models/lstm_model.h5"
)

print("\nModel saved.")
print("Chart saved.")