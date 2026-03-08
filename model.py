import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Dropout, LSTM
import warnings
import os
warnings.filterwarnings("ignore")
print("Starting model training...")

# Load data
data = pd.read_csv('RELIANCE.NS.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Split data
length_data = len(data)
split_ratio = 0.7
length_train = round(length_data * split_ratio)
length_validation = length_data - length_train

train_data = data[:length_train][['Date', 'Open']]
validation_data = data[length_train:][['Date', 'Open']]

# Prepare dataset
dataset_train = train_data.Open.values.reshape(-1, 1)

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_train_scaled = scaler.fit_transform(dataset_train)

# Create X_train and y_train
X_train = []
y_train = []
time_step = 50

for i in range(time_step, length_train):
    X_train.append(dataset_train_scaled[i-time_step:i, 0])
    y_train.append(dataset_train_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = np.reshape(y_train, (y_train.shape[0], 1))

# RNN Model
regressor = Sequential()
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(SimpleRNN(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
regressor.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

# LSTM Model
model_lstm = Sequential()
model_lstm.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_lstm.add(LSTM(64, return_sequences=False))
model_lstm.add(Dense(32))
model_lstm.add(Dense(1))
model_lstm.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model_lstm.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)

# Save models
try:
    regressor.save('rnn_model.h5')
    print("RNN model saved successfully")
except Exception as e:
    print("Error saving RNN model:", e)

try:
    model_lstm.save('lstm_model.h5')
    print("LSTM model saved successfully")
except Exception as e:
    print("Error saving LSTM model:", e)

# Prepare for prediction
dataset_validation = validation_data.Open.values.reshape(-1, 1)
scaled_dataset_validation = scaler.transform(dataset_validation)

X_test = []
y_test = []
for i in range(time_step, length_validation):
    X_test.append(scaled_dataset_validation[i-time_step:i, 0])
    y_test.append(scaled_dataset_validation[i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Save scaler
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models trained and saved.")
print("RNN model saved:", 'rnn_model.h5' in os.listdir('.'))
print("LSTM model saved:", 'lstm_model.h5' in os.listdir('.'))
print("Scaler saved:", 'scaler.pkl' in os.listdir('.'))
