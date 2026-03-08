import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Parameters
TIME_STEP = 50
EPOCHS = 5
BATCH_SIZE = 16

print("Loading data...")
# Load data
data = pd.read_csv('RELIANCE.NS.csv')
prices = data['Open'].values.reshape(-1, 1)

# Fit scaler on prices
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

# Create sequences
X = []
y = []
for i in range(TIME_STEP, len(prices_scaled)):
    X.append(prices_scaled[i - TIME_STEP:i, 0])
    y.append(prices_scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

def build_rnn(input_shape):
    model = Sequential()
    model.add(SimpleRNN(32, activation='tanh', return_sequences=False, input_shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, input_shape=input_shape))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

print("Preparing training data shapes:", X.shape, y.shape)

# Train small RNN
print("Training RNN model (this may take a little while)...")
rnn_model = build_rnn((TIME_STEP, 1))
early = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
rnn_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early], verbose=1)
print("Saving rnn_model.h5...")
rnn_model.save('rnn_model.h5')

# Train small LSTM
print("Training LSTM model...")
lstm_model = build_lstm((TIME_STEP, 1))
lstm_model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early], verbose=1)
print("Saving lstm_model.h5...")
lstm_model.save('lstm_model.h5')

# Save scaler
print("Saving scaler.pkl...")
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("All done. Generated files: rnn_model.h5, lstm_model.h5, scaler.pkl")
