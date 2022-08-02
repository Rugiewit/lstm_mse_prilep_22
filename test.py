#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 14:32:45 2022
https://stackoverflow.com/questions/69785891/how-to-use-the-lstm-model-for-multi-step-forecasting

"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# download the data
df = yf.download(tickers=['^IXIC'], period='5y')
y = df['Close'].fillna(method='ffill').values.reshape(- 1, 1)

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# generate the training sequences
n_forecast = 1
n_lookback = 60

X = []
Y = []

for i in range(n_lookback, len(y) - n_forecast + 1):
    X.append(y[i - n_lookback: i])
    Y.append(y[i: i + n_forecast])

X = np.array(X)
Y = np.array(Y)

# train the model
tf.random.set_seed(0)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X, Y, epochs=100, batch_size=128, validation_split=0.2, verbose=0)

# generate the multi-step forecasts
n_future = 365
y_future = []

x_pred = X[-1:, :, :]  # last observed input sequence
y_pred = Y[-1]         # last observed target value

for i in range(n_future):

    # feed the last forecast back to the model as an input
    a =x_pred[:, 1:, :]
    b =  y_pred.reshape(1, 1, 1)
    x_pred = np.append(a,b, axis=1)

    # generate the next forecast
    y_pred = model.predict(x_pred)

    # save the forecast
    y_future.append(y_pred.flatten()[0])

# transform the forecasts back to the original scale
y_future = np.array(y_future).reshape(-1, 1)
y_future = scaler.inverse_transform(y_future)

# organize the results in a data frame
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index': 'Date'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan

df_future = pd.DataFrame(columns=['Date', 'Close', 'Forecast'])
df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_future)
df_future['Forecast'] = y_future.flatten()
df_future['Close'] = np.nan

results = df_past.append(df_future).set_index('Date')

# plot the results
results.plot(title='NASDAQ')
