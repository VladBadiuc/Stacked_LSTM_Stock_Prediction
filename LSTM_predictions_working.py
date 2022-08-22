#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:04:28 2022

@author: vladbad
"""

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

# DATA DOWNLOAD
ticker = "PLTR"  #"ITX.MC"
df = yf.download(tickers=[ticker], period='2y', interval='1d')
y = df['Close'].fillna(method='ffill')
y = y.values.reshape(-1,1)

# DATA SCALING
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(y)
y = scaler.transform(y)

# Generate input and output Sequences
n_lookback = 90 #90
n_forecast = 30 #30

X = []
Y = []

for i in range(n_lookback, len(y)-n_forecast+1):
    X.append(y[i-n_lookback : i])
    Y.append(y[i : i+n_forecast])
    
X = np.array(X)
Y = np.array(Y)

# FIT MODEL
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(n_forecast))

# model.add(LSTM(150,return_sequences=True,input_shape=(n_lookback,1)))
# model.add(Dropout(0.2))
# model.add(LSTM(150,return_sequences=True))
# model.add(LSTM(150, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(150))
# model.add(Dropout(0.2))
# model.add(Dense(n_forecast))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=50, batch_size=32, verbose=0)

# FORECAST GENERATION
X_ = y[-n_lookback:]
X_ = X_.reshape(1,n_lookback,1)

Y_ = model.predict(X_).reshape(-1,1)
Y_ = scaler.inverse_transform(Y_)

# RESULTS ORGANIZATION
df_past = df[['Close']].reset_index()
df_past.rename(columns={'index':'Date','Close':'Actual'}, inplace=True)
df_past['Date'] = pd.to_datetime(df_past['Date'])
df_past['Forecast'] = np.nan
df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
df_future['Date'] = pd.date_range(start = df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
df_future['Forecast'] = Y_.flatten()
df_future['Actual'] = np.nan

results = df_past.append(df_future).set_index('Date')
results.plot(title=ticker, grid=True)


