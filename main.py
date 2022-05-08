# guide: https://towardsdatascience.com/rnn-recurrent-neural-networks-how-to-successfully-model-sequential-data-in-python-5a0b9e494f92


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential  # for creating a linear stack of layers for our Neural Network
from keras import Input  # for instantiating a keras tensor
from keras.layers import Dense, SimpleRNN  # for creating regular densely-connected NN layers and RNN layers

# Data manipulation
import pandas as pd
from pandas import read_csv
import numpy as np
import math

import sklearn  # for model evaluation
from sklearn.model_selection import train_test_split  # for splitting the data into train and test samples
from sklearn.metrics import mean_squared_error  # for model evaluation metrics
from sklearn.preprocessing import MinMaxScaler  # for feature scaling

from matplotlib import pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

# Set Pandas options to display more columns
pd.options.display.max_columns = 50

# Read in the power data csv, only keep specified columns
df = pd.read_csv('TrainData.csv', encoding='utf-8')[['TIMESTAMP', 'POWER']]
nov = pd.read_csv('WeatherForecastInput.csv', encoding='utf-8')
combi = pd.concat([df, nov], axis=1)

df[['DATETIME', 'HOURSTRING']] = df.TIMESTAMP.str.split(" ", expand=True)
df[['HOUR', 'MINUTE']] = df.HOURSTRING.str.split(":", expand=True)
# Show a snaphsot of data
print(df)


##### Step 0 - We will use this function in step 3 to get the data into the right shape
def prep_data(datain, time_step):
    # 1. y-array
    # First, create an array with indices for y elements based on the chosen time_step
    y_indices = np.arange(start=time_step, stop=len(datain), step=time_step)
    # Create y array based on the above indices
    y_tmp = datain[y_indices]

    # 2. X-array
    # We want to have the same number of rows for X as we do for y
    rows_X = len(y_tmp)
    # Since the last element in y_tmp may not be the last element of the datain,
    # let's ensure that X array stops with the last y
    X_tmp = datain[range(time_step * rows_X)]
    # Now take this array and reshape it into the desired shape
    X_tmp = np.reshape(X_tmp, (rows_X, time_step, 1))
    return X_tmp, y_tmp


##### Step 1 - Select data for modeling and apply MinMax scaling
X = df[['POWER']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

##### Step 2 - Create training and testing samples
train_data, test_data = train_test_split(X_scaled, test_size=0.1, shuffle=False)

##### Step 3 - Prepare input X and target y arrays using previously defined function
time_step = 4
X_train, y_train = prep_data(train_data, time_step)
X_test, y_test = prep_data(test_data, time_step)

##### Step 4.1 - Specify the structure of a Neural Network
model = Sequential(name="WindPower-RNN-Model")  # Model
model.add(Input(shape=(time_step, 1), name='Input-Layer'))
model.add(SimpleRNN(units=1, activation='tanh', name='Hidden-Recurrent-Layer'))
model.add(Dense(units=1, activation='tanh', name='Hidden-Layer'))
model.add(Dense(units=1, activation='linear', name='Output-Layer'))

##### Step 4.2 - Specify the structure of a Neural Network
ann_model = Sequential(name="WindPower-ANN-Model")  # Model
ann_model.add(Input(shape=(time_step, 1), name='Input-Layer'))
ann_model.add(Dense(units=3, activation='relu', name='Hidden-Layer1'))
ann_model.add(Dense(units=2, activation='relu', name='Hidden-Layer2'))
ann_model.add(Dense(units=1, activation='sigmoid', name='Output-Layer'))

##### Step 5 - Compile keras model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['MeanSquaredError', 'MeanAbsoluteError', tf.keras.metrics.RootMeanSquaredError()],
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None
              )

ann_model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['MeanSquaredError', 'MeanAbsoluteError', tf.keras.metrics.RootMeanSquaredError()],
                  loss_weights=None,
                  weighted_metrics=None,
                  run_eagerly=None,
                  steps_per_execution=None)

##### Step 6 - Fit keras model on the dataset
model.fit(X_train,  # input data
          y_train,  # target data
          batch_size=1,  # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=10,
          verbose='auto',
          callbacks=None,
          validation_split=0.2,
          validation_data=(X_test, y_test),
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=2,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          )

ann_model.fit(X_train,  # input data
              y_train,  # target data
              batch_size=1,  # Number of samples per gradient update. If unspecified, batch_size will default to 32.
              epochs=10,
              verbose='auto',
              callbacks=None,
              validation_split=0.2,
              validation_data=(X_test, y_test),
              shuffle=True,
              class_weight=None,
              sample_weight=None,
              initial_epoch=0,
              steps_per_epoch=None,
              validation_steps=None,
              validation_batch_size=None,
              validation_freq=2,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False,
              )

##### Step 8 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary()  # print model summary
print("")

print('-------------------- Model2 Summary --------------------')
ann_model.summary()  # print model summary
print("")

# With the current setup, we feed in 7 days worth of data and get the prediction for the next day
# We want to create an array that contains 7-day chunks offset by one day at a time
# This is so we can make a prediction for every day in the data instead of every 7th day
X_every = df[['POWER']]
X_every = scaler.transform(X_every)

for i in range(0, len(X_every) - time_step):
    if i == 0:
        X_comb = X_every[i:i + time_step]
    else:
        X_comb = np.append(X_comb, X_every[i:i + time_step])
X_comb = np.reshape(X_comb, (math.floor(len(X_comb) / time_step), time_step, 1))
print(X_comb.shape)

# Use the reshaped data to make predictions and add back into the dataframe
# np.zeros(time_step) - Set the first 7 numbers to 0 as we do not have data to predict

nov_pred = []
df[['POWER_RNN_PRED']] = np.append(np.zeros(time_step), scaler.inverse_transform(model.predict(X_comb)))

df.to_csv('ForecastTemplate3-RNN.csv', encoding='utf-8', index=False)

print(df)
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['TIMESTAMP'],
                         y=df['POWER'],
                         mode='lines',
                         name='POWER - Actual',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                         ))
fig.add_trace(go.Scatter(x=df['TIMESTAMP'],
                         y=df['POWER_RNN_PRED'],
                         mode='lines',
                         name='POWER - RNN Predicted',
                         opacity=0.8,
                         line=dict(color='red', width=1)
                         ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor='white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black',
                 title='Time'
                 )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black',
                 title='POWER'
                 )

# Set figure title
fig.update_layout(title=dict(text="Power by timeslot",
                             font=dict(color='black')),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                  )

fig.show()
