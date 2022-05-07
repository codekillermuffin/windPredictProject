# guide: https://towardsdatascience.com/rnn-recurrent-neural-networks-how-to-successfully-model-sequential-data-in-python-5a0b9e494f92


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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

from matplotlib import pyplot
import plotly.graph_objects as go


# Set Pandas options to display more columns
pd.options.display.max_columns = 50

# Read in the power data csv, only keep specified columns
df = pd.read_csv('TrainData.csv', encoding='utf-8')[['TIMESTAMP', 'POWER']]
df[['DATETIME', 'HOUR']] = df.TIMESTAMP.str.split(" ", expand=True,)

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
X2 = df[['Hour']]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X2_scaled = scaler.fit_transform(X2)

##### Step 2 - Create training and testing samples
train_data, test_data = train_test_split(X_scaled, test_size=0.1, shuffle=False)

##### Step 3 - Prepare input X and target y arrays using previously defined function
time_step = 24
X_train, y_train = prep_data(train_data, time_step)
X_test, y_test = prep_data(test_data, time_step)

##### Step 4 - Specify the structure of a Neural Network
model = Sequential(name="WindPower-RNN-Model")  # Model
model.add(Input(shape=(time_step, 1), name='Input-Layer'))
model.add(SimpleRNN(units=1, activation='tanh',
                    name='Hidden-Recurrent-Layer'))
model.add(Dense(units=1, activation='tanh',
                name='Hidden-Layer'))
model.add(Dense(units=1, activation='linear', name='Output-Layer'))

##### Step 5 - Compile keras model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['MeanSquaredError', 'MeanAbsoluteError'],
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None
              )

##### Step 6 - Fit keras model on the dataset
model.fit(X_train,  # input data
          y_train,  # target data
          batch_size=1,  # Number of samples per gradient update. If unspecified, batch_size will default to 32.
          epochs=20,
          verbose='auto',
          callbacks=None,
          validation_split=0.1,
          validation_data=(X_test, y_test),
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          )

##### Step 7 - Use model to make predictions
# Predict the result on training data
pred_train = model.predict(X_train)
# Predict the result on test data
pred_test = model.predict(X_test)

##### Step 8 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary()  # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
print("Note, the last parameter in each layer is bias while the rest are weights")
print("")
for layer in model.layers:
    print(layer.name)
    for item in layer.get_weights():
        print("  ", item)
print("")
print('---------- Evaluation on Training Data ----------')
print("MSE: ", mean_squared_error(y_train, pred_train))
print("")

print('---------- Evaluation on Test Data ----------')
print("MSE: ", mean_squared_error(y_test, pred_test))
print("")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(range(0, len(y_test))),
                         y=scaler.inverse_transform(y_test).flatten(),
                         mode='lines',
                         name='Power - Actual (Test)',
                         opacity=0.8,
                         line=dict(color='black', width=1)
                         ))
fig.add_trace(go.Scatter(x=np.array(range(0, len(pred_test))),
                         y=scaler.inverse_transform(pred_test).flatten(),
                         mode='lines',
                         name='Power - Predicted (Test)',
                         opacity=0.8,
                         line=dict(color='red', width=1)
                         ))

# Change chart background color
fig.update_layout(dict(plot_bgcolor='white'))

# Update axes lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black',
                 title='Timeline'
                 )

fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                 zeroline=True, zerolinewidth=1, zerolinecolor='lightgrey',
                 showline=True, linewidth=1, linecolor='black',
                 title='Power'
                 )

# Set figure title
fig.update_layout(title=dict(text="Power generation by hour",
                             font=dict(color='black')),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                  )

fig.show()

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
df['POWER_prediction'] = np.append(np.zeros(time_step), scaler.inverse_transform(model.predict(X_comb)))
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
                         y=df['POWER_prediction'],
                         mode='lines',
                         name='POWER - Predicted',
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
