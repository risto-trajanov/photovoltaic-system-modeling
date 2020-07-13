#!/usr/bin/env python
# coding: utf-8

# In[5]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import TimeDistributed, BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# import keras
# from keras import backend as K
import tensorflow as tf
#import keras_metrics as km
import os
from numpy import array
from utils import read_data
#from utils import upload_s3

# from airflow.hooks.S3_hook import S3Hook
# from airflow.hooks.postgres_hook import PostgresHook
# from sqlalchemy import create_engine
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ## Univariate LSTM
# 

# In[3]:


def split_sequence(sequence, n_steps=3):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_model():
    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


# In[14]:
# def get_connection():
#     hook = PostgresHook()
#     conn = hook.get_connection()
#     SQLALCHEMY_DATABASE_URI = 'postgresql://{db_user}:{db_password}@{db_host}:5432/{db_name}'.format(
#                 db_host = '34.69.215.94',
#                 db_name = 'snpiao_data',
#                 db_password = 'ucamnafinki',
#                 db_user = 'snpiao_data',
#     )
#
#     engine = create_engine(SQLALCHEMY_DATABASE_URI)
#     return engine.connect()
#
#
# def read_data(table):
#     connection = get_connection()
#     data = pd.read_sql_table(table, connection)
#     data = data.set_index('datetime')
#     connection.close()
#     return data
table = 'tsdb_cams_mera_cleaned'
df = read_data(table)
#df = pd.read_csv('clean_targets.csv')
print(df)
seq = np.array(df['consumption'])
print(seq)

# choose a number of time steps
# 24 steps for 24 hours (1day)
n_steps = 24
# choose a number of features
n_features = 1
# split into samples
X, y = split_sequence(seq, n_steps)

def train_save_uni():
    table = 'tsdb_cams_mera_cleaned'
    df = read_data(table)
    #df = pd.read_csv('clean_targets.csv')
    print(df)
    seq = np.array(df['consumption'])
    print(seq)

    # choose a number of time steps
    # 24 steps for 24 hours (1day)
    n_steps = 24
    # choose a number of features
    n_features = 1
    # split into samples
    X, y = split_sequence(seq, n_steps)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # train the model
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    model = create_model()
    model.fit(X, y, epochs=200, verbose=1, validation_split=0.25)

    # make a prediction
    x_input = seq[-n_steps:]
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print(f'Утре очекуваме да имаме: {round(int(yhat))}W/h потрошувачка!')

    model.save("univariate.h5")
    print("Saved model to disk")


# In[18]:


# plot the model
# tf.keras.utils.plot_model(
#     model,
#     to_file="model.png",
#     show_shapes=False,
#     show_layer_names=True,
#     rankdir="TB",
#     #expand_nested=False,
#     #dpi=96,
# )


# ### Multi-step model for predicting several days ahead

# In[26]:


# multi-step data preparation

 
# split a univariate sequence into samples
def split_multi_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def create_multi_step_model(n_steps_in, n_steps_out):
    # define model
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(100, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    return model


# In[31]:

def model_uni():
# choose a number of time steps
    n_steps_in, n_steps_out = 24, 2
    # split into samples
    X, y = split_multi_sequence(seq, n_steps_in, n_steps_out)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model = create_multi_step_model(n_steps_in, n_steps_out)
    # fit model
    history = model.fit(X, y, epochs=50, verbose=1, validation_split=0.25)


    # In[34]:


    # demonstrate prediction
    x_input = seq[-n_steps:]
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    arr = yhat.flatten()
    utre = round(int(arr[0]))
    zadutre = round(int(arr[1]))
    print(f'Утре очекуваме да имаме: {utre} W/h потрошувачка! А задутре: {zadutre} W/h потрошувачка! ')



    plt.subplot()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()


    # In[71]:


    prediction_sequence = df.iloc[::24,:]
    prediction_sequence
    predictions = []

    predictions = model.predict(X)

    for prediction in prediction_sequence:
        x_input = np.array(df['consumption'])[-24:]
        predictions.append(model.predict(x_input))

    print(predictions)
    #df['predictions'] = predictions
    # _ = df[['consumption','predictions']].plot(figsize=(15, 5))
    # import matplotlib.pyplot as plt

    # plt.subplot()
    # plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.legend()
    # plt.show()


    # In[104]:


    predictions = model.predict(X)
    print(len(predictions))
    consumptions = df['consumption']
    print(len(consumptions))
    # df['prediction'] = predictions
    # df
    # shift train predictions for plotting
    # trainPredictPlot = np.empty_like(df)
    # trainPredictPlot[:, :] = np.nan
    trainPredictPlot = []
    trainPredictPlot[n_steps:len(consumptions)+n_steps] = consumptions[:-24]
    # shift test predictions for plotting
    # testPredictPlot = np.empty_like(df)
    # testPredictPlot[:, :] = np.nan
    testPredictPlot = []
    testPredictPlot[len(predictions)+(n_steps*2)+1:len(df)-1] = predictions
    # plot baseline and predictions
    #plt.plot(scaler.inverse_transform(df))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


    # In[121]:


    # Plot the forecast with the actuals

    # report = pd.DataFrame(np.array([df['datetime'], df['consumption'], testPredictPlot]),
    #                    columns=['datetime', 'consumption', 'prediction'])
    # report
    #d = {'predictions': testPredictPlot}
    predict_set = df[:1450]
    predict_set['predictions'] = testPredictPlot[0]
    #predictions = pd.DataFrame(data=d, dtype=np.float, index = df['datetime'])
    report = pd.concat([df, predict_set], sort=False)
    f, ax = plt.subplots(1)
    f.set_figheight(5)
    f.set_figwidth(15)
    _ = report[['consumption','predictions']].plot(ax=ax,
                                                  style=['-','-'])
    ax.set_xbound(lower='02-13-2020', upper='03-03-2020')
    ax.set_ylim(-100, 1000)
    plot = plt.suptitle('Forecast vs Actuals')

    model.save("univariate.h5")
    print("Saved model to disk")

   # upload_s3('multivariate.h5')


# load and evaluate a saved model
# from numpy import loadtxt
# from keras.models import load_model
#
# # load model
# model = load_model('model.h5')
# # summarize model.
# model.summary()
# # load dataset
# dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:, 0:8]
# Y = dataset[:, 8]
# # evaluate the model
# score = model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
# In[ ]:




