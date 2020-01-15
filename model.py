import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler
import warnings 
import tensorflow as tf
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

df = pd.read_csv('crypto_data/BCH-USD.csv', names=['time','low','high','open','close','volume'])
df = np.array(df[['close']])

scaler = MinMaxScaler((-1,1))
scaler.fit(df)
df = scaler.transform(df)

#just keep in mind that there probably exists a better way to preprocess the data,
#but idgaf
while True:
	try:
		df = np.reshape(df,(-1,20,1))
		break
	except:
		df = df[:-1]

y = []
for r in range(df.shape[0]):
	try:
		y.append([float(df[r+1][0]),float(df[r+1][1]),float(df[r+1][2])])
	except:
		df = df[:-1]
		break


df = np.array(df)
y = np.array(y)

x_train, x_test = model_selection.train_test_split(df,test_size=0.2,train_size=0.8)
y_train, y_test = model_selection.train_test_split(y,test_size=0.2,train_size=0.8)

model = tf.keras.Sequential()
model.add(LSTM(128, input_shape=(20,1), activation='relu',return_sequences = True))
model.add(LSTM(128, activation='relu'))

model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation="softmax"))

opt = tf.keras.optimizers.Adam(lr=0.03)
loss = tf.keras.losses.MSE
model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=1000)

#lol ended with 0.359 acc






