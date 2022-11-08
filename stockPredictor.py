import tensorflow as tf
print("TensorFlow version:", tf.__version__)

import pandas as pd
import numpy as np
from google.colab import files

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams

from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

np.set_printoptions(precision=3, suppress=True)

market_data = pd.read_csv('aapl.us.csv')

print(market_data.head())

#Removing date column 
del market_data['Date']

#================ Pre Processing =================
#Smoothing data
market_data.ewm(alpha=0.65).mean()
close_vals = market_data.loc[7587:]
#Changing features 
"""
  Using exponential moving averages to get a greater weight and significance on the most recent data points.
"""
market_data['ema50'] = market_data['Close'] / market_data['Close'].ewm(50).mean()
market_data['ema21'] = market_data['Close'] / market_data['Close'].ewm(21).mean()
market_data['ema15'] = market_data['Close'] / market_data['Close'].ewm(14).mean()
market_data['ema5'] = market_data['Close'] / market_data['Close'].ewm(5).mean()

#Split into training data set and eval set
train, test = train_test_split(market_data, test_size=0.1)

#Creating Scaler
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
train = pd.DataFrame(train, columns=market_data.columns)

#Adding timesteps
x_train_data,y_train_data=[],[]

for i in range(60,len(train)):
    x_train_data.append(train.iloc[i-60:i,0])
    y_train_data.append(train.iloc[i,3])

#Converting to numpy array
x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

#Applying scaler
y_train_data = y_train_data.reshape(-1,1)
x_train_data = scaler_x.fit_transform(x_train_data)
y_train_data = scaler_y.fit_transform(y_train_data)

#Reshapping to 3D array for LSTM
x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

#================ Building the model =================
model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train_data.shape[1], 1)))
model.add(LSTM(units = 50, return_sequences = True))
model.add(LSTM(units = 50))
model.add(Dense(units = 1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Train model
model.fit(x_train_data, y_train_data, epochs = 100, batch_size = 32)

#====================== Evaluate/predict model ==============
#Processing test data
test = np.array(test)

x_test = []
y_test = []
for i in range(60,test.shape[0]):
    x_test.append(test[i-60:i,0])
    y_test.append(test[i,3])
x_test=np.array(x_test)
y_test=np.array(y_test)

y_test = y_test.reshape(-1,1)
x_test = scaler_x.fit_transform(x_test)
y_test = scaler_y.fit_transform(y_test)

x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

#Evalutation 
eval = model.evaluate(x_test, y_test, batch_size=32, verbose = 0)
print('Validation accuarcy: ', eval)
#NOTE: evaluation accuracy will be somewhat low due to nature of very precise results needing to be acheived

#Saving predictions and plotting them
predictions = model.predict(x_test)
predictions = scaler_y.inverse_transform(predictions)

#Plotting predictions 
close_vals['Predictions'] = predictions
plt.plot(close_vals[['Close', 'Predictions']])