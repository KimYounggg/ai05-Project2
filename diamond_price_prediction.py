# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 16:02:11 2022

@author: Kim Young
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

#Read csv file
diamonds_data = pd.read_csv(r"C:\Users\Kim Young\Desktop\SHRDC\Deep Learning\TensorFlow Deep Learning\Datasets\diamonds.csv")

#%%
#Drop columns that are not useful
diamonds_data = diamonds_data.drop('Unnamed: 0', axis = 1)

#Split data into features and labels
diamond_features = diamonds_data.copy()
diamond_labels = diamond_features.pop('price')

#%%
#Ordinal Encoder to encode categorical features
cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
color_categories = ['J','I','H','G','F','E','D']
clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

encoder = OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories])
diamond_features[['cut', 'color', 'clarity']] = encoder.fit_transform(diamond_features[['cut', 'color', 'clarity']])

#%%
#Split data into train-validation-test sets
x_inter, x_eval, y_inter, y_eval = train_test_split(diamond_features, 
                                                    diamond_labels, 
                                                    test_size = 0.2, 
                                                    random_state = 12345)

x_train, x_test, y_train, y_test = train_test_split(x_inter, 
                                                    y_inter, 
                                                    test_size = 0.4, 
                                                    random_state = 12345)

#%%
#Normalize the features
ss = StandardScaler()
ss.fit(x_train)

x_eval = ss.transform(x_eval)
x_train = ss.transform(x_train)
x_test = ss.transform(x_test)

#%%
from tensorflow.keras.layers import InputLayer, Dense, Dropout

#Create feedforward neural network
inputs = x_train.shape[-1]

model = tf.keras.Sequential()
model.add(InputLayer(input_shape = inputs))
model.add(Dense(128, activation = 'elu'))
model.add(Dense(64, activation = 'elu'))
model.add(Dense(32, activation = 'elu'))
model.add(Dropout(0.3))
model.add(Dense(1))

#Compile the model
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'mse'])

#%%
#Train model
log_path = r'C:\\Users\\Kim Young\\Desktop\\SHRDC\\Deep Learning\\TensorFlow Deep Learning\\Tensorboard\\logs'
tb = tf.keras.callbacks.TensorBoard(log_dir = log_path)
es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 2)

history = model.fit(x_train, y_train, validation_data = (x_eval, y_eval), batch_size = 64,epochs = 100, callbacks = [tb, es])

#%%
#Evaluate model
result = model.evaluate(x_test, y_test, batch_size = 64)

print(f"Test loss = {result[0]}")
print(f"Test MAE = {result[1]}")
print(f"Test MSE = {result[2]}")

#%%
import matplotlib.pyplot as plt

predictions = np.squeeze(model.predict(x_test))
labels = np.squeeze(y_test)
plt.plot(predictions, labels, ".")
plt.xlabel("Predictions")
plt.ylabel("Labels")
plt.title("Graph of Predictions vs Labels")
plt.show()