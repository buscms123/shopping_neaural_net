# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 17:19:54 2021

@author: senay
"""


import pandas as pd



# first neural network with keras tutorial
from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf


#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/ #
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


dataset=pd.read_csv(r"C:\Users\senay\OneDrive\MSA\Classes\Python\Review\Neural\online_shoppers_intention.csv",delimiter=',')



dataset.head()

X=dataset.iloc[:,0:17]
y = dataset.iloc[:,17:18]
# y=np.asarray(y,dtype=np.float)
X.Month=X.Month.astype('category')
X.VisitorType=X.VisitorType.astype('category')
y.Revenue=y.Revenue.astype('string')
X.Weekend=X.Weekend.astype('category')
X.head()
y.head()

X.dtypes
y.dtypes


y.loc[y['Revenue'] == 'False', 'target'] = 0
y.loc[y['Revenue'] == 'True', 'target'] = 1

y=y.drop(['Revenue'],axis=1)


y = np.asarray(y).astype('float32')


X=pd.get_dummies(X)



for col in X:
    print(col,X[col].nunique(),dataset[col].dtypes)

# X = np.expand_dims(X, -1)
# y   = np.expand_dims(y, -1)
# show_shapes()

train_data = tf.data.Dataset.from_tensor_slices((X))
test_data = tf.data.Dataset.from_tensor_slices((y))


# define the keras model
model = tf.keras.Model()
#model = Sequential()
model.add(Dense(12, input_dim=29, activation='relu'))
model.add(Dense(29, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



    
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))