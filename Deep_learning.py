#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:31:02 2023

@author: mainaoshige
"""

# installing modules

# numpy https://numpy.org
# To install,
# pip install numpy
#  or
# conda install numpy

# pandas https://pandas.pydata.org
# To install,
# pip install pandas
#  or
# conda install -c conda-forge pandas

# tensorflow https://www.tensorflow.org/?hl=en
# To install,
# pip install tensorflow==2.10.0

# tensorflow.js https://www.tensorflow.org/js?hl=en
# To install,
# pip install tensorflowjs==3.21.0

# Matplotlib https://matplotlib.org
# To install,
# pip install matplotlib
# conda install -c conda-forge matplotlib

# scikit-learn https://scikit-learn.org/stable/
# To install,
# pip install -U scikit-learn 
#  or
# conda install -c conda-forge matplotlib

# imbalanced-learn https://imbalanced-learn.org/stable/index.html
# To install,
# pip install -U imbalanced-learn
#  or
# conda install -c conda-forge imbalanced-learn


# Loading the library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout

import pandas as pd
import numpy as np

# Prevent OMP Abort error with anaconda
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Declaring variables 4 categories
classes = 4
# # Declaring variables 2 categories
# classes = 2

# Read the CSV file
coviddata =pd.read_csv('4deep_clinicalbloodAI.csv', encoding="utf-8").to_numpy()

covid_all=coviddata.astype('float')

# Shuffle the data
np.random.shuffle(covid_all)

# Divide the shuffled data into 5
covid1,covid2,covid3,covid4,covid5 = np.array_split(covid_all,5)

# Concatenate covid1,covid2,covid3,covid4 with 4/5 as training data
covid_train = np.concatenate([covid1, covid2,covid3,covid4])


# Set the ExplanatoryVariables
ExplanatoryVariables = len(covid_all[1])-1

# Set explanatory variables
covid_xtrain=(covid_train[:,1:])

# Use the severity data as the target variable
covid_ytrain=(covid_train[0:len(covid_train),0])

# Use 1/5 as verification data
covid_test = covid5

# Set explanatory variables
covid_xtest=(covid_test[:,1:])

# Use the severity data as the target variable
covid_ytest=(covid_test[0:len(covid_test),0])


# Creation of neural circuits
model = Sequential()
model.add(Dense(128, input_shape=(ExplanatoryVariables,)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.1))
model.add(Dense(units=128))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.2))
model.add(Dense(units=64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.3))
model.add(Dense(units=32))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.4))

# 4 categories output
model.add(Dense(4,  activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='Adagrad' ,
              metrics=["accuracy", "Recall", "Precision"]
              )

# #2 categories output
# model.add(Dense(2,  activation='sigmoid'))
# model.compile(loss='binary_crossentropy',
#               optimizer='Adagrad' ,
#               metrics=["accuracy", "Recall", "Precision"]
#               )


# View the shape of the created neural circuit
model.summary()

# # One-hot encode
from tensorflow.keras.utils import to_categorical

covid_ytest_binary = to_categorical(covid_ytest)
covid_ytrain_binary = to_categorical(covid_ytrain)


history = model.fit(covid_xtrain, covid_ytrain_binary, 
                    batch_size=32, epochs=200, 
                    validation_split=0.1)

score = model.evaluate(covid_xtest, covid_ytest_binary, verbose=0)

# View learning progress in a graph
import matplotlib.pyplot as plt

history_dict = history.history
# Put Loss (error from correct answer) into the loss_values
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
# Put accuracy in ACC
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
# Make a list from 1 to epoch number
epochlist = range(1, len(loss_values) +1)

# Create an accuracy graph
plt.plot(epochlist, acc, 'go', label='Accuracy at training')
plt.plot(epochlist, val_acc, 'b', label='Accuracy at validation')
# Create a graph of Loss (error from correct answer)
plt.plot(epochlist, loss_values, 'mo', label='Loss at training')
plt.plot(epochlist, val_loss_values, 'r', label='Loss at validation')

# Title
plt.title('Number of training, accuracy, and loss')
plt.xlabel('Number of learnings (epoch number)')
plt.legend()
# View the graph
plt.show()
# Use validation datasets to display loss and percentage of correct answers
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test Recall:",score[2])
print("Test Precision:",score[3])

prediction = model.predict(covid_xtest, verbose=0)
print(covid_ytest)
y_pred = prediction.argmax(axis=1)
print(prediction.argmax(axis=1))



