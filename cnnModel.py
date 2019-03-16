#!/usr/bin/env python
# coding: utf-8
# MCSZN with cherry picking


########################
#IMPORT DEPENDENCIES   #
########################
import os, cv2, re
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib import pyplot as plt
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils import plot_confusion_matrix, natural_keys, atoi


#############################
#SET UP ENV & IMPORT DATA   #
#############################


# increase img size for better results
# requires lot more computing power
img_width = 64
img_height = 64
TRAIN_DIR = 'input/train/'
TEST_DIR = 'input/test/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = [] # images as arrays
    y = [] # labels
    
    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (img_width,img_height), interpolation=cv2.INTER_CUBIC))
    
    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
            
    return x, y

train_images_dogs_cats.sort(key=natural_keys)

########################
#DATA PROCESSING       #
########################

# get Xs and Ys and split the data
X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.5)


########################
# CONV NET MODELLING   #
########################

# select optimizer and hyperparams
optimizer = optimizers.Adam() #lr= 1e-6, beta_1=0.9, beta_2=0.9)

model = models.Sequential()

# 3 conv layers with maxpooling
# 2 hidden flat layers and 1 output
model.add(layers.Conv2D(16, (2, 2), input_shape=(img_width, img_height, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(4, 4)))

model.add(layers.Conv2D(32, (2, 2)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3)))

model.add(layers.Conv2D(64, (2, 2)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(32))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


########################
# FITTING OUR MODEL    #
########################

# convert list to np.array
X_train = np.array(X_train)

# train on 10 epochs, turn of verbose for slightly faster training
history = model.fit(
    X_train, Y_train,
    epochs=10, verbose=1,
    batch_size = 128
)


########################
# PLOT TRAINING        #
########################


plt.plot(history.history['loss'], label="loss")
plt.plot(history.history['acc'], label="accuracy")
plt.title("Evolution of neural net on training set")
plt.legend()
plt.show()

########################
# PREDICT TEST DATA    #
########################

X_test , Y_test = np.array(X_test), Y_test
Y_pred = model.predict( X_test, batch_size=64, verbose=1)
Y_pred = np.where(Y_pred > 0.5, 1, 0)

########################
# PLOT CONFUSION MATRIX#
########################

plt.figure()
plot_confusion_matrix(confusion_matrix(Y_test, Y_pred), ["dogs", "cats"])
plt.show()

