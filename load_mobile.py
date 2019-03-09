"""Example job for running a neural network."""

import h5py
import numpy as np
import time

from preprocessing.make_keras_input import data
from models import MobileNet
from keras.optimizers import Adam 
from keras import callbacks

def load_model_weights(name, model):
    try:
        model.load_weights(name)
    except:
        print("Can't load weights!")


img_rows, img_cols = 64, 64
X_train, y_train, X_test, y_test, input_shape, inv_map = data(mode='kanji')
n_output = y_train.shape[1]

model = MobileNet(classes=n_output, input_shape=input_shape)

load_model_weights('weights/weights_in.h5', model)

print ("Training size: ", X_train.shape[0])
print ("Test size: ", X_test.shape[0])


