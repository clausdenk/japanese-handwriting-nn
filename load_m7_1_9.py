"""Example job for running a neural network."""

import h5py
import numpy as np

from preprocessing.make_keras_input import data
from preprocessing.data_utils import index_to_kanji
from models import M7_1_9
from keras.optimizers import Adam 
from keras import callbacks

def load_model_weights(name, model):
    try:
        model.load_weights(name)
    except:
        print("Can't load weights!")


def save_model_weights(name, model):
    try:
        model.save_weights(name)
    except:
        print("failed to save classifier weights")
    pass

img_rows, img_cols = 64, 64

print ("Loading training and test data ..")
X_train, y_train, X_test, y_test, input_shape, inv_map = data(mode='kanji')
n_output = y_train.shape[1]
print ("Training size: ", X_train.shape[0])
print ("Test size: ", X_test.shape[0])
print ("Classes: ", n_output)

# setup model
model = M7_1_9(n_output=n_output, input_shape=input_shape)

adam = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# try to load weights




