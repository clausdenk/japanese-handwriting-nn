"""Example job for running a neural network."""

import h5py
import numpy as np

from preprocessing.make_keras_input import data
from models import M7_1
from keras import backend as K
from keras.optimizers import Adam 



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
X_train, y_train, X_test, y_test = data(mode='kanji')
n_output = y_train.shape[1]

# if K.image_dim_ordering() == 'th':
#     X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
#     X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
#     X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# print ("using shape",input_shape)
# model = M7_1(n_output=n_output, input_shape=input_shape)

# load_model_weights('weights/weights_in.h5', model)

# adam = Adam(lr=1e-4)
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# model.fit(X_train, y_train,
#           epochs=20,
#           batch_size=16)

# score, acc = model.evaluate(X_test, y_test,
#                             batch_size=16,
#                             verbose=0)
# print ("Training size: ", X_train.shape[0])
# print ("Test size: ", X_test.shape[0])
# print ("Test Score: ", score)
# print ("Test Accuracy: ", acc)
# save_model_weights('weights/weights_out.h5', model)
