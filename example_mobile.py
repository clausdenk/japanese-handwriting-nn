"""Example job for running a neural network."""

import h5py
import numpy as np

from preprocessing.make_keras_input import data
from models import MobileNet
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
X_train, y_train, X_test, y_test, input_shape = data(mode='kanji')
n_output = y_train.shape[1]

model = MobileNet(classes=n_output, input_shape=input_shape)

adam = Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=5, batch_size=16) #  verbose=2 

score, acc = model.evaluate(X_test, y_test, batch_size=16, verbose=0)

print ("Training size: ", X_train.shape[0])
print ("Test size: ", X_test.shape[0])
print ("Test Score: ", score)
print ("Test Accuracy: ", acc)

save_model_weights('weights/weights_out.h5', model)


model.summary()
