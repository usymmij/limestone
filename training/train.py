import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np

import wandb
from wandb.keras import WandbCallback

wandb.init(project="limestone", entity="usymmij")

BATCHES=1
EPOCHS=1000

input_shape = (21,3)

labls = np.load('lbl.npy', allow_pickle=True) 
data = np.load('data.npy',allow_pickle=True)

model = keras.models.Sequential([
    keras.layers.Conv1D(64, kernel_size=(9), strides=3,
                        padding= 'same', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling1D(pool_size=(3), strides= 1,
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),

    keras.layers.Conv1D(64, kernel_size=(6), strides=1,
                        padding= 'same', activation= 'relu',
                        kernel_initializer= 'he_normal'),
    keras.layers.MaxPooling1D(pool_size=(3), strides= 1,
                              padding= 'valid', data_format= None),
    keras.layers.BatchNormalization(),

    keras.layers.Dropout(0.1),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(48, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(2, activation="relu"),
])

model.summary()

print('compiling model')

# we use mse loss here because there needs to be numerical accuracy rather than categorical
model.compile(
    optimizer = keras.optimizers.SGD(learning_rate=0.015),
    loss="mse",
    metrics=["mse", "mae"])

model.fit(data, labls, BATCHES, EPOCHS, shuffle=True, callbacks=[WandbCallback()])
model.save("model.h5")