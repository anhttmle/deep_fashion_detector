import time
import sys
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class Model():
    def __init__(self):
        self.build_model()
        return

    def preprocess(self):
        return

    def postprocess(self):
        return

    def build_model(self):
        # input_tf = keras.layers.Input(shape=(100, 100, 3))
        #
        # X = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))(input_tf)
        #
        #
        # X = keras.layers.Flatten()(X)
        #
        # output_tf = keras.layers.Dense(units=1, activation=keras.activations.sigmoid)(X)
        #
        # self.model = keras.models.Model(inputs=input_tf, outputs=output_tf)
        # self.model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam())

        base_model = keras.applications.NASNetLarge()
        self.model = base_model
        base_model.compile(optimizer = keras.optimizers.Adam(), loss=tf.losses.categorical_crossentropy)

        base_model.summary()
        return

    @classmethod
    def run(cls):
        return


if __name__ == "__main__":
    fake_data = np.random.random_sample(size=(1000, 224, 224, 3))
    fake_label = np.random.random_sample(size=fake_data.shape[0])
    print("START MAIN")
    model = Model()
    model.model.fit(x=fake_data, y=fake_label, epochs=100)
    # model.run()

print(__name__)