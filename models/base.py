import tensorflow as tf
import tensorflow.keras as keras
import os
import sys

import numpy


class DeModel(keras.models.Model):
    def __init__(self):
        keras.models.Model.__init__(self)

        return

    def call(self, inputs, training=None, mask=None):
        keras.models.Model.call(self, inputs=inputs, training=training, mask=mask)
        return






