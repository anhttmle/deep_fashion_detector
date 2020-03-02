import os
import sys
import re
import shutil
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50

import utils.path as path
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from generator.image import ImageGeneratorByPath





def build_model():
    K.clear_session()
    model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for layer in model_resnet.layers[:-12]:
        # 6 - 12 - 18 have been tried. 12 is the best.
        layer.trainable = False

    x = model_resnet.output
    x = keras.layers.Dense(512, activation='elu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    y = keras.layers.Dense(60, activation='softmax', name='img')(x)

    # x_bbox = model_resnet.output
    # x_bbox = keras.layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    # x_bbox = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x_bbox)
    # bbox = keras.layers.Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

    final_model = keras.models.Model(inputs=model_resnet.input,
                        outputs=y)
                        # outputs=[y, bbox])

    # print(final_model.summary())

    opt = keras.optimizers.Adam(lr=0.0001)

    final_model.compile(optimizer=opt,
                        loss={
                            'img': 'sparse_categorical_crossentropy',
                            # 'bbox': 'mean_squared_error'
                        },
                        metrics={
                            'img': ['accuracy', 'top_k_categorical_accuracy'],  # default: top-5
                                 # 'bbox': ['mse']
                        }
                        )
    return final_model


train_set, dev_set, test_set = get_dict_bboxes()
train_datagen = ImageGeneratorByPath(
    train_set,
    ImageDataGenerator(),
)

dev_datagen = ImageGeneratorByPath(
    dev_set[:1],
    ImageDataGenerator(),
)

test_datagen = ImageGeneratorByPath(
    test_set,
    ImageDataGenerator()
)

lr_reducer = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=12,
    factor=0.5,
    verbose=1
)

# # tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')
early_stopper = keras.callbacks.EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)
# # checkpoint = keras.callbacks.ModelCheckpoint('./models/model.h5')
#
final_model = build_model()
final_model.fit_generator(
    train_datagen,
    epochs=1,
    validation_data=dev_datagen,
    verbose=1,
    shuffle=True,
    callbacks=[lr_reducer, early_stopper],
    workers=1
)

print("FINISH TRAINING")

scores = final_model.evaluate_generator(
    test_datagen,
    verbose=1
)

print('Multi target loss: ' + str(scores[0]))
print('Image loss: ' + str(scores[1]))
print('Bounding boxes loss: ' + str(scores[2]))
print('Image accuracy: ' + str(scores[3]))
print('Top-5 image accuracy: ' + str(scores[4]))
print('Bounding boxes error: ' + str(scores[5]))
