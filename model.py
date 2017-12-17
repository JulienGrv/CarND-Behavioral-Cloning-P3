#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2
import numpy as np
import pandas as pd
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Lambda
from keras.models import Sequential
from keras.optimizers import Adam

ROOT_DIRPATH = os.path.dirname(os.path.realpath(__file__))
RECORDED_DATA_DIRNAME = 'data'
IMAGES_DIRNAME = 'IMG'
DRIVING_LOG_FILENAME = 'driving_log.csv'
DRIVING_LOG_FILEPATH = os.path.join(ROOT_DIRPATH, RECORDED_DATA_DIRNAME,
                                    DRIVING_LOG_FILENAME)


def test_model(img_height=160, img_width=320, img_chanels=3):
    input_shape = (img_height, img_width, img_chanels)

    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))

    # model.add(
    #     BatchNormalization(
    #         epsilon=0.001, mode=2, axis=1, input_shape=img_shape))

    model.add(
        Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(
        Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    return model


def build_nvidia_model(img_height=160, img_width=320, img_chanels=3):
    img_shape = (img_height, img_width, img_chanels)

    model = Sequential()

    model.add(
        BatchNormalization(
            epsilon=0.001, mode=2, axis=1, input_shape=img_shape))

    model.add(
        Conv2D(24, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(36, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(48, (5, 5), padding='valid', activation='relu', strides=(2, 2)))
    model.add(
        Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(
        Conv2D(64, (3, 3), padding='valid', activation='relu', strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    return model


def generate_batch(batch_size=64):
    df = pd.read_csv(DRIVING_LOG_FILEPATH)
    print(df)


def read_images(paths):
    basepath = os.path.join(ROOT_DIRPATH, RECORDED_DATA_DIRNAME)
    images = []
    for filepath in paths:
        image_path = os.path.join(basepath, filepath)
        images.append(cv2.imread(image_path))
    images = np.array(images)
    return images


def test_1():
    driving_log = pd.read_csv(DRIVING_LOG_FILEPATH)
    center_col = 'center'
    images = read_images(driving_log[center_col])
    steering_col = 'steering'
    X_train = images[0:1000]
    y_train = driving_log[steering_col].values[0:1000]
    print(X_train)
    print(y_train)
    model = train(X_train, y_train, test_model, 0.001, 7)
    model.save('model.h5')


def train(x, y, model=build_nvidia_model, learning_rate=0.0001, epochs=10):
    model = model()
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    model.fit(
        x, y, batch_size=64, epochs=epochs, validation_split=0.2, shuffle=True)
    return model


def train_generator(model=build_nvidia_model, learning_rate=0.0001, epochs=10):
    model = model()
    adam = Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
    generator = None
    steps_per_epoch = None
    validation_data = None
    validation_steps = None
    model.fit_generator(
        generator,
        steps_per_epoch,
        epochs=epochs,
        validation_data=validation_data,
        validation_steps=validation_steps)


def main():
    # train()
    # generate_batch()
    # read_images()
    test_1()


if __name__ == '__main__':
    sys.exit(main())
