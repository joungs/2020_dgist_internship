import cv2, os, glob
import numpy as np
import cv2
import os
import glob
from tensorflow import keras
from tensorflow.keras import regularizers, backend, initializers
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, Dropout, Input, Flatten, Subtract, \
    BatchNormalization, LayerNormalization, Softmax, concatenate, add, LeakyReLU, subtract
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.activations import selu
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.optimizers
from scipy import spatial
import pickle
import tensorflow as tf
import xlsxwriter
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from pandas import DataFrame
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from PIL import ImageGrab
import matplotlib.pyplot as plt
import itertools


# MAX_lenth, unknown, width and height for later resizing
#Max_length = number in each training images in given
MAX_LENGTH = 1
WIDTH = 105
HEIGHT = 105


#Dictionary - binary to list
def dict_to_rand_images(dictionary: dict, shuffle: bool = True):
    r_value = []
    for key in dictionary.keys():
        for d in dictionary[key]:
            r_value.append((key, d))
#shuffle
    if shuffle:
        for _ in range(0, 5):
            np.random.shuffle(r_value)
    return r_value


# If result is 1, then return 1, if result is either 2 or 3, return [2,3] and so on
def make_set(number):
    if number == 1:
        return [1]
    elif number == 2 or number == 3:
        return [2, 3]
    elif number == 4 or number == 5:
        return [4, 5]
    elif number == 6 or number == 7:
        return [6, 7]
    else:
        return [8, 9]


# __main__ for later usage in different code, extract this part for later
if __name__ == '__main__':

    # pathway of images for labels and image
    path = '/Users/JunJoung/PycharmProjects/Siamese/meat_images2'
    path_label = '/Users/JunJoung/PycharmProjects/Siamese/rs'
    #whatever were inside of path, change to list form
    grades = os.listdir(path)
    labels = os.listdir(path_label)

#initially set train_image as dictionary
    train_images = dict()
    for grade in grades:
        #split dataset, since there were combined (2,3)
        for g in grade.split(','):
            # import dataset for train_image, max Length?
            train_images[g] = glob.glob('%s/%s/*' % (path, grade))[0:MAX_LENGTH]
    train_images = dict_to_rand_images(train_images)

#set label images as dictionary
    label_images = dict()
    for label in labels:
        #same processes as training
        for g in label.split(','):
            label_images[g] = glob.glob('%s/%s/*' % (path_label, label))
    label_images = dict_to_rand_images(label_images)

    for label in label_images:
        print(label)

# empty list as x, x1,y
    x, x1, y, y1,y2 = [], [],[],[],[]
#list of label_images into label
    for train in train_images:
        for label in label_images:
            x.append(train)
            x1.append(label)
            x2.append(label)
            y.append(1.0 if make_set(int(train[0]))==make_set(int(label[0])) else 0.0)
    index = [i for i in range(0, len(y))]

    x = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x]
    x1 = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x1]

    np.random.shuffle(index)

    x_right = []
    x_pos = []

    for i in range(0, len(index)):
        x_right.append(x[index[i]])
        x_pos.append(x1[index[i]])
        y1.append(y[index[i]])

    x_right = np.array(x_right)
    x_pos = np.array(x_pos)
    y1 = np.array(y1)
#change list to array

#input size
    X = Input(shape=(HEIGHT, WIDTH, 3))
#convolutional layers
    conv_1 = Convolution2D(64, (3, 3), activation='selu')(X)
    batch1 = BatchNormalization()(conv_1)
    max_1 = MaxPooling2D((2, 2), strides=(2, 2))(batch1)
    conv_2 = Convolution2D(128, (7, 7), activation='selu')(max_1)
    batch2 = BatchNormalization()(conv_2)
    max_2 = MaxPooling2D((2, 2), strides=(2, 2))(batch2)
    conv_3 = Convolution2D(256, (3, 3), activation='selu')(max_2)
    batch3 = BatchNormalization()(conv_3)
    max_3 = MaxPooling2D((2, 2), strides=(2, 2))(batch3)
    conv_4 = Convolution2D(256, (3, 3), activation='selu')(max_3)
    batch4 = BatchNormalization()(conv_4)
#flatten and dense twice
    fcl = Flatten()(batch4)
    dense_1 = Dense(4096, activation='selu')(fcl)
    dense_2 = Dense(1024, activation='selu')(dense_1)

#same as first input
    X1 = Input(shape=(HEIGHT, WIDTH, 3))
    conv2_1 = Convolution2D(64, (3, 3), activation='selu')(X1)
    batch2_1 = BatchNormalization()(conv2_1)
    max2_1 = MaxPooling2D((2, 2), strides=(2, 2))(batch2_1)
    conv2_2 = Convolution2D(128, (7, 7), activation='selu')(max2_1)
    batch2_2 = BatchNormalization()(conv2_2)
    max2_2 = MaxPooling2D((2, 2), strides=(2, 2))(batch2_2)
    conv2_3 = Convolution2D(256, (3, 3), activation='selu')(max2_2)
    batch2_3 = BatchNormalization()(conv2_3)
    max2_3 = MaxPooling2D((2, 2), strides=(2, 2))(batch2_3)
    conv2_4 = Convolution2D(256, (3, 3), activation='selu')(max2_3)
    batch2_4 = BatchNormalization()(conv2_4)
    fcl2 = Flatten()(batch2_4)
    dense2_1 = Dense(4096, activation='selu')(fcl2)
    dense2_2 = Dense(1024, activation='selu')(dense2_1)

    X2 = Input(shape=(HEIGHT, WIDTH, 3))
    # flatten and dense twice
    fcl3 = Flatten()(batch3_4)
    dense3_1 = Dense(4096, activation='selu')(fcl3)
    dense3_2 = Dense(1024, activation='selu')(dense3_1)
    dense_layer = backend.abs(subtract([dense_2, dense2_2]))
#dense to 1 output
    out = Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l1(1e-3))(dense_layer)
    #hyper
    siamese_net = Model(inputs=[X, X1], outputs=out)
    print(siamese_net.summary())

    siamese_net.compile(loss='binary_crossentropy',
                        optimizer=keras.optimizers.Adam(learning_rate=5e-6, beta_1=0.9, beta_2=0.999),
                        metrics=['accuracy'])
    siamese_net.fit([x_right, x_pos], y1,  batch_size=100, epochs=1, verbose=1, validation_split=0.1)
    siamese_net.save('/Users/JunJoung/PycharmProjects/Siamese/siamese_net_12.hd5')
