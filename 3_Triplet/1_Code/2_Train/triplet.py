import cv2, os, glob
import numpy as np
import cv2
import os
import glob
from model import create_model
from tensorflow import keras
from tensorflow.keras import regularizers, backend, initializers
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, Dropout, Input, Flatten, Subtract, \
    ZeroPadding2D, BatchNormalization, LayerNormalization, Softmax, concatenate, add, LeakyReLU, subtract, \
    AveragePooling2D, GlobalAveragePooling2D, Input, Layer, Embedding
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.activations import selu
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.optimizers
from scipy import spatial
import pickle
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
import random
import tensorflow as tf
from tensorflow.keras.layers import Lambda

# MAX_lenth, unknown, width and height for later resizing
# Max_length = number in each training images in given
MAX_LENGTH = 1
WIDTH = 224
HEIGHT = 224

from tensorflow.python.framework import ops

ops.reset_default_graph()

# Dictionary - binary to list
def dict_to_rand_images(dictionary: dict, shuffle: bool = True):
    r_value = []
    for key in dictionary.keys():
        for d in dictionary[key]:
            r_value.append((key, d))
    # shuffle
    if shuffle:
        for _ in range(0, 5):
            np.random.shuffle(r_value)
    return r_value


# If result is 1, then return 1, if result is either 2 or 3, return [2,3] and so on
def make_set2(file_path):
    if file_path == ('2', '/Users/JunJoung/PycharmProjects/Siamese/rs/2/2.jpg'):
        return '3', '/Users/JunJoung/PycharmProjects/Siamese/rs/3/3.jpg'
    elif file_path == ('3', '/Users/JunJoung/PycharmProjects/Siamese/rs/3/3.jpg'):
        return '2', '/Users/JunJoung/PycharmProjects/Siamese/rs/2/2.jpg'
    elif file_path == ('4', '/Users/JunJoung/PycharmProjects/Siamese/rs/4/4.jpg'):
        return '5', '/Users/JunJoung/PycharmProjects/Siamese/rs/5/5.jpg'
    elif file_path == ('5', '/Users/JunJoung/PycharmProjects/Siamese/rs/5/5.jpg'):
        return '4', '/Users/JunJoung/PycharmProjects/Siamese/rs/4/4.jpg'
    elif file_path == ('6', '/Users/JunJoung/PycharmProjects/Siamese/rs/6/6.jpg'):
        return '7', '/Users/JunJoung/PycharmProjects/Siamese/rs/7/7.jpg'
    elif file_path == ('7', '/Users/JunJoung/PycharmProjects/Siamese/rs/7/7.jpg'):
        return '6', '/Users/JunJoung/PycharmProjects/Siamese/rs/6/6.jpg'
    elif file_path == ('8', '/Users/JunJoung/PycharmProjects/Siamese/rs/8/8.jpg'):
        return '9', '/Users/JunJoung/PycharmProjects/Siamese/rs/9/9.jpg'
    else:
        return '8', '/Users/JunJoung/PycharmProjects/Siamese/rs/8/8.jpg'

# __main__ for later usage in different code, extract this part for later
if __name__ == '__main__':

    # pathway of images for labels and image
    path = '/Users/JunJoung/PycharmProjects/Siamese/meat_images1'
    path_label = '/Users/JunJoung/PycharmProjects/Siamese/rs'
    # whatever were inside of path, change to list form
    grades = os.listdir(path)
    labels = os.listdir(path_label)

    # initially set train_image as dictionary
    train_images = dict()
    for grade in grades:
        # split dataset, since there were combined (2,3)
        for g in grade.split(','):
            # import dataset for train_image, max Length?
            train_images[g] = glob.glob('%s/%s/*' % (path, grade))[0:MAX_LENGTH]
    train_images = dict_to_rand_images(train_images)
    train_images.sort()
    print(len(train_images))
    # set label images as dictionary
    label_images = dict()
    for label in labels:
        # same processes as training
        for g in label.split(','):
            label_images[g] = glob.glob('%s/%s/*' % (path_label, label))[0:2]
    label_images = dict_to_rand_images(label_images)
    label_images.sort()
    # empty list as x, x1,y
    x, x1, x2, y, y1, y2 = [], [], [], [], [], []
    x23 = []
    for label in label_images:
        x23.append(label)
    # list of label_images into label
    for train in train_images:
        for label in label_images:
            x.append(train)
            y.append(0.0)
    index = [i for i in range(0, len(y))]

    for a in x:
        for label in label_images:
            if int(a[0]) == int(label[0]):
                x1.append(label)
                x12 = label
                x23.remove(x12)
                x23.remove(make_set2(x12))
                x2.append(random.choice(x23))
                x23.append(x12)
                x23.append(make_set2(x12))
    print(x)
    print(x1)
    print(x2)

    print(y)
    print(len(x))
    print(len(x1))
    print(len(x2))
    np.random.shuffle(index)

    x = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x]
    x1 = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x1]
    x2 = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x2]

    x_right = []
    x_pos = []
    x_neg = []

    for i in range(0, len(index)):
        x_right.append(x[index[i]])
        x_pos.append(x1[index[i]])
        x_neg.append(x2[index[i]])
        y1.append(y[index[i]])

    x_right = np.array(x_right)
    x_pos = np.array(x_pos)
    x_neg = np.array(x_neg)
    y1 = np.array(y1)

    siamese_net = create_model()
    siamese_net.summary()

    in_a = Input(shape=(224, 224, 3))
    in_p = Input(shape=(224, 224, 3))
    in_n = Input(shape=(224, 224, 3))

    emb_a = siamese_net(in_a)
    emb_p = siamese_net(in_p)
    emb_n = siamese_net(in_n)


    class TripletLossLayer(Layer):
        def __init__(self, alpha, **kwargs):
            self.alpha = alpha
            super(TripletLossLayer, self).__init__(**kwargs)

        def triplet_loss(self, inputs):
            a, p, n = inputs
            p_dist = K.sum(K.square(a - p))
            n_dist = K.sum(K.square(a - n))
            return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0))

        def call(self, inputs):
            loss = self.triplet_loss(inputs)
            self.add_loss(loss)
            return loss


    triplet_loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')([emb_a, emb_p, emb_n])

    siamese_net_train = Model([in_a, in_p, in_n], triplet_loss_layer)

    siamese_net_train.summary()

    siamese_net_train.compile(loss=tf.losses.mean_squared_error,
                              optimizer=keras.optimizers.Adam(learning_rate=5e-6, beta_1=0.9, beta_2=0.999, epsilon=0.1),
                              metrics=['accuracy'])

    print(x_right.shape)
    print(x_pos.shape)
    print(x_neg.shape)
    print(y1.shape)

    siamese_net_train.fit([x_right, x_pos, x_neg],y1, batch_size=100, epochs=10, verbose=1, validation_split=0.1)
    siamese_net.save('/Users/JunJoung/PycharmProjects/Siamese/siamese_net_17.hd5')
    result = siamese_net_train.evaluate([x_right, x_pos, x_neg], y1)
    result = siamese_net_train.predict([x_right, x_pos, x_neg])
    print(result)
    print(result.shape)