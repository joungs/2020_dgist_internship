import cv2, os, glob
import numpy as np
import cv2
import os
import glob
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, MaxPooling2D, Convolution2D, Dropout, Input, Flatten, Subtract, BatchNormalization, Softmax, concatenate,add, LeakyReLU
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
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from tqdm import tqdm
#siamese_network, weights
siamese_net = load_model('/Users/JunJoung/PycharmProjects/Siamese/siamese_net.hd5')
# weights = siamese_net.get_weights()
# print(weights)
MAX_LENGTH = 25
WIDTH = 105
HEIGHT = 105


def dict_to_rand_images(dictionary: dict, shuffle: bool = True):
    r_value = []
    for key in dictionary.keys():
        for d in dictionary[key]:
            r_value.append((key, d))
    if shuffle:
        for _ in range(0, 5):
            np.random.shuffle(r_value)
    return r_value

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

path = '/Users/JunJoung/PycharmProjects/Siamese/meat_images1'
path_label = '/Users/JunJoung/PycharmProjects/Siamese/rs'
grades = os.listdir(path)
labels = os.listdir(path_label)

train_images = dict()
for grade in grades:
    for g in grade.split(','):
        train_images[g] = glob.glob('%s/%s/*' % (path, grade))[0:MAX_LENGTH]
train_images = dict_to_rand_images(train_images)

label_images = dict()
for label in labels:
    for g in label.split(','):
        label_images[g] = glob.glob('%s/%s/*' % (path_label, label))
label_images = dict_to_rand_images(label_images)

x, x1, y = [], [], []
for train in train_images:
    for label in label_images:
        x.append(train)
        x1.append(label)
        y.append(1.0 if make_set(int(train[0]))==make_set(int(label[0])) else 0.0)
index = [i for i in range(0, len(y))]

x = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x]
x1 = [cv2.resize(cv2.imread(name[1], cv2.IMREAD_COLOR), dsize=(WIDTH, HEIGHT)) for name in x1]

np.random.shuffle(index)

x_right = []
x_pos = []
y1 = []
for i in range(0, len(index)):
    x_right.append(x[index[i]])
    x_pos.append(x1[index[i]])
    x_neg.append(x2[index[i]])
    y1.append(y[index[i]])

x_right = np.array(x_right)
x_pos = np.array(x_pos)
x_neg = np.array(x_neg)
y1 = np.array(y1)
x_right = x_right.astype(np.float32)
x_pos = x_pos.astype(np.float32)
x_neg = x_neg.astype(np.float32)

print(len(x_right))
print(len(x_pos))

result = siamese_net.predict([x_right, x_pos])
print(result)
np.set_printoptions(threshold=np.inf)
print(result.shape)
result = np.resize(result, (len(train_images), len(label_images)))
print(result.shape)
for i in range(0, len(train_images)):
    print(np. argmin(result[i])+1)
result = np.transpose(result)
print(result.shape)
print('______')

results = []
for j in range(9):
    for i in range(9):
        results.append(dot(result[j], result[i])/(norm(result[j])*norm(result[i])))
results = np.resize(results, (9,9))
np.set_printoptions(threshold=np.inf)
print(results)


#0.999316 = same dataset
#0.7414379 = different dataset



