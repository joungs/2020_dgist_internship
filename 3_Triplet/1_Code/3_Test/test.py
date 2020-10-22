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
import random
#siamese_network, weights
siamese_net = load_model('/Users/JunJoung/PycharmProjects/Siamese/siamese_net_17.hd5')
# weights = siamese_net.get_weights()
# print(weights)
MAX_LENGTH = 2
WIDTH = 224
HEIGHT = 224


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

def make_set2(file_path):
    if file_path == ('2', '/Users/JunJoung/PycharmProjects/Siamese/rs/2,3/2.png'):
        return '3', '/Users/JunJoung/PycharmProjects/Siamese/rs/2,3/2.png'
    elif file_path == ('3', '/Users/JunJoung/PycharmProjects/Siamese/rs/2,3/2.png'):
        return '2', '/Users/JunJoung/PycharmProjects/Siamese/rs/2,3/2.png'
    elif file_path == ('4', '/Users/JunJoung/PycharmProjects/Siamese/rs/4,5/4.png'):
        return '5', '/Users/JunJoung/PycharmProjects/Siamese/rs/4,5/4.png'
    elif file_path == ('5', '/Users/JunJoung/PycharmProjects/Siamese/rs/4,5/4.png'):
        return '4', '/Users/JunJoung/PycharmProjects/Siamese/rs/4,5/4.png'
    elif file_path == ('6', '/Users/JunJoung/PycharmProjects/Siamese/rs/6,7/7.png'):
        return '7', '/Users/JunJoung/PycharmProjects/Siamese/rs/6,7/7.png'
    elif file_path == ('7', '/Users/JunJoung/PycharmProjects/Siamese/rs/6,7/7.png'):
        return '6', '/Users/JunJoung/PycharmProjects/Siamese/rs/6,7/7.png'
    elif file_path == ('8', '/Users/JunJoung/PycharmProjects/Siamese/rs/8,9/8.png'):
        return '9', '/Users/JunJoung/PycharmProjects/Siamese/rs/8,9/8.png'
    else:
        return '8', '/Users/JunJoung/PycharmProjects/Siamese/rs/8,9/8.png'

# __main__ for later usage in different code, extract this part for later

# pathway of images for labels and image
path = '/Users/JunJoung/PycharmProjects/Siamese/meat_images1'
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
train_images.sort()
#set label images as dictionary
label_images = dict()
for label in labels:
    #same processes as training
    for g in label.split(','):
        label_images[g] = glob.glob('%s/%s/*' % (path_label, label))[0:1]
label_images = dict_to_rand_images(label_images)
label_images.sort()
print(len(train_images))
print(len(label_images))

# empty list as x, x1,y
x, x1, x2, y, y1, y2 = [], [], [], [], [], []
x23 = []
for label in label_images:
    x23.append(label)
# list of label_images into label
for train in train_images:
    for label in label_images:
        x.append(train)
        y.append(1.0 if make_set(int(train[0])) == make_set(int(label[0])) else 0.0)
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
x_right = x_right.astype(np.float32)
x_pos = x_pos.astype(np.float32)
x_neg = x_neg.astype(np.float32)

print(len(x_right))
print(len(x_pos))
print(len(x_neg))

result = siamese_net.predict([x_right, x_pos, x_neg])
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




