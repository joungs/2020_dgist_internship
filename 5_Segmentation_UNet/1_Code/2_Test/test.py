import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

image_size = 128
model = load_model('/Users/JunJoung/PycharmProjects/beef_segmentation/U_net1.hd5')
test = cv2.imread('/Users/JunJoung/PycharmProjects/beef_segmentation/3/e17.jpg',3)
test = cv2.resize(test, (image_size,image_size))
test = test/255.0

test = np.asarray(test)
test = test.astype(np.float32)
test = test.reshape(1, image_size, image_size, 3)

result = model.predict(test)


fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray");plt.show()