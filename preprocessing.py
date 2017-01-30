import numpy as np
import cv2
from scipy.misc import imresize
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def preprocess(image):
    image = crop(image)
    image = resize(image)
    image = get_saturation(image)
    image = normalize(image)
    return image

def crop(image):
    height = image.shape[0]
    width = image.shape[1]
    top_crop = int(height * .36)
    return image[top_crop:height, 0:width]

def resize(image):
    height = image.shape[0]
    width = image.shape[1]
    return imresize(image, (int(height/2), int(width/2)))

def get_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv[:,:,1]

def normalize(image):
    image = np.interp(image, [0, 255], [-0.5, 0.5])
    return np.expand_dims(image, 2)

def flip_image(image):
    return np.fliplr(image)

def flip_steer(steer):
    return -1.0 * steer

def filter_shuffle_split(X_train, y_train):
    X_train, y_train = filter_zeros(X_train, y_train)
    X_train, y_train = shuffle(X_train, y_train)
    return train_test_split(X_train, y_train, test_size=0.2, random_state=21)

def filter_zeros(X_train, y_train):
    mask = (y_train == 0.0) & (np.random.rand(y_train.shape[0]) > 0.25)
    y_train = y_train[np.logical_not(mask)]
    X_train  = X_train[np.logical_not(mask)]
    return X_train, y_train