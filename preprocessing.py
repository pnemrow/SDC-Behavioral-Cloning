import numpy as np
import cv2
from scipy.misc import imresize

def preprocess(image):
    image = resize(image)
    image = grayscale(image)
    image = normalize(image)
    return image

def resize(image):
    height = 40
    weight = 40
    return imresize(image, (height, weight))

def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return gray[:,:,1]

def normalize(image):
    image = np.interp(image, [0, 255], [-0.5, 0.5])
    return np.expand_dims(image, 2)

def flip_image(image):
    return np.fliplr(image)