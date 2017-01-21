from pandas import read_csv as csv
import random
from scipy.misc import imread, imresize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.utils import shuffle
from preprocessing import preprocess, flip_image
import matplotlib.pyplot as plt

csv_data = csv('driving_log.csv')
csv_data.columns = ['Center_Images', 'Left_Images', 'Right_Images', 'Steering_Angle', 'Throttle', 'Break', 'Speed']

X_train = np.empty(shape=(0, 40, 40, 1))
y_train = np.array([])


for i in range(len(csv_data)):
    if i % 1000 == 0:
        print(i/len(csv_data))

    center_steer = csv_data['Steering_Angle'][i]
    
    if not (center_steer == 0 and random.random() > 0.1):
        center_image = preprocess(imread(csv_data['Center_Images'][i], mode='RGB'))
        left_image = preprocess(imread(csv_data['Left_Images'][i], mode='RGB'))
        left_steer = center_steer + 0.3
        right_image = preprocess(imread(csv_data['Right_Images'][i], mode='RGB'))
        right_steer = center_steer - 0.3
        flipped_center_image = flip_image(center_image)
        flipped_center_steer = -1.0 * center_steer
        flipped_left_image = flip_image(left_image)
        flipped_left_steer = -1.0 * left_steer
        flipped_right_image = flip_image(right_image)
        flipped_right_steer = -1.0 * right_steer

        X_train = np.append(X_train, np.array([center_image, left_image, right_image, flipped_center_image, flipped_left_image, flipped_right_image]), axis=0)
        y_train = np.append(y_train, np.array([center_steer, left_steer, right_steer, flipped_center_steer, flipped_left_steer, flipped_right_steer]))


X_train, y_train = shuffle(X_train, y_train)

# train
model = Sequential()
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(40, 40, 1)))
model.add(BatchNormalization(mode=1))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(BatchNormalization(mode=1))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(BatchNormalization(mode=1))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(BatchNormalization(mode=1))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")


model.fit(X_train, y_train, nb_epoch=25, batch_size=256)
with open('model.json', mode='w', encoding='utf8') as f:
    f.write(model.to_json())

model.save_weights('model.h5')