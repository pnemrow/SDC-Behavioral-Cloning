from pandas import read_csv as csv
from scipy.misc import imread, imresize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Convolution2D, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.utils import shuffle
import preprocessing


csv_data = csv('driving_log.csv')
csv_data.columns = ['Center_Images', 'Left_Images', 'Right_Images', 'Steering_Angle', 'Throttle', 'Break', 'Speed']

X_train = np.zeros(shape=(csv_data.shape[0], 40, 40, 1))
y_train = csv_data['Steering_Angle']

for i, image_path in enumerate(csv_data['Center_Images']):
    image = imread(image_path, mode='RGB')
    image = preprocessing.preprocess(image)
    X_train[i] = image

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



# model = Sequential()
# model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf', input_shape=(40, 40, 1)))
# model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='tf'))
# model.add(Flatten())
# model.add(Dense(output_dim=64))
# model.add(Activation("relu"))
# model.add(Dropout(p=0.5))
# model.add(Dense(output_dim=1))

# model.compile(loss='mean_squared_error', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X_train, y_train, nb_epoch=200, batch_size=256)
with open('model.json', mode='w', encoding='utf8') as f:
    f.write(model.to_json())

model.save_weights('model.h5')