from data_extraction import get_training_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, Flatten
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization

def get_model():
    model = Sequential()
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf', input_shape=(51, 160, 1)))
    model.add(BatchNormalization(mode=1))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf'))
    model.add(BatchNormalization(mode=1))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', dim_ordering='tf'))
    model.add(BatchNormalization(mode=1))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', dim_ordering='tf'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(BatchNormalization(mode=1))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(BatchNormalization(mode=1))
    model.add(ELU())
    model.add(Dense(1))
    model.summary()
    return model

def save_model(model):
    with open('model.json', mode='w', encoding='utf8') as f:
        f.write(model.to_json())
    model.save_weights('model.h5')

X_train, X_validation, y_train, y_validation = get_training_data()
model = get_model()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=128, nb_epoch=5, verbose=1, validation_data=(X_validation, y_validation))
save_model(model)