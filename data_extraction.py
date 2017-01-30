from pandas import read_csv as csv
from scipy.misc import imread
import numpy as np
from preprocessing import preprocess, flip_image, flip_steer, filter_shuffle_split
import matplotlib.pyplot as plt

def get_training_data():
    csv_data = get_csv_data('driving_log.csv')
    X_train, y_train = csv_to_X_y(csv_data)
    return filter_shuffle_split(X_train, y_train)

def get_csv_data(filename):
    csv_data = csv(filename)
    csv_data.columns = ['Center_Images', 'Left_Images', 'Right_Images', 'Steering_Angle', 'Throttle', 'Break', 'Speed']
    return csv_data

def csv_to_X_y(csv_data):
    X_train = np.zeros(shape=(len(csv_data) * 6, 51, 160, 1))
    y_train = np.zeros(len(csv_data) * 6)
    train_index =  0

    for row in range(0, len(csv_data)):
        row_values = get_row_values(csv_data, row)
        X_train = add_X_row_data(X_train, row_values, train_index)
        y_train = add_y_row_data(y_train, row_values, train_index)        
        train_index += 6

    return X_train, y_train

def get_row_values(csv_data, row):
    values = {
        'center_steer': csv_data['Steering_Angle'][row],
        'left_steer': csv_data['Steering_Angle'][row] + 0.18,
        'right_steer': csv_data['Steering_Angle'][row] - 0.18,
        'center_image': preprocess(imread(csv_data['Center_Images'][row], mode='RGB')),
        'left_image': preprocess(imread(csv_data['Left_Images'][row], mode='RGB')),
        'right_image': preprocess(imread(csv_data['Right_Images'][row], mode='RGB'))
    }
    return values

def add_X_row_data(X_train, row_values, train_index):
    X_train[train_index+0] = row_values['center_image']
    X_train[train_index+1] = row_values['left_image']
    X_train[train_index+2] = row_values['right_image']
    X_train[train_index+3] = flip_image(row_values['center_image'])
    X_train[train_index+4] = flip_image(row_values['left_image'])
    X_train[train_index+5] = flip_image(row_values['right_image'])
    return X_train

def add_y_row_data(y_train, row_values, train_index):
    y_train[train_index+0] = row_values['center_steer']
    y_train[train_index+1] = row_values['left_steer']
    y_train[train_index+2] = row_values['right_steer']
    y_train[train_index+3] = flip_steer(row_values['center_steer'])
    y_train[train_index+4] = flip_steer(row_values['left_steer'])
    y_train[train_index+5] = flip_steer(row_values['right_steer'])
    return y_train
