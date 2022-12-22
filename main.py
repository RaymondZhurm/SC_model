# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import  optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau,CSVLogger,LearningRateScheduler,EarlyStopping
from tensorflow.keras.layers import Input,Dense, Lambda,Conv1D,Conv2DTranspose, LeakyReLU,Activation,Flatten,Reshape, BatchNormalization
from tensorflow.python.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from pymatgen import Composition
import pandas as pd
import utils
from utils import *
import featurizer
from featurizer import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import ensemble
from sklearn import linear_model
from sklearn import neural_network
import matplotlib
import seaborn as sns
from sklearn import decomposition
import argparse
import sys
import warnings


warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='SC_model'
                                             '')
parser.add_argument('--rp', choices=['include_ftcp', 'exclude_ftcp','atomic'],
                    default='exclude_ftcp')


def main():
    # set number of elements and sites limit for the model
    num_ele = 3
    num_sites = 20
    args = parser.parse_args(sys.argv[1:])


    # read ternary compound data into dataframe
    df = pd.read_pickle('data/df.pkl')
    print('---------Building Input Data---------------')
    print('')
    if args.rp == 'exclude_ftcp':
        Crystal = crystal_represent_2(df, num_ele, num_sites)
    elif args.rp == 'atomic':
        Crystal = atomic_represent(df,num_ele, num_sites)
    else:
        Crystal = crystal_represent(df, num_ele, num_sites)

    X = np.stack(Crystal, axis=0)
    X = pad(X, 2)
    X, scaler_x = minmax(X)
    X.shape
    Y = df[['icsd_check']].values
    sup_dim = 1
    scaler_y_un = MinMaxScaler()
    scaler_y_l = MinMaxScaler()
    Y[:, :sup_dim] = scaler_y_un.fit_transform(Y[:, :sup_dim])


    # print input shape
    print('---------Printing Input Shape---------------')
    print(X.shape, Y.shape)

    # split data into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    print('')


    # define dimensions of the NN
    print('---------Building NN---------------')
    input_dim = X_train.shape[1]
    channel_dim = X_train.shape[2]
    sample_dim = y_train.shape[1]

    latent_dim = 256

    max_filter = 128

    strides = [2, 2, 1]
    kernel = [5, 3, 3]


    # build NN
    def nn():
        K.clear_session()
        x = Input(shape=(input_dim, channel_dim,))

        ####  Encoder crystal information into latent space ####
        en0 = Conv1D(max_filter // 4, kernel[0], strides=strides[0], padding='SAME')(x)
        en0 = BatchNormalization()(en0)
        en0 = LeakyReLU(0.2)(en0)
        en1 = Conv1D(max_filter // 2, kernel[1], strides=strides[1], padding='SAME')(en0)
        en1 = BatchNormalization()(en1)
        en1 = LeakyReLU(0.2)(en1)
        en2 = Conv1D(max_filter, kernel[2], strides=strides[2], padding='SAME')(en1)
        #    en2 = MaxPooling1D(2)(en2)
        en2 = BatchNormalization()(en2)
        en2 = LeakyReLU(0.2)(en2)
        en3 = Flatten()(en2)
        en4 = Dense(1024, activation='relu')(en3)
        #    en5 = Dense(max_filter,activation = 'sigmoid')(en4)
        #    en6= Multiply()([en2,en5])
        #    en7 = GlobalAveragePooling1D()(en6)

        z_mean = Dense(latent_dim, activation='linear')(en4)

        ####  Linear model from latent space to desired property ####
        de0 = Activation('relu')(z_mean)
        de1_un = Dense(128, activation="relu", kernel_regularizer='l2')(de0)
        de1_un = Dense(32, activation="relu", kernel_regularizer='l2')(de1_un)
        y_predict_sup = Dense(sup_dim, activation='sigmoid', kernel_regularizer='l2')(de1_un)

        model = Model(x, y_predict_sup)

        return model

    forward = nn()
    forward.summary()
    print('')


    # model training
    print('---------Training NN---------------')

    CSV = CSVLogger('model_result/training_log.csv', append=True)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.3,
                                  patience=10, min_lr=4e-6, verbose=1,)

    ES = EarlyStopping(patience=50, verbose=1, restore_best_weights=True)

    forward.compile(optimizer=Adam(lr=8e-5),
                    loss='binary_crossentropy', )
    forward.fit(x=X_train, y=y_train, shuffle=True,
                batch_size=1024, epochs=100, callbacks=[reduce_lr, ES, CSV],  # CSV, LRS,
                validation_data=(X_test, y_test),
                initial_epoch=0)
    print('')

    print('---------Printing Training Set Result---------------')
    print('Default Threshold is 0.5')
    # sns.set_style("whitegrid", {'axes.grid': False})
    plot_confusion_matrix(forward, X_train, y_train, 'TrainCM')
    print('')

    print('---------Printing Test Set Result---------------')
    print('Default Threshold is 0.5')
    # sns.set_style("whitegrid", {'axes.grid': False})
    plot_confusion_matrix(forward, X_test, y_test, 'TestCM')
    print('')

    print('---------End---------------')


if __name__ == '__main__':
    main()