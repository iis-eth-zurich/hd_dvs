#*----------------------------------------------------------------------------*
#* Copyright (C) 2020 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: GPL-3.0-only                                      *
#*                                                                            *
#* This program is free software: you can redistribute it and/or modify       *
#* it under the terms of the GNU General Public License as published by       *
#* the Free Software Foundation, either version 3 of the License, or          *
#* (at your option) any later version.                                        *
#*                                                                            *
#* This program is distributed in the hope that it will be useful,            *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
#* GNU General Public License for more details.                               *
#*                                                                            *
#* You should have received a copy of the GNU General Public License          *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.     *
#*                                                                            *
#* Author:  Edoardo Mello Rella <edoardom@student.ethz.ch>                    *
#*----------------------------------------------------------------------------*

"""mlp_train.py
multi-layer perceptron implementation of training and prediction with previous feature extraction
"""
import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import scipy
import scipy.ndimage
import math
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import keras
import keras.backend as K
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import Sequential

from utils.image2vec import *

if __name__ == '__main__':

    def aee_sq(y_true, y_pred):
        return K.sum(K.square(y_pred - y_true), axis=-1)

    def aee_abs(y_true, y_pred):
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

    def aee_rel(y_true, y_pred):
        GT = K.sqrt(K.sum(K.square(y_true), axis=-1)) + K.epsilon()
        return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)) / GT


    def model(shape=(6,)):
        """
        Model initialization
        args:
            shape: shape of input features
        returns:
            model: initialized model to be trained
        """

        dense_layers = [80, 160, 130, 100, 75, 40, 12]

        model = Sequential()
        model.add(Dense(24, activation='relu', input_shape=shape, kernel_regularizer=keras.regularizers.l2(l=0.1)))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())

        for dl in dense_layers:
            model.add(Dense(dl, activation='relu', kernel_regularizer=keras.regularizers.l2(l=0.1)))
            model.add(Dropout(0.2))
            model.add(BatchNormalization())

        model.add(Dense(2, activation='linear'))
        model.compile(loss=aee_sq, optimizer=Adam(lr=0.001), metrics=[aee_abs, aee_rel])
        return model

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False,
                        help="Folder containing the data to test the trained model")
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        required=False,
                        help="Batch size used in training")
    parser.add_argument('--training_epochs',
                        type=int,
                        default=200,
                        required=False,
                        help="Number of training epochs")

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    batch_size = args.batch_size
    epochs = args.training_epochs

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    Kt              = sl_npz['K']
    D              = sl_npz['D']
    gt_poses       = sl_npz['poses']
    gt_ts          = sl_npz['gt_ts']
    gT_x           = sl_npz['Tx']
    gT_y           = sl_npz['Ty']
    gT_z           = sl_npz['Tz']
    gQ_x           = sl_npz['Qx']
    gQ_y           = sl_npz['Qy']
    gQ_z           = sl_npz['Qz']

    first_ts = cloud[0][0]
    last_ts = cloud[-1][0]
 
    gQ_y = scipy.ndimage.median_filter(gQ_y, 11)
    gT_x = scipy.ndimage.median_filter(gT_x, 11)
    gT_z = scipy.ndimage.median_filter(gT_z, 11)

    vec_image = []
    for i, t in enumerate(gt_ts):
        if i%100 == 0:
            print("build", i)
        if (t > last_ts or t < first_ts):
            continue
        sl, _ = pydvs.get_slice(cloud, idx, t, 0.2, 0, discretization)
        vec_i = VecImageCloud((260, 346), sl)
        vec_image.append(vec_i.feature_list)

    training_set = len(gt_ts)//8 * 3

    X_train = np.array(vec_image[:training_set])
    y_train = np.zeros((training_set, 2))
    y_train[:, 0] = np.array(gT_z[:training_set])
    y_train[:, 1] = np.array(gT_x[:training_set])

    X_test = np.array(vec_image[len(gt_ts)//2:])
    y_test = np.zeros((len(gt_ts)//2, 2))
    y_test[:, 0] = np.array(gT_z[len(gt_ts)//2:])
    y_test[:, 1] = np.array(gT_x[len(gt_ts)//2:])

    scaler_y = StandardScaler()
    y_t = scaler_y.fit_transform(y_train)

    scaler_x = StandardScaler()
    X_t = scaler_x.fit_transform(X_train)

    mlp_model = model(shape=(6,))
    mlp_model.fit(x=X_t, y=y_t, epochs=epochs, verbose=0, validation_split=0.1, batch_size=batch_size, shuffle=True)

    X_test = scaler_x.transform(X_test)
    prediction = mlp_model.predict(X_test)
    prediction = scaler_y.inverse_transform(prediction)
    prediction = np.array(prediction)

    #dimension rescaling
    lo_tz = scipy.ndimage.median_filter(prediction[:, 0], 21) * 40
    lo_tx = scipy.ndimage.median_filter(prediction[:, 1], 21) * 40

    gt_tx = np.array(y_test[:, 1]) * 40
    gt_tz = np.array(y_test[:, 0]) * 40

    # Compute errors
    l_gt = np.sqrt(gt_tz * gt_tz + gt_tx * gt_tx)
    l_lo = np.sqrt(lo_tz * lo_tz + lo_tx * lo_tx)
    ARPE = 0.0
    for i in range(len(gt_tz)):
        RPE_cos = (gt_tz[i] * lo_tz[i] + gt_tx[i] * lo_tx[i])
        if (l_gt[i] > 0):
            RPE_cos /= (l_gt[i] * l_lo[i])
        if (RPE_cos >  1.0): RPE_cos =  1.0
        if (RPE_cos < -1.0): RPE_cos = -1.0
        ARPE += math.acos(RPE_cos)

    print('ARPE -->', ARPE/len(gt_tz))

    K.clear_session()
    del mlp_model
