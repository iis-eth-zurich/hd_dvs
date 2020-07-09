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

"""online_model.py
training and testing on model exploiting online model update
"""
from random import seed
from random import randint

import argparse
import numpy as np
import os, sys, shutil, signal, glob, time
import scipy
import scipy.ndimage

from utils.image2vec import *
from utils.model_a import *
from utils.hv_type import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default='.',
                        required=False,
                        help="Folder containing the data to test the trained model")
    parser.add_argument('--data_type',
                        type=str,
                        default='rafe',
                        required=False,
                        help="rafe/skc/all/gradient --> types of data to use")
    parser.add_argument('--sparse',
                        type=bool,
                        default=True,
                        required=False,
                        help="determines if the methods applied to build memories are based on sparse vectors --> True for rafe and skc and False for all and gradient")
    parser.add_argument('--cdt',
                        type=bool,
                        default=False,
                        required=False,
                        help="determines if cdt has to be applied --> only applicable if sparse == True")
    parser.add_argument('--integer',
                        type=bool,
                        default=False,
                        required=False,
                        help="determines if the vector is binary or bipolar --> only applicable if sparse == False")

    args = parser.parse_args()

    print ("Opening", args.base_dir)

    sl_npz = np.load(args.base_dir + '/recording.npz')
    cloud          = sl_npz['events']
    idx            = sl_npz['index']
    discretization = sl_npz['discretization']
    K              = sl_npz['K']
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
    with open(os.path.join(args.base_dir, 'hd_vector_'+args.data_type+'.txt')) as fin:
        for k, line in enumerate(fin.readlines()):
            if (k % 100 == 0):
                print ("Training:", k)
            split_line = line.split(sep=' ')
            vec = HD_vector()
            vec.init_from_string(split_line[0])
            vec_image.append(vec)

    training_set = len(gt_ts)//8 #amount of data to be used for initial model training
    online_set = len(gt_ts)//2 #amount of data to be used for online model update
    threshold = 2.5 #certainty threshold
        
    QY_mapper = ClassMapper(0.0005, 0.0005)
    QYModel = Model_a(QY_mapper)

    TX_mapper = ClassMapper(0.002, 0.002)
    TXModel = Model_a(TX_mapper)

    TZ_mapper = ClassMapper(0.002, 0.002)
    TZModel = Model_a(TZ_mapper)

    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        
        if (i > training_set): break

        if (i % 100 == 0):
            print ("Training:", i, "/", len(gt_ts))

        if gT_z[i] < 0.005: continue

        if args.integer:
            QYModel.add_integer(vec_image[i], gQ_y[i])
            TXModel.add_integer(vec_image[i], gT_x[i])
            TZModel.add_integer(vec_image[i], gT_z[i])
        else:
            QYModel.add(vec_image[i], gQ_y[i])
            TXModel.add(vec_image[i], gT_x[i])
            TZModel.add(vec_image[i], gT_z[i])

    if args.sparse:
        if args.cdt:

            sequence = []
            for T in range(40):
                sequence.append(randint(1, 8159))
            
            QYModel.build_sparse_cdt(sequence)
            TXModel.build_sparse_cdt(sequence)
            TZModel.build_sparse_cdt(sequence)

        else:

            sequence = []
            for T in range(40):
                sequence.append(randint(1, 8159))
            
            QYModel.build_sparse()
            TXModel.build_sparse()
            TZModel.build_sparse()

    else:
        if args.integer:

            QYModel.build_integer(sequence)
            TXModel.build_integer(sequence)
            TZModel.build_integer(sequence)
        
        else:

            QYModel.build_integer(sequence)
            TXModel.build_integer(sequence)
            TZModel.build_integer(sequence)

    print ("\n\n\n------------")

    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        
        if (i < training_set): continue #necessary if the set used for training and online update is the same
        if (i > online_set): break

        if (i % 100 == 0):
            print ("Training:", i, "/", len(gt_ts))

        if args.sparse:
        
            tick = 0
            (lo, hi), [score, cl], tick = QYModel.infer_sparse_online(vec_image[i], threshold)
            if tick == 1:
                QYModel.add(vec_image[i], lo)
            
            (lo, hi), [score, cl], _ = TXModel.infer_sparse_online(vec_image[i], threshold)
            if tick == 1:
                TXModel.add(vec_image[i], lo)
            
            (lo, hi), [score, cl], _ = TZModel.infer_sparse_online(vec_image[i], threshold)
            if tick == 1:
                TZModel.add(vec_image[i], lo)

        else:
            if args.integer:
        
                tick = 0
                (lo, hi), [score, cl], tick = QYModel.infer_integer_online(vec_image[i], threshold)
                if tick == 1:
                    QYModel.add(vec_image[i], lo)
                
                (lo, hi), [score, cl], _ = TXModel.infer_integer_online(vec_image[i], threshold)
                if tick == 1:
                    TXModel.add(vec_image[i], lo)
                
                (lo, hi), [score, cl], _ = TZModel.infer_integer_online(vec_image[i], threshold)
                if tick == 1:
                    TZModel.add(vec_image[i], lo)

            else:
        
                tick = 0
                (lo, hi), [score, cl], tick = QYModel.infer_online(vec_image[i], threshold)
                if tick == 1:
                    QYModel.add(vec_image[i], lo)
                
                (lo, hi), [score, cl], _ = TXModel.infer_online(vec_image[i], threshold)
                if tick == 1:
                    TXModel.add(vec_image[i], lo)
                
                (lo, hi), [score, cl], _ = TZModel.infer_online(vec_image[i], threshold)
                if tick == 1:
                    TZModel.add(vec_image[i], lo)

    if args.sparse:
        if args.cdt:

            sequence = []
            for T in range(40):
                sequence.append(randint(1, 8159))
            
            QYModel.build_sparse_cdt(sequence)
            TXModel.build_sparse_cdt(sequence)
            TZModel.build_sparse_cdt(sequence)

        else:

            sequence = []
            for T in range(40):
                sequence.append(randint(1, 8159))
            
            QYModel.build_sparse()
            TXModel.build_sparse()
            TZModel.build_sparse()

    else:
        if args.integer:

            QYModel.build_integer(sequence)
            TXModel.build_integer(sequence)
            TZModel.build_integer(sequence)
        
        else:

            QYModel.build_integer(sequence)
            TXModel.build_integer(sequence)
            TZModel.build_integer(sequence)

    gt_tx = []
    gt_tz = []
    gt_qy = []

    lo_tx = []
    lo_tz = []
    lo_qy = []

    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue

        if i < online_set: #necessary if the training, online update and testing set are the same
            continue

        if (i % 100 == 0):
            print ("Inference:", i, "/", len(gt_ts))

        if args.sparse:
        
            (lo, hi), [score, cl] = QYModel.infer_sparse(vec_image[i])
            lo_qy.append(lo)
            
            (lo, hi), [score, cl] = TXModel.infer_sparse(vec_image[i])
            lo_tx.append(lo)
            
            (lo, hi), [score, cl] = TZModel.infer_sparse(vec_image[i])
            lo_tz.append(lo)

        else:
            if args.integer:
        
                (lo, hi), [score, cl] = QYModel.infer_integer(vec_image[i])
                lo_qy.append(lo)
                
                (lo, hi), [score, cl] = TXModel.infer_integer(vec_image[i])
                lo_tx.append(lo)
                
                (lo, hi), [score, cl] = TZModel.infer_integer(vec_image[i])
                lo_tz.append(lo)

            else:
        
                (lo, hi), [score, cl] = QYModel.infer(vec_image[i])
                lo_qy.append(lo)
                
                (lo, hi), [score, cl] = TXModel.infer(vec_image[i])
                lo_tx.append(lo)
                
                (lo, hi), [score, cl] = TZModel.infer(vec_image[i])
                lo_tz.append(lo)

        gt_tx.append(gT_x[i])
        gt_tz.append(gT_z[i])
        gt_qy.append(gQ_y[i])   

    print ("\n\n===============================")

    #dimension rescaling
    gt_qy = np.array(gt_qy) * 40
    lo_qy = scipy.ndimage.median_filter(lo_qy, 21) * 40

    gt_tx = np.array(gt_tx) * 40
    lo_tx = scipy.ndimage.median_filter(lo_tx, 21) * 40

    gt_tz = np.array(gt_tz) * 40
    lo_tz = scipy.ndimage.median_filter(lo_tz, 21) * 40

    l_gt = np.sqrt(gt_tz * gt_tz + gt_tx * gt_tx)
    l_lo = np.sqrt(lo_tz * lo_tz + lo_tx * lo_tx)

    # Compute errors
    ARPE = 0.0
    for i in range(len(gt_tz)):
        RPE_cos = (gt_tz[i] * lo_tz[i] + gt_tx[i] * lo_tx[i])
        if (l_gt[i] > 0 and l_lo[i] > 0):
            RPE_cos /= (l_gt[i] * l_lo[i])
        if (RPE_cos >  1.0): RPE_cos =  1.0
        if (RPE_cos < -1.0): RPE_cos = -1.0
        ARPE += math.acos(RPE_cos)

    print('ARPE -->', ARPE/len(gt_tz))

