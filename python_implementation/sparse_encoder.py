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

"""sparse_encoder.py
Extract the features from the DVS sequences and embeds them into HD sparse vectors
"""
import pyhdc
import argparse
import numpy as np
import os, sys, shutil, signal, glob, time

from utils.image2vec import *
from utils.hv_type import *
from utils.sparse_creator import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default="/outdoor_day_1/",
                        required=False,
                        help="Folder containing the DVS sequences to be converted into HD vectors")
    parser.add_argument('--density',
                        type=float,
                        default="10",
                        required=False,
                        help="Density of vectors to create for SKC in #ones per 1000-d vector")
    parser.add_argument('--distance',
                        type=float,
                        default="0.01",
                        required=False,
                        help="Threshold distance to consider an element 1 for RAFE")
    parser.add_argument('--method',
                        type=str,
                        default="rafe",
                        required=False,
                        help="rafe/skc --> method to use to obtain an HD sparse vector from the extracted features")
    args = parser.parse_args()

    # load the dataset
    X = []
    y = []

    print('Opening ' + args.base_dir)
    density = args.density / 1000.0

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

    initializer = HD_vector()
    max_feat = 0
    min_feat = 0
    name = 'hd_vector_'+args.method+'.txt'

    if args.method == "rafe":
        kanerva_creator = Sparse_encoder(activation_dim=1360, n_feat_matrix=1)
    elif args.method == "skc":
        kanerva_creator = Sparse_encoder(activation_dim=8160, n_feat_matrix=6)
    
    f = open(os.path.join(args.base_dir, name), 'w')

    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        y.append([gQ_y[i], gT_x[i], gT_z[i]])
        if (i % 25 == 0):
            print ("Training:", i, "/", len(gt_ts))

        sl, _ = pydvs.get_slice(cloud, idx, t, 0.2, 0, discretization)
        
        vec_image = VecImageCloud((260, 346), sl)
        vec_image.feature_list = (np.array(vec_image.feature_list) - 250) / 250.0
        to_encode = torch.from_numpy(vec_image.feature_list).type(dtype=torch.float).to(initializer.device)
        if args.method == "rafe":
            v = kanerva_creator.get_rafe(to_encode, args.distance)
        elif args.method == "skc":
            v = kanerva_creator.get_skc(to_encode, density)

        s = v.represent_type()

        for vel in y[i]:
            s += " " + str(vel)
        f.write(s + "\n")

    f.close()