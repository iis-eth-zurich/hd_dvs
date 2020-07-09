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

"""image_encoder.py
This file executes the embedding of DVS sequences into HD vectors
Specifically it leverages the computation of the time image and the gradient image and proceeds to their embedding into vectors
In both cases it takes the selected image and encodes each pixel into a vector and later bundles them together after permuting them according to the position of the corresponding pixel
"""


import pyhdc
import argparse
import numpy as np
import os, sys, shutil, signal, glob, time

try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

from utils.image2vec import *
from utils.hv_type import *


def i2v_baseline_sum(image, vmap, perx, pery, propx, propy):
    """
    Executes the actual embedding
    Each pixel is embedding into a vector depending on its value, it is permuted according to its x and y position and then bundled with all other vectors
    """
    tot = HD_vector(dtype=torch.float, binary=False)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            val = image[i, j]
            if (val < 1):
                continue

            # permute
            if (i >= 0 and j >= 0):
                tot.integer_sum(vmap[val - 1].permute(perx[:, i*propx], pery[:, j*propy]))

    x = HD_vector()
    x = tot.get_resulting_vector()
    del tot
    return x


def img2vec(img, vmap, perx0, pery0, perx1, pery1, propx=1, propy=1):
    """
    The function does sanity check on the images given as input and handles single and multiple (2) channels images
    """
    # Make sure it is the grayscale image
    image = img
    if (len(img.shape) > 2 and (img.shape[2] == 3 or img.shape[2] == 1)):
        image = img[:,:,1]
        x = HD_vector()
        x = i2v_baseline_sum(image, vmap, perx0, pery0, propx, propy)
        print('done 1 iteration')
    elif (len(img.shape) > 2 and img.shape[2] == 2):
        image0 = img[:, :, 0]
        image1 = img[:, :, 1]
        z = i2v_baseline_sum(image0, vmap, perx0, pery0, propx, propy)
        z.binary=False
        z.vector = z.vector.type(torch.float)
        y = i2v_baseline_sum(image1, vmap, perx1, pery1, propx, propy)
        print('done 1 iteration')
        z.integer_sum(y.vector)
        x = HD_vector()
        x = y.get_resulting_vector()
    else:
        print ("Unsupported image size: ", image.shape)
    if (len(vmap) != 256):
        print ("vmap should have length of 256, not", len(vmap))

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir',
                        type=str,
                        default="/home/",
                        required=True,
                        help="Folder containing the DVS sequences to be converted into HD vectors")
    parser.add_argument('--perm_dir',
                        type=str,
                        default="./precomputed_data/",
                        required=True,
                        help="folder containing permutations map in order to encode bit position into HD vectors")
    parser.add_argument('--method',
                        type=str,
                        default="all",
                        required=False,
                        help="all/gradient --> it either encodes the plain image obtained projecting the DVS sequence or the one obtained after computing the gradients from the image")
    args = parser.parse_args()
    method = args.method

    grad = []
    img = []
    y = []

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

    # vector map
    print ("\n=======================\nReading vmap...")
    vmap = []
    f = open('./precomputed_data/vmap.txt', 'r')
    for i, line in enumerate(f.readlines()):
        v = HD_vector()
        v.init_from_string(line)
        vmap.append(v)
    f.close()

    print ("Read", len(vmap), "vectors")

    permuter = HD_vector()
    permuter.load_permutations(args.perm_dir)
    perx0 = permuter.perx0.to(permuter.device)
    pery0 = permuter.pery0.to(permuter.device)
    perx1 = permuter.perx1.to(permuter.device)
    pery1 = permuter.pery1.to(permuter.device)

    # convert to vectors
    ref_vec = HD_vector()
    name = 'hd_vector_' + method + '.txt'
    f = open(os.path.join(args.base_dir, name), 'w')
    for i, t in enumerate(gt_ts):
        if (t > last_ts or t < first_ts):
            continue
        y.append([gQ_y[i], gT_x[i], gT_z[i]])
        if (i % 100 == 0):
            print ("Training:", i, "/", len(gt_ts))

        sl, _ = pydvs.get_slice(cloud, idx, t, 0.2, 0, discretization)
        
        vec_image = VecImageCloud((260, 346), sl)

        if method == 'all':
            vec_image.dvs_img = vec_image.dvs_img * 255 / 0.2
            vec_image.dvs_img = vec_image.dvs_img.astype(int)
            img.append(vec_image.dvs_img)
            v = img2vec(img[i], vmap, perx0, pery0, perx1, pery1)

        elif method == 'gradient':
            vec_image.dgrad[vec_image.dgrad!=0] = (vec_image.dgrad[vec_image.dgrad!=0] + 1.5) * 255 / 3.0
            vec_image.dgrad[vec_image.dgrad>255] = 255
            vec_image.dgrad = vec_image.dgrad.astype(int)
            grad.append(vec_image.dgrad)
            v = img2vec(grad[i], vmap, perx0, pery0, perx1, pery1)

        s = v.represent_type()

        for vel in y[i]:
            s += " " + str(vel)
        f.write(s + "\n")

    f.close()
