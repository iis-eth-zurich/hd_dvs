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

"""sparse_creator.py
Library to produce sparse HD vectors with either SKC or RAFE from features set
"""
import numpy as np
import torch
from hv_type import HD_vector

class Sparse_encoder:
    """
    Class to crate sparse vectors
    Initialization:
    args:
        activation_dim: activation vector dimensions --> vector dimension for SKC and vector dimension / #features for RAFE
        n_feat_matrix: number of features for activation matrix --> #features in SKC and 1 for RAFE
    """
    def __init__(self, activation_dim=8160, n_feat_matrix=6, dtype=torch.float):
        use_cuda = torch.cuda.is_available()
        Sparse_encoder.device = torch.device("cuda" if use_cuda else "cpu")
        Sparse_encoder.n_feat_matrix = n_feat_matrix
        Sparse_encoder.dim = activation_dim
        Sparse_encoder.mapper = torch.randn((self.dim, self.n_feat_matrix), dtype=torch.float, device=self.device)

    def get_skc(self, feat_vec, density):
        """
        Creates an HD sparse vector using SKC algorithm
        args:
            feat_vec: vectorf with features to embed
            density: proportion of 1 in the sparse vector
        returns:
            final: initialized HD vector
        """
        final = HD_vector(dimension=self.dim)
        output = torch.reshape(torch.mm(self.mapper, torch.reshape(feat_vec, (-1, 1))), (1, -1))
        index = torch.topk(output, k=int(final.dimension*density), sorted=False)
        final.vector[index.indices] = 1
        return final

    def get_rafe(self, feat_vec, distance):
        """
        Creates an HD sparse vector using RAFE algorithm
        args:
            feat_vec: vectorf with features to embed
            distance: threshold distance to consider an element 1
        returns:
            final: initialized HD vector
        """
        n_feat = feat_vec.size(0)
        final = HD_vector(dimension=self.dim*n_feat)
        for i in range(n_feat):
            for k in range(self.dim):
                if torch.abs(feat_vec[i] - self.mapper[k]) <= distance: 
                    final.vector[i*self.dim + k] = 1
        return final

