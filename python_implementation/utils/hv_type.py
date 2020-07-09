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

"""
Declares HD_vector class
Class needed throughout the whole code to handle hyperdimensional vectors
"""
import torch
import numpy as np

VEC_DIM = 8160

class HD_vector:
    """
    Initialize instance of the class with:
    dimension: the vector dimension
    dtype: pytorch data type
    cuda_device: if multiple gpu available, the gpu to use
    binary: True if the instance of the vector is binary
    bipolar: True if the instance of the vector is bipolar
    randomized: 'random' if the new vector needs to be initialized with random values, otherwise every element will be zero
    """
    def __init__(self, dimension=VEC_DIM, dtype=torch.float, cuda_device='cuda', binary=True, bipolar=False, randomized=''):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(cuda_device if use_cuda else "cpu")
        if randomized=='random':
            self.vector = torch.FloatTensor(dimension).bernoulli_().to(self.device)
            if bipolar==True:
                self.vector[self.vector==0] = -1
        else:
            self.vector = torch.zeros(dimension, dtype=dtype).to(self.device)
        self.binary = binary
        self.num_vec = 0
        self.dimension = dimension
        self.bipolar = bipolar
        #print('device:', self.device)

    def init_from_string(self, HV_str):
        """
        reads a string and initializes the vector accordingly
        """
        btc = 0
        counttot = 0
        for i, bit in enumerate(HV_str):
            if (bit!='1' and bit!='0'):
                counttot = counttot +1
            if (bit == '1'):
                btc += 1
                self.vector[i-counttot] = 1
        if self.bipolar == True:
            self.vector[self.vector==0] = -1

    def integer_sum(self, a):
        """
        Sums non binary vectors in-place and registers that the vector contains multiple vectors
        """
        if self.binary == True:
            print('Trying to sum a purely binary vector!')
            return
        self.vector.add_(a.to(self.device))
        self.num_vec += 1

    def xor(self, a):
        """
        in-place xor operation for binary vectors
        """
        if self.binary == False:
            print('Trying to xor a nonbinary vector!')
            return
        self.vector = (torch.add(self.vector, a) == 1).float()
        if self.bipolar==True:
            self.vector[self.vector==0] = -1

    def get_resulting_vector(self):
        """
        gets the resulting vector for majority sum operation
        """
        resulting_v = HD_vector()
        result = (self.vector > self.num_vec/2).float()
        resulting_v.vector = result.type(torch.float)
        #self.num_vec = 1
        return resulting_v

    def permute(self, perx, pery):
        """
        permutes vector
        """
        permuted = self.vector[perx]
        permuted = permuted[pery]
        return permuted

    def create_permutation_x(self, num_perm):
        """
        creates permutation along x axis
        """
        self.perx = torch.IntTensor(self.dimension, num_perm)
        initializer = np.arange(self.dimension)
        initializer = torch.from_numpy(initializer)
        for i in range(num_perm):
            randomizer = torch.rand(self.dimension)
            rand_permuter = torch.randperm(self.dimension)
            if i == 0:
                self.perx[:, i] = initializer.clone()
            else:
                self.perx[:, i] = self.perx[:, i-1]
            for j in range(self.dimension):
                if randomizer[j] > 0.99:
                    temp = self.perx[j, i]
                    self.perx[j, i] = self.perx[rand_permuter[j], i]
                    self.perx[rand_permuter[j], i] = temp
        self.perx = self.perx.type(torch.long)

    def create_permutation_y(self, num_perm):
        """
        creates permutation along y axis
        """
        self.pery = torch.IntTensor(self.dimension, num_perm)
        initializer = np.arange(self.dimension)
        initializer = torch.from_numpy(initializer)
        for i in range(num_perm):
            randomizer = torch.rand(self.dimension)
            rand_permuter = torch.randperm(self.dimension)
            if i == 0:
                self.pery[:, i] = initializer.clone()
            else:
                self.pery[:, i] = self.pery[:, i-1]
            for j in range(self.dimension):
                if randomizer[j] > 0.99:
                    temp = self.pery[j, i]
                    self.pery[j, i] = self.pery[rand_permuter[j], i]
                    self.pery[rand_permuter[j], i] = temp
        self.pery = self.pery.type(torch.long)


    def load_permutations(self, base_dir):
        """
        loads previously stored permutations
        """
        namex = base_dir + 'perm_x_0.tf'
        namey = base_dir + 'perm_y_0.tf'
        self.perx0 = torch.load(namex)
        self.pery0 = torch.load(namey)
        namex = base_dir + 'perm_x_1.tf'
        namey = base_dir + 'perm_y_1.tf'
        self.perx1 = torch.load(namex)
        self.pery1 = torch.load(namey)

    def count(self):
        """
        Counts the number of ones of a binary vector
        """
        if self.binary == False:
            print('Trying to count ones of a nonbinary vector!')
            return
        tot = 0
        for i in range(self.dimension):
            if self.vector[i] == 1:
                tot += 1
        return tot

    def represent_type(self):
        """
        produces string representation of the vector grouping elements in 4 bytes groups
        """
        representation = ""
        for i in range(self.dimension):
            if i%32 == 0 and i != 0:
                representation += "_"
            if self.vector[i] == 1:
                representation += "1"
            else:
                representation += "0"
        return representation

    def dot(self, a):
        """
        computes dot product of vectors
        """
        result = torch.dot(self.vector, a.type(torch.float))
        return result

    def hv_or(self, a):
        """
        computes bitwise or operation for binary or bipolar vectors
        """
        if self.bipolar == True:
            result = torch.add(self.vector, a)
            result[result>=0] = 1
            result[result<0] = -1
        else:
            result = self.vector.add_(a)
            result[result>=1] = 1
        return result

    def shift(self, size):
        """
        evecutes cyclic shift and returns new instance of vector
        """
        result = HD_vector()
        result.vector[size:] = self.vector[:self.dimension-size]
        result.vector[:size] = self.vector[self.dimension-size:]
        return result

    def hv_and(self, a):
        """
        executes bitwise and operation on binary vectors
        """
        result = HD_vector()
        result.vector = torch.add(self.vector, a)
        result.vector[result.vector==1] = 0
        result.vector[result.vector==2] = 1
        return result
