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

"""model_a.py
classes to handle training and testing of different HD methods (dense, sparse, integer)
"""

import numpy as np
import os, sys, shutil, signal, glob, time
from scipy.stats import entropy
from scipy.ndimage import median_filter, uniform_filter
from numpy import linalg as LA
from utils.image2vec import *
from utils.hv_type import *


class Memory_a:

    def __init__(self, bipolar=False):
        self.masked_vectors = []
        self.vcount = 0
        self.m = HD_vector(bipolar=bipolar)
        self.density = 0

    def add(self, v):
        """
        adds new vector to memory
        args:
            v: vector to add
        """
        self.vcount += 1

        masked_v = HD_vector()
        masked_v.xor(v.vector)

        self.masked_vectors.append(masked_v)

    def add_integer(self, v):
        """
        adds new vector to memory
        args:
            v: vector to add
        """
        self.vcount += 1

        masked_v = HD_vector(bipolar=True)
        masked_v.xor(v.vector)

        self.masked_vectors.append(masked_v)

    def build(self, to_adjust=-1):
        """
        builds the memory with the vectors previously stored for each velocity
        for HD binary vectors this is a majority sum operation
        args:
            to_adjust: number of vectors to use for majority sum (usually one more than the stored vectors if their number is even)
        """
        self.m.xor(self.m.vector)
        if (self.vcount == 0):
            self.m = HD_vector(randomized='random')
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = HD_vector(randomized='random') 
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors
        if (len(csum_vectors) == 1):
            self.m.xor(csum_vectors[0].vector)
            return

        self.m.vector = self.m.vector.type(torch.float)
        self.m.binary = False

        for v in csum_vectors:
            self.m.integer_sum(v.vector)

        self.m = self.m.get_resulting_vector()

    def build_sparse(self, to_adjust):
        """
        builds the memory with the vectors previously stored for each velocity
        for HD sparse vectors this is an OR operation
        args:
            to_adjust: number of vectors to use for majority sum (usually one more than the stored vectors if their number is even)
        """
        self.m = HD_vector()
        if (self.vcount == 0):
            self.m = HD_vector(randomized='random')
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = HD_vector(randomized='random') 
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors

        for v in csum_vectors:
            self.m.hv_or(v.vector)

        self.density = self.m.count()
        return

    def build_sparse_cdt(self, to_adjust, sequence, dimension=8160):
        """
        builds the memory with the vectors previously stored for each velocity
        for HD sparse vectors this is an OR operation and CDT is applied to the result
        args:
            to_adjust: number of vectors to use for majority sum (usually one more than the stored vectors if their number is even)
        """
        self.m = HD_vector()
        if (self.vcount == 0):
            self.m = HD_vector(randomized='random')
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = HD_vector(randomized='random') 
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors

        for v in csum_vectors:
            self.m.hv_or(v.vector)

        self.density = self.m.count()
        if self.density > dimension/11 and self.density < dimension/9:
            resulting = HD_vector()
            for j, i in enumerate(sequence):
                if j > 30: continue
                shifted = self.m.shift(i)
                resulting.vector = resulting.hv_or(shifted.vector)
            result = self.m.hv_and(resulting.vector)
            self.m.vector = result.vector
            self.density = self.m.count()
        elif self.density > dimension/9:
            resulting = HD_vector()
            for j, i in enumerate(sequence):
                if j > 8: continue
                shifted = self.m.shift(i)
                resulting.vector = resulting.hv_or(shifted.vector)
            result = self.m.hv_and(resulting.vector)
            self.m.vector = result.vector
            self.density = self.m.count()

        return

    def build_integer(self, to_adjust=-1):
        """
        builds the memory with the vectors previously stored for each velocity
        for HD integer vectors this is a sum operation
        args:
            to_adjust: number of vectors to use for majority sum (usually one more than the stored vectors if their number is even)
        """
        self.m = HD_vector(bipolar=True)
        if (self.vcount == 0):
            self.m = HD_vector(randomized='random', bipolar=True)
            return

        random_vectors = []
        if (to_adjust > self.vcount):
            x = HD_vector(randomized='random', bipolar=True) 
            random_vectors.append(x)

        csum_vectors = self.masked_vectors + random_vectors

        self.m.vector = self.m.vector.type(torch.float)
        self.m.binary = False

        for v in csum_vectors:
            self.m.integer_sum(v.vector)

        self.m.vector = self.m.vector.type(torch.float)
        self.m.vector = self.m.vector * 90 / torch.norm(self.m.vector)

        return

    def find(self, v):
        """
        returns the similarity score between the inference vector and the one stored in memory
        for HD binary vectors it is the Hamming distance
        args:
            v: inference vector
        """
        mem_test = HD_vector()
        mem_test.xor(self.m.vector)
        mem_test.xor(v.vector)

        min_score = self.m.count()
        min_id = 0
        
        return min_score, min_id

    def find_integer(self, v):
        """
        returns the similarity score between the inference vector and the one stored in memory
        for HD integer vectors it is the dot product
        args:
            v: inference vector
        """
        score = 0
        score = self.m.dot(v.vector)
        m_id = 0
        return score, m_id

    def find_sparse(self, v):
        """
        returns the similarity score between the inference vector and the one stored in memory
        for HD sparse binary vectors it is the overlap
        args:
            v: inference vector
        """
        score = 0
        score = self.m.dot(v.vector)
        m_id = 0
        return score, m_id


class Model_a:

    def __init__(self, m):
        self.cl_mapper = m
        self.bins = {}
        self.infer_db = []
           
    def add(self, vec_image, val):
        """
        adds new vectors to the memory corresponding to the current velocity
        args:
            vec_image: hd vector corresponding to the current seqence
            val: current ground truth velocity
        """
        classes = self.cl_mapper.get_class(val)
        for cl in classes:
            if cl not in self.bins.keys():
                self.bins[cl] = Memory_a()

            self.bins[cl].add(vec_image)

    def add_integer(self, vec_image, val):
        """
        adds new vectors to the memory corresponding to the current velocity (function for HD integer vectors)
        args:
            vec_image: hd vector corresponding to the current seqence
            val: current ground truth velocity
        """
        classes = self.cl_mapper.get_class(val)
        for cl in classes:
            if cl not in self.bins.keys():
                self.bins[cl] = Memory_a(bipolar=True)

            self.bins[cl].add_integer(vec_image)
    
    def build(self):
        """
        builds the model computing the memories for each velocity --> for HD binary vectors
        """
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        for i, cl in enumerate(sorted(self.bins.keys())):
            self.bins[cl].build(to_adjust)

    def build_integer(self):
        """
        builds the model computing the memories for each velocity --> for HD integer vectors
        """
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        for i, cl in enumerate(sorted(self.bins.keys())):
            self.bins[cl].build_integer(to_adjust)

    def build_sparse_cdt(self, sequence):
        """
        builds the model computing the memories for each velocity --> for HD sparse vectors with CDT
        """
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        for i, cl in enumerate(sorted(self.bins.keys())):
            self.bins[cl].build_sparse_cdt(to_adjust, sequence)

    def build_sparse(self):
        """
        builds the model computing the memories for each velocity --> for HD sparse vectors
        """
        to_adjust = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount > to_adjust): to_adjust = self.bins[cl].vcount
        if (to_adjust % 2 == 0): to_adjust += 1
        for i, cl in enumerate(sorted(self.bins.keys())):
            self.bins[cl].build_sparse(-1)

    def infer(self, vec_image):
        """
        finds the vectors with highest similarity to the inference vector --> for HD binary vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
        returns:
            sequence of most likely inferences
        """
        clusters = []
        scores = []

        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                continue
            
            score, id_ = self.bins[cl].find(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        scores -= np.min(scores)
        scores /= np.max(scores)
        scores = 1 - scores

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]

    def infer_integer(self, vec_image):
        """
        finds the vectors with highest similarity to the inference vector --> for HD integer vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
        returns:
            sequence of most likely inferences
        """
        clusters = []
        scores = []

        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                continue

            score, id_ = self.bins[cl].find_integer(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        scores -= np.min(scores)
        scores /= np.max(scores)

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]

    def infer_integer_online(self, vec_image, threshold):
        """
        finds the vectors with highest similarity to the inference vector --> for HD integer vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
            threshold: certainty level to determine whether to add the predicted element to the model or not
        returns:
            sequence of most likely inferences
            binary variable telling if the inference certainty was above the threshold
        """
        clusters = []
        scores = []
        tot = 0

        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                tot += 1
                continue

            score, id_ = self.bins[cl].find_integer(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        tick = 0
        probs = np.abs(scores - np.amin(scores))
        probs = probs / float(np.sum(probs)) * (len(self.bins)-tot)
        if np.amax(probs) > threshold:
            tick = 1
        scores -= np.min(scores)
        scores /= np.max(scores)

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1], tick

    def infer_online(self, vec_image, threshold):
        """
        finds the vectors with highest similarity to the inference vector --> for HD binary vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
            threshold: certainty level to determine whether to add the predicted element to the model or not
        returns:
            sequence of most likely inferences
            binary variable telling if the inference certainty was above the threshold
        """
        clusters = []
        scores = []
        tot = 0

        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                tot += 1
                continue

            score, id_ = self.bins[cl].find(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        tick = 0
        probs = np.abs(scores - np.amin(scores))
        probs = probs / float(np.sum(probs)) * (len(self.bins)-tot)
        if np.amax(probs) > threshold:
            tick = 1
        scores -= np.min(scores)
        scores /= np.max(scores)

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1], tick

    def infer_sparse(self, vec_image):
        """
        finds the vectors with highest similarity to the inference vector --> for HD sparse vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
        returns:
            sequence of most likely inferences
        """
        clusters = []
        scores = []
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                continue

            score, id_ = self.bins[cl].find_sparse(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        scores -= np.min(scores)
        if np.max(scores) != 0:
            scores /= np.max(scores)

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1]

    def infer_sparse_online(self, vec_image, threshold):
        """
        finds the vectors with highest similarity to the inference vector --> for HD sparse vectors
        args:
            vec_image: inference vector corresponding to the observed sequence
            threshold: certainty level to determine whether to add the predicted element to the model or not
        returns:
            sequence of most likely inferences
            binary variable telling if the inference certainty was above the threshold
        """
        clusters = []
        scores = []
        tot = 0
        for i, cl in enumerate(sorted(self.bins.keys())):
            if (self.bins[cl].vcount <= 7):
                tot+=1
                continue

            score, id_ = self.bins[cl].find_sparse(vec_image)
  
            clusters.append(cl)
            scores.append(score)

        scores = np.array(scores, dtype=np.float)
        tick = 0
        probs = np.abs(scores)
        probs = probs / float(np.sum(probs)) * (len(self.bins)-tot)
        if np.amax(scores) > threshold:
            tick = 1
        scores -= np.min(scores)
        scores /= np.max(scores)

        self.infer_db.append(scores)
        result = sorted(zip(scores, clusters))

        return self.cl_mapper.get_val_range([result[-1][1]]), result[-1], tick

