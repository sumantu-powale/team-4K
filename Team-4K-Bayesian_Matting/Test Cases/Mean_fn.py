#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:18:31 2022

@author: kaushikrachakonda
"""
import numpy as np
from PIL import Image


def calcMeanCovariance(pixel_input, weight_input):
    A = weight_input.shape
    W = np.sum(weight_input)
    avg = np.array(
        [np.sum(np.multiply(pixel_input[0], weight_input)) / W,
          np.sum(np.multiply(pixel_input[1], weight_input)) / W,
          np.sum(np.multiply(pixel_input[2], weight_input)) / W])
    covar = np.zeros((3, 3))
    for i in range(A[0]):
        for j in range(A[1]):
            if weight_input[i, j] != 0:
                pixel_single = np.array([pixel_input[:, i, j]])
                a = pixel_single - avg
                b = np.transpose(a)
                # Mean = np.dot(b, a)
                Mean = a*b
                Y = weight_input[i, j] * Mean
                covar = covar + Y
    covar = covar / W
    return avg, covar
    # return covar

##################


f = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]])

w_b = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])

# Kernel_colou2 = np.zeros((3, 3, 3))
# Kernel_colou2[:, :, 0] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Kernel_colou2[:, :, 1] = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
# Kernel_colou2[:, :, 2] = np.array([[1, 4, 7], [2, 5, 8], [3, 6, 9]])
# w_f2 = np.zeros((3, 3, 3))
# w_f2[:, :, 0] = np.array([[1,1,1],[1,1,1],[1,0,0]])
# w_f2[:, :, 1] = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
# w_f2[:, :, 2] = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
# w_b2 = np.zeros((3, 3, 3))
# w_b2[:, :, 0] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
# w_b2[:, :, 1] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
# w_b2[:, :, 2] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
#         #self.w_f2 = np.array([[1,1,1],[1,1,1],[1,0,0]])
#         #self.w_b2 = np.array([[0,0,0],[0,0,0],[0,1,1]])
#         #print(self.w_f2)
# F2 = np.multiply(Kernel_colou2, w_f2)
# B2 = np.multiply(Kernel_colou2, w_b2) 
        
# F2_bar = np.array([[4], [6], [4.28]])
# B2_bar = np.array([[8.5], [1.5], [7.5]])
        
result1, result2 = calcMeanCovariance(f, w_b)
print(result2)