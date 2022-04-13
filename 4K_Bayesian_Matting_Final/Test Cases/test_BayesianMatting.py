#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:32:37 2022

@author: kaushikrachakonda
"""

import unittest
import BayesianMatting
import numpy as np


class TestBayesianMatting(unittest.TestCase):
    
    def setUp(self):
        """
        # print('setUp \n')
        self.Kernel_colour_1 = np.array([[[0.2, 0.4, 0.5, 0.8, 0.1],
                                      [0.9, 0.9, 0.7, 0.5, 0.2],
                                      [0.2, 0.5, 0.1, 0.6, 0.4],
                                      [0.6, 0.7, 0.4, 0.2, 0.2],
                                      [0.5, 0.3, 0.2, 0.2, 0.7]],
                    
                                    [[0.6, 0.2, 0.9, 0.2, 0.7],
                                      [0.8, 0.8, 0.4, 0.4, 0.1],
                                      [0.8, 0.6, 0.3, 0.2, 0.6],
                                      [0.1, 0.1, 0.7, 0.5, 0.5],
                                      [0.7, 0.4, 0.3, 0.7, 0.2]],
                    
                                    [[0.3, 0.1, 0.1, 0.8, 0.8],
                                      [0.4, 0.5, 0.6, 0.7, 0.1],
                                      [0.1, 0.2, 0.5, 0.9, 0.6],
                                      [0.3, 0.3, 0.5, 0.6, 0.5],
                                      [0.4, 0.2, 0.8, 0.8, 0.1]]])
        """
        self.Kernel_colour_1 = np.zeros((5, 5, 3))
        self.Kernel_colour_1[:, :, 0] = np.array([[0.2, 0.4, 0.5, 0.8, 0.1],
                                                [0.9, 0.9, 0.7, 0.5, 0.2],
                                                [0.2, 0.5, 0.1, 0.6, 0.4],
                                                [0.6, 0.7, 0.4, 0.2, 0.2],
                                                [0.5, 0.3, 0.2, 0.2, 0.7]])
        self.Kernel_colour_1[:, :, 1] = np.array([[0.6, 0.2, 0.9, 0.2, 0.7],
                                                [0.8, 0.8, 0.4, 0.4, 0.1],
                                                [0.8, 0.6, 0.3, 0.2, 0.6],
                                                [0.1, 0.1, 0.7, 0.5, 0.5],
                                                [0.7, 0.4, 0.3, 0.7, 0.2]])

        self.Kernel_colour_1[:, :, 2] = np.array([[0.3, 0.1, 0.1, 0.8, 0.8],
                                                [0.4, 0.5, 0.6, 0.7, 0.1],
                                                [0.1, 0.2, 0.5, 0.9, 0.6],
                                                [0.3, 0.3, 0.5, 0.6, 0.5],
                                                [0.4, 0.2, 0.8, 0.8, 0.1]])
        """
        self.Kernel_colour_1 = self.Kernel_colour_1.reshape(-1, 3)
        #
        #
        self.w_f1 = np.array(           [[[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0]],

                                       [[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0]],

                                       [[1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [1, 1, 0, 0, 0]]])
        self.w_f1 = self.w_f1.reshape(-1, 3)
        print(self.w_f1.shape)
        self.w_b1 = np.array([[[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1]],

                              [[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1]],

                              [[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1],
                               [0, 0, 1, 1, 1]]])
        self.w_b1 = self.w_b1.reshape(-1, 3)
        """
        #self.w_b1 = self.w_b1.reshape(5, 5, 3)
        self.w_f1 = np.zeros((5, 5, 3))
        self.w_b1 = np.zeros((5, 5, 3))
        self.w_f1[:, :, 0] = np.array([[1, 1, 1, 1, 1], 
                                       [1, 1, 1, 1, 1], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0]])
        self.w_f1[:, :, 1] = np.array([[1, 1, 1, 1, 1], 
                                       [1, 1, 1, 1, 1], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0]])
        self.w_f1[:, :, 2] = np.array([[1, 1, 1, 1, 1], 
                                       [1, 1, 1, 1, 1], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0], 
                                       [1, 1, 0, 0, 0]])
        self.w_b1[:, :, 0] = np.array([[0, 0, 0, 0, 0], 
                                       [0, 0, 0, 0, 0], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1]])
        self.w_b1[:, :, 1] = np.array([[0, 0, 0, 0, 0], 
                                       [0, 0, 0, 0, 0], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1]])
        self.w_b1[:, :, 2] = np.array([[0, 0, 0, 0, 0], 
                                       [0, 0, 0, 0, 0], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1], 
                                       [0, 0, 1, 1, 1]])
        #print(self.w_f1.shape)
        self.F1 = np.multiply(self.Kernel_colour_1, self.w_f1)
        self.B1 = np.multiply(self.Kernel_colour_1, self.w_b1)
        #print(self.F1)
        self.F1_bar = np.array([[0.5], [0.4875], [0.3687]])
        self.B1_bar = np.array([[0.3333], [0.4444], [0.5888]])
        
        self.Cov1_f = np.array([[ 0.06125 , -0.00125 ,  0.018125],
                        [-0.00125 ,  0.076094,  0.000234],
                        [ 0.018125,  0.000234,  0.057148]])
        

        
        self.Cov1_b = np.array([[ 0.037778, -0.013704, -0.014074],
                            [-0.013704,  0.035802,  0.006049],
                            [-0.014074,  0.006049,  0.049877]])
        """
        self.Kernel_colou2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                  [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]])
        """
        self.Kernel_colou2 = np.zeros((3, 3, 3))
        self.Kernel_colou2[:, :, 0] = np.array([[1, 2, 3], 
                                                [4, 5, 6], 
                                                [7, 8, 9]])
        self.Kernel_colou2[:, :, 1] = np.array([[9, 8, 7], 
                                                [6, 5, 4], 
                                                [3, 2, 1]])
        self.Kernel_colou2[:, :, 2] = np.array([[1, 4, 7], 
                                                [2, 5, 8],
                                                [3, 6, 9]])
        self.w_f2 = np.zeros((3, 3, 3))
        self.w_f2[:, :, 0] = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
        self.w_f2[:, :, 1] = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
        self.w_f2[:, :, 2] = np.array([[1, 1, 1], [1, 1, 1], [1, 0, 0]])
        self.w_b2 = np.zeros((3, 3, 3))
        self.w_b2[:, :, 0] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        self.w_b2[:, :, 1] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        self.w_b2[:, :, 2] = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
        #self.w_f2 = np.array([[1,1,1],[1,1,1],[1,0,0]])
        #self.w_b2 = np.array([[0,0,0],[0,0,0],[0,1,1]])
        #print(self.w_f2)
        self.F2 = np.multiply(self.Kernel_colou2, self.w_f2)
        self.B2 = np.multiply(self.Kernel_colou2, self.w_b2) 
        
        self.F2_bar = np.array([[4], [6], [4.28]])
        self.B2_bar = np.array([[8.5], [1.5], [7.5]])

    def test_getFbarBbar(self):
        
        # print('\n Test Mean function: \n')
        result1 = BayesianMatting.getFbarBbar(self.F1, self.B1, 
                                              self.w_f1, self.w_b1)
        # print(result1)
        # Ans_f = np.array([[4], [6], [4.28]])
        # Ans_b = np.array([[8.5], [1.5], [7.5]])
        
        Ans_f1 = np.array([[0.5], [0.4875], [0.3687]])
        Ans_b1 = np.array([[0.3333], [0.4444], [0.5888]])
        Ans1 = np.array([Ans_f1,Ans_b1])

        # self.assertAlmostEqual(result[0], Ans)
        # np.testing.assert_array_almost_equal_nulp(Ans[0], result[0])
        np.testing.assert_allclose(Ans1[0], result1[0], rtol = 1e-02)
                
        
        
        Ans_f2 = np.array([[4], [6], [4.28]])
        Ans_b2 = np.array([[8.5], [1.5], [7.5]])
        Ans2 = np.array([Ans_f2,Ans_b2])
        result2 = BayesianMatting.getFbarBbar(self.F2, self.B2, 
                                              self.w_f2, self.w_b2)
        # print(result2)
        # self.assertAlmostEqual(result[0], Ans)
        # np.testing.assert_array_almost_equal_nulp(Ans[0], result[0])
        np.testing.assert_allclose(Ans2, result2, rtol = 1e-02)
        # print ('Mean fn pass')
  

    def test_getFcovBcov(self):
        
        result_cov1 = BayesianMatting.getFcovBcov(self.F1, self.B1, 
                                                  self.F1_bar, self.B1_bar, 
                                                  self.w_f1, self.w_b1)
        # print(result_cov1)
        
        Cov1_f = np.array([[ 0.06125 , -0.00125 ,  0.018125],
                        [-0.00125 ,  0.076094,  0.000234],
                        [ 0.018125,  0.000234,  0.057148]])
        

        
        Cov1_b = np.array([[ 0.037778, -0.013704, -0.014074],
                            [-0.013704,  0.035802,  0.006049],
                            [-0.014074,  0.006049,  0.049877]])
        Cov1 = np.array([Cov1_f, Cov1_b])
        
        np.testing.assert_allclose(Cov1, result_cov1, rtol = 1e-02)
       
        
        
        Cov2_f = np.array([[ 4,          -4,          1.71428571],
                           [-4,           4,         -1.71428571],
                           [ 1.71428571, -1.71428571,  5.63265306]])
        
        Cov2_b = np.array([[ 0.25, -0.25,  0.75],
                           [-0.25,  0.25, -0.75],
                           [ 0.75, -0.75,  2.25]])
        
        Cov2 = np.array([Cov2_f, Cov2_b])

        
        result_cov2 = BayesianMatting.getFcovBcov(self.F2, self.B2, 
                                                  self.F2_bar, self.B2_bar, 
                                                  self.w_f2, self.w_b2)
        # print(result2)
        
        np.testing.assert_allclose(Cov2, result_cov2, rtol = 1e-02)

    # def test_matrixequation(self):
        
    #     alpha_bar = np.array([0.1, 0,5, 0,7])
    #     sigma_C_1 = 0.1
    #     like_threshold = 5e-7
    #     max_it = 100
        
    #     result1_matrix = BayesianMatting.matrixequation(self.Kernel_colour_1, 
    #                                                     alpha_bar, self.F1_bar,
    #                                                     self.B1_bar, 
    #                                                     self.Cov1_f, 
    #                                                     self.Cov1_b, sigma_C_1, 
    #                                                     like_threshold, max_it)
        
        
        
        
        

if __name__ == '__main__':
    unittest.main()