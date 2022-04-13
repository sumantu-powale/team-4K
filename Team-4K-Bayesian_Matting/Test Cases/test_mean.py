#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:40:30 2022

@author: kaushikrachakonda
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 21:32:37 2022

@author: kaushikrachakonda
"""

import unittest
import Mean_fn
import numpy as np


class TestBayesianMatting(unittest.TestCase):

    def test_calcMeanCovariance(self):
        # Kernel_colour = np.array([[[0.2, 0.4, 0.5, 0.8, 0.1],
        #                               [0.9, 0.9, 0.7, 0.5, 0.2],
        #                               [0.2, 0.5, 0.1, 0.6, 0.4],
        #                               [0.6, 0.7, 0.4, 0.2, 0.2],
        #                               [0.5, 0.3, 0.2, 0.2, 0.7]],
                    
        #                             [[0.6, 0.2, 0.9, 0.2, 0.7],
        #                               [0.8, 0.8, 0.4, 0.4, 0.1],
        #                               [0.8, 0.6, 0.3, 0.2, 0.6],
        #                               [0.1, 0.1, 0.7, 0.5, 0.5],
        #                               [0.7, 0.4, 0.3, 0.7, 0.2]],
                    
        #                             [[0.3, 0.1, 0.1, 0.8, 0.8],
        #                               [0.4, 0.5, 0.6, 0.7, 0.1],
        #                               [0.1, 0.2, 0.5, 0.9, 0.6],
        #                               [0.3, 0.3, 0.5, 0.6, 0.5],
        #                               [0.4, 0.2, 0.8, 0.8, 0.1]]])
        
        

        f = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                  [[9, 8, 7], [6, 5, 4], [3, 2, 1]],
                  [[1, 4, 7], [2, 5, 8], [3, 6, 9]]])

        w_f = np.array([[1,1,1],[1,1,1],[1,1,1]])
        # result = calcMeanCovariance(f, w_f)
        
        # ans_1 = np.array([0.500, 0.4875, 0.3688])
        # ans_2 = np.array([0.3333, 0.4444, 0.5889])
        # Ans = np.array([ans_1, ans_2])
        
        

        result = Mean_fn.calcMeanCovariance(f, w_f)
        Ans = np.array([5, 5, 5])
        
        # self.np.testing.assert_allclose(Ans, result[0])
        np.testing.assert_allclose(Ans, result[0])
        np.testing.assert_allclose(Ans, result[1])
        np.testing.assert_allclose(Ans, result[2])

        



    # def test_isupper(self):
    #     self.assertTrue('FOO'.isupper())
    #     self.assertFalse('Foo'.isupper())

    # def test_split(self):
    #     s = 'hello world'
    #     self.assertEqual(s.split(), ['hello', 'world'])
    #     # check that s.split fails when the separator is not a string
    #     with self.assertRaises(TypeError):
    #         s.split(2)

if __name__ == '__main__':
    unittest.main()