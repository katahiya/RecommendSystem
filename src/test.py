# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from FC import MatrixFactorization

if __name__ == '__main__':
    a = np.array([[1, 2, 0, 4],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4],
                  ])
    MF = MatrixFactorization(eta=0.5, alpha=0.1, beta=10, lam=10)
    MF.fit(a)
    print(MF.test())
