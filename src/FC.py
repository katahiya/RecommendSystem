# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy

class MatrixFactorization(object):
    #MatrixFactorizationを実装するクラス

    def __init__(self, alpha, beta):
        '''
        Constructor
        '''
        self.u = None   #ユーザーの特徴ベクトル
        self.v = None   #アイテムの特徴ベクトル
        self.eta = None
        self.alpha = alpha
        self.beta = beta

    def __get_error(self, dataij, ui, vj):
        return dataij- numpy.dot(ui.T, vj)

    def fit(self, data, K=2):
        n = len(data)   #ユーザー数
        m = len(data[0]) #アイテム数
        count = 0   #更新回数
        self.u = numpy.random.rand(K, n)    #K×ユーザー数の行列を生成
        self.v = numpy.random.rand(K, m)    #K×アイテム数の行列を生成
        for i in range(n):
            for j in range(m):
                if data[i][j] == 0:
                    continue
                error = self.__get_error(data[i][j],self. u[:, i], self.v[:, j])
                