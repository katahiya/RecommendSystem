# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy
from math import fabs

class MatrixFactorization(object):
    #MatrixFactorizationを実装するクラス

    def __update_eta(self, count):
        self.eta = 1 / (self.alpha*(count+self.beta))

    def __init__(self, eta, alpha, beta, lam):
        '''
        Constructor
        '''
        self.u = None   #ユーザーの特徴ベクトル
        self.v = None   #アイテムの特徴ベクトル
        self.alpha = alpha
        self.beta = beta
        self.eta = None
        self.__update_eta(0)     #eta = 1/(alpha(更新回数+beta))の式で更新
        self.lam = lam  #行列の更新に使用

    def __get_error(self, dataij, ui, vj):
        return dataij- numpy.dot(ui.T, vj)

    def __get_f(self, data):
    #関数fを求める
        f = 0;
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == 0:
                    continue
                f += 0.5 * pow(self.__get_error(data[i][j], self.u[:, i], self.v[:, j]), 2)
        f += 0.5 * self.lam * (numpy.linalg.norm(self.u)+numpy.linalg.norm(self.v))
        return f

    def __get_f_error(self, data):
    #fの値を更新してその誤差を取得
        old_f = self.f
        self.f = self.__get_f(data)
        return fabs(self.f - old_f)

    def __update_lambda(self, count):
        self.lam = 1 / count

    def fit(self, data, K=30, steps=5000, criteria=0.0001):
        n = len(data)   #ユーザー数
        m = len(data[0]) #アイテム数
        count = 0   #更新回数
        self.u = numpy.random.rand(K, n)    #K×ユーザー数の行列を生成
        self.v = numpy.random.rand(K, m)    #K×アイテム数の行列を生成
        self.f = self.__get_f(data)
        #stepsの回数を上限に行列を修正
        for step in range(steps):
            for i in range(n):
                for j in range(m):
                    if data[i][j] == 0:
                        continue
                    ui = self.u[:, i].copy()
                    vj = self.v[:, j].copy()
                    error = self.__get_error(data[i][j], ui, vj)
                    #パラメータの更新
                    self.u[:, i] -= self.eta * (-error*vj+self.lam*ui)
                    self.v[:, j] -= self.eta * (-error*ui+self.lam*vj)
                    count += 1
#                     self.__update_lambda
                    self.__update_eta(count)
            #fの誤差がcriteria未満なら終了
            if self.__get_f_error(data) < criteria:
                print("stop")
                print("count is %d"% count)
                break

    def test(self):
        return numpy.dot(self.u.T, self.v)