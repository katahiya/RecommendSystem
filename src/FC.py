# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from math import fabs
from numba import jit

class MatrixFactorization(object):
    #MatrixFactorizationを実装するクラス

    def __update_eta(self, count):
        #学習率の更新
        #eta = 1/(alpha(更新回数+beta))の式で更新
        self.eta = 1 / (self.alpha*(count+self.beta))

    def __init__(self, alpha=0.05, beta=500, lam=0.1):
        '''
        Constructor
        '''
        self.u = None   #ユーザーの特徴ベクトル
        self.v = None   #アイテムの特徴ベクトル
        self.alpha = alpha
        self.beta = beta
        self.eta = None #学習率
        self.__update_eta(0)
        self.lam = lam  #行列の更新に使用

    @jit
    def __get_error(self, dataij, ui, vj):
        return dataij- np.dot(ui.T, vj)

    @jit
    def __calculate_f(self, data):
    #二乗誤差に正則項を加えた関数fを求める
        f = 0;
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == 0:
                    continue
                f += 0.5 * pow(self.__get_error(data[i][j], self.u[:, i], self.v[:, j]), 2)
        f += 0.5 * self.lam * (np.linalg.norm(self.u)+np.linalg.norm(self.v))
        return f

    @jit
    def __update_f(self, data):
    #fの値を更新してその誤差を取得
        old_f = self.f
        self.f = self.__calculate_f(data)
        return fabs(self.f - old_f)

    @jit
    def fit(self, data, K=30, steps=5000, criteria=0.0001):
        n = len(data)   #ユーザー数
        m = len(data[0]) #アイテム数
        count = 0   #更新回数
        self.u = np.random.rand(K, n)    #K×ユーザー数の行列を生成
        self.v = np.random.rand(K, m)    #K×アイテム数の行列を生成
        self.f = self.__calculate_f(data)
        #stepsの回数を上限に行列を修正
        for step in range(steps):
            for i in range(n):
                for j in range(m):
                    if data[i][j] == 0:
                        continue
                    ui = self.u[:, i]
                    vj = self.v[:, j]
                    error = self.__get_error(data[i][j], ui, vj)
                    #パラメータの更新
                    (self.u[:, i], self.v[:, j]) = ui-self.eta * (-error*vj+self.lam*ui), vj-self.eta * (-error*ui+self.lam*vj)
                    #反復回数の増加
                    count += 1
                    self.__update_eta(count)
            #修正前と修正後のfの差がcriteria未満なら終了
            if self.__update_f(data) < criteria:
                print("stop")
                print("count is %d"% count)
                break

    @jit
    def get_unrated(self, data, index):
    #ユーザーのインデックスから未評価のアイテムのインデックスのリストを取得S
        return np.where(data[index]==0)[0]

    @jit
    def predict(self, data, index):
    #ユーザーのインデックスから推薦するアイテムのインデックスを取得
        unrated = self.get_unrated(data, index)
        unrated_items = np.dot(self.u[:, index].T, self.v[:, unrated])
        plausible_rate = np.max(unrated_items)
        return unrated[np.where(unrated_items==plausible_rate)[0][0]], plausible_rate

    def get_result_error(self, data):
    #二乗誤差の取得
        error = 0;
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] == 0:
                    continue
                error += 0.5 * pow(self.__get_error(data[i][j], self.u[:, i], self.v[:, j]), 2)
        return error

    @jit
    def verification(self):
    #ユーザーとアイテムの特徴ベクトルから元のデータセットを復元
        return np.dot(self.u.T, self.v)