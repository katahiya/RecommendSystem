# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from pandas import DataFrame

if __name__ == '__main__':
    #データの読み込み
    original_data = np.loadtxt("u.data", delimiter="\t")
    original_users = original_data[:,0]
    original_items = original_data[:,1]
    score = original_data[:,2]
    #ユーザーid，映画idのリストから重複を削除
    users = np.unique(original_users)
    items = np.unique(original_items)
    #ユーザー*アイテムのデータフレーム作成
    #全要素を0で初期化
    df = DataFrame(np.zeros((len(users), len(items))), index=users, columns=items)
    #評価点の代入
    for i in range(len(original_users)):
        df.ix[original_users[i], original_items[i]] = score[i]
    data = df.as_matrix()
    print(len(data[0]))
