# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from pandas import DataFrame
from FC import MatrixFactorization

def convert_user_id_to_index(users, user_id):
    #ユーザーidから配列のインデックスを取得
    return np.where(users==user_id)[0][0]

def convert_index_to_item_id(items, index):
    #配列のインデックスからアイテムのidを取得
    return items[index]

def get_item_name(item_id):
    #アイテムidから対応する映画名を取得
    f = open("u.item", "r")
    line = f.readline()
    while line:
        info = line.split("|")
        if int(info[0]) == item_id:
            f.close()
            return info[1]
        line = f.readline()
    f.close()
    return None

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
    MF = MatrixFactorization()
    MF.fit(data)
    index = convert_user_id_to_index(users, 2)
    item_index = MF.predict(data, index)
    item_id = convert_index_to_item_id(items, item_index)
    print(get_item_name(item_id))
