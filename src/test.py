# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from pandas import DataFrame
from FC import MatrixFactorization

def convert_user_id_to_index(user, user_id):
    #ユーザーidから配列のインデックスを取得
    return np.where(user==user_id)[0][0]

def convert_index_to_item_id(item, index):
    #配列のインデックスからアイテムのidを取得
    return item[index]

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
#     test = np.array([[1, 2, 0, 4],
#                   [4, 0, 0, 1],
#                   [1, 1, 0, 5],
#                   [1, 0, 0, 4],
#                   [0, 1, 5, 4],
#                   ])
    test = np.random.randint(0, 6, (100, 100))
    user = np.arange(1, len(test)+1)
    item = np.arange(1, len(test[0])+1)
    index = convert_user_id_to_index(user, 4)
    print(index)
    df = DataFrame(test, index=user, columns=item)
    print(df)
    MF = MatrixFactorization()
    MF.fit(test)
    print(MF.verification())
    item_index, rate = MF.predict(test, index)
    print(item_index)
    item_id = convert_index_to_item_id(item, item_index)
    print(item_id)
    print(get_item_name(item_id), rate)
    print("error=%f"%MF.get_result_error(test))
#     unrated = MF.get_unrated(test, index)
#     print(unrated[0])
