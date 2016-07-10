# coding: utf-8
'''
Created on 2016/07/06

'''

import numpy as np
from pandas import DataFrame
from FC import MatrixFactorization

def convert_id_to_index(user, user_id):
    result_array = np.where(user==user_id)
    return result_array[0][0]

if __name__ == '__main__':
    test = np.array([[1, 2, 0, 4],
                  [4, 0, 0, 1],
                  [1, 1, 0, 5],
                  [1, 0, 0, 4],
                  [0, 1, 5, 4],
                  ])
#     test = np.random.randint(0, 6, (4, 6))
    user = np.arange(1, len(test)+1)
    item = np.arange(1, len(test[0])+1)
    index = convert_id_to_index(user, 2)
    print(index)
    df = DataFrame(test, index=user, columns=item)
    print(df)
    MF = MatrixFactorization()
    MF.fit(test)
    print(MF.verification())
    print(MF.predict(test, index))
#     unrated = MF.get_unrated(test, index)
#     print(test[:, unrated])
