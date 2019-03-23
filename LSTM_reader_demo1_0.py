from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from matplotlib import pyplot
import excel_exercise1 as aa1
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''这个版本就是准备直接从xls里读取数组存到csv里,运行时间很长,但是存到csv里就好多了,所以的话就是这样.'''
# 窗口的大小
lookback = 1
n_train_and_test = 1500
size = 2000
start = 3
end = start + size - 1
col_end = -1
# end = -1
# 构造最初始的数据集


excelpath = ['氧含量.xlsx', '流量.xlsx', '压力.xlsx']
# 从文件里读取相应的行和列的数据,输入到一个tf.constant里
a = []
for name in excelpath:
    # 每个各读全部所有的数据列,1000行
    a.append(aa1.xls_arrays(name, start, 1, end, col_end))
X = np.hstack((np.hstack((a[0], a[1])), a[2]))
Y = aa1.xls_arrays('温度.xlsx', start, 1, end, -1)

# 将空行的删掉,重新制作数据集,并且获取这个数据集的shape
a = [i for i in range(X.shape[1]) if (X[size - 1, i] == '' or X[1, i] == '')]
aa = np.delete(X, a, axis=1)
b = [i for i in range(Y.shape[1]) if (Y[size - 1, i] == '' or Y[1, i] == '')]
Y = np.delete(Y, b, axis=1)

# 将所有的转化为float类型
aa.astype('float32')
Y.astype('float32')

