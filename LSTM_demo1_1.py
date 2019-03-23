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
# 窗口的大小
lookback = 1
n_train_and_test = 1500
size = 2000
start = 3
end = start + size - 1
col_end = -1
# end = -1
# 构造最初始的数据集

'''这个版本是通过一个循环来读取所有的数组的,最后出现了一点错误吧,应该是删除部分空行的时候的错误'''
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
# 归一化输入和输出
# integer encode direction
encoder = LabelEncoder()
# normalize features
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(aa)
scaler_Y = MinMaxScaler(feature_range=(0, 1))
Y_scaled = scaler_Y.fit_transform(Y)

# 制作训练集和预测集
train_X = X_scaled[:n_train_and_test, :]
train_X = train_X.reshape(int(train_X.shape[0] / lookback), lookback, train_X.shape[1])
train_Y = Y_scaled[:n_train_and_test:lookback, :]

test_X = X_scaled[n_train_and_test:, :]
test_X = test_X.reshape(int(test_X.shape[0] / lookback), lookback, test_X.shape[1])
test_Y = Y_scaled[n_train_and_test::lookback, :]

# design network
model = Sequential()
model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(Y.shape[1]))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_Y, epochs=20, batch_size=150, validation_data=(test_X, test_Y), verbose=2,
                    shuffle=False)

# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
inv_y = concatenate((test_Y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
