from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from matplotlib import pyplot
import excel_exercise1 as aa1
import os
'''这个版本的是最开始直接从xls读取数组文档的,读取速度极其慢,基本都是卡住不动了'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

n_train_and_test = 80
look_back = 1  # 使用连续多长的时间的数值预测数据
numbers_of_datas = 100  # 所需要载入的一列数据的数量
row_s, col_s = 4, 2
row_e, col_e = row_s + numbers_of_datas, col_s
excel_path, excel_path1, excel_path2 = '温度.xlsx', '氧含量.xlsx', '流量.xlsx'
# 从文件里读取相应的行和列的数据,输入到一个tf.constant里
ndarray_w = aa1.xls_arrays(excel_path, row_s, col_s, row_e, col_e)
ndarray_y = aa1.xls_arrays(excel_path1, row_s, col_s, row_e, col_e)
ndarray_l = aa1.xls_arrays(excel_path2, row_s, col_s, row_e, col_e)
nd_array_yl = np.append(ndarray_l, ndarray_y, axis=1)
train_X = nd_array_yl[:n_train_and_test, :]
train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
train_y = ndarray_w[:n_train_and_test, :]

test_X = nd_array_yl[n_train_and_test:, :]
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
test_y = ndarray_w[n_train_and_test:, :]

# design network
model = Sequential()
model.add(LSTM(20, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2,
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
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# 读取array数据之后制作数据集
# def create_dataset(ndarray_y, ndarray_l):
#     dataX = [], dataY = []
#     for i in range(len(dataset) - look_back - 1):
#         x = dataset[i:i + look_back, 0]
#
#     return np.array(dataX), np.array(dataY)

# tensor_w = tf.constant(ndarray_w)
# tensor_y = tf.constant(ndarray_y)
# tensor_l = tf.constant(ndarray_l)
