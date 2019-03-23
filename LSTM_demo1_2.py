from numpy import concatenate
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 窗口的大小
'''这个版本的是直接从已经存好的csv里读的,csv的存储方式还是直接从xls里读取的,比较慢'''
lookback = 1
n_train_and_test = 1500
size = 2000
start = 3
end = start + size - 1
col_end = -1
# end = -1
aa = pd.read_csv('X_2000_all.csv', engine='python').values
Y = pd.read_csv('Y_2000_0310_all.csv', engine='python').dropna(axis=1, how='any').values
b = [i for i in range(Y.shape[1]) if (np.isnan(float(Y[size - 1, i])) or np.isnan(float(Y[1, i])))]
Y = np.delete(Y, b, axis=1)
Y = np.delete(Y, 0, axis=1)

c = [i for i in range(aa.shape[1]) if (np.isnan(float(aa[- 1, i])) or np.isnan(float(aa[1, i])))]
aa = np.delete(aa, c, axis=1)
# 将所有的转化为float类型
aa.astype('float64')
Y.astype('float64')

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
yhat = scaler_Y.inverse_transform(yhat)
test_X = scaler_X.inverse_transform(test_X)
# inv_yhat = concatenate((yhat, test_X[:, :]), axis=1)
# inv_yhat = scaler.inverse_transform(inv_yhat)
# inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_Y = scaler_Y.inverse_transform(test_Y)
# inv_y = concatenate((test_Y, test_X[:, :]), axis=1)
# inv_y = inv_y[:, 0]
# calculate RMSE
rmse = sqrt(mean_squared_error(yhat[:, :], test_Y[:, :]))
print('Test RMSE: %.3f' % rmse)
