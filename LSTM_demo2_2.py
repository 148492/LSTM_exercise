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
'''此版本的目的是修改之后的打印输出pyplot部分,能把预测的结果和原结果进行对比
另外还把这个函数的具体内容进行了优化,就是函数的开始和结束能定义了,可以跳过那些被自动补零的数'''

lookback = 1
start = 3100
size = 2000
n_train_and_test = 1500
end = start + size
middle = start + n_train_and_test

units_ = 70
epochs_ = 20
batch_size_ = 2

yang_han_liang = pd.read_excel('氧含量.xlsx', header=[0, 1, 2], index_col=[0])
liu_liang = pd.read_excel('流量.xlsx', header=[0], index_col=[0])
ya_li = pd.read_excel('压力.xlsx', header=[0, 1, 2], index_col=[0])
wen_du = pd.read_excel('温度.xlsx', header=[0, 1, 2], index_col=[0])

yang_han_liang = yang_han_liang.fillna(0)
liu_liang = liu_liang.fillna(0)
ya_li = ya_li.fillna(0)
wen_du = wen_du.fillna(0)

ya_yang_liu = pd.concat([ya_li, liu_liang, yang_han_liang], sort=False, axis=1, join='inner')

inputX = ya_yang_liu
outputY = wen_du

X = inputX.values
Y = outputY.values

# 将所有的转化为float类型
X.astype('float64')
Y.astype('float64')

# 归一化输入和输出
# integer encode direction
encoder = LabelEncoder()
# normalize features
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X)
scaler_Y = MinMaxScaler(feature_range=(0, 1))
Y_scaled = scaler_Y.fit_transform(Y)

# 制作训练集和预测集(已经归一化)
train_X = X_scaled[start:middle, :]
train_X = train_X.reshape(int(train_X.shape[0] / lookback), lookback, train_X.shape[1])
train_Y = Y_scaled[start:middle:lookback, :]

test_X = X_scaled[middle:end, :]
test_X = test_X.reshape(int(test_X.shape[0] / lookback), lookback, test_X.shape[1])
test_Y = Y_scaled[middle:end:lookback, :]

# 构造神经网络
model = Sequential()
model.add(LSTM(units_, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(Y.shape[1]))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_Y, epochs=epochs_, batch_size=batch_size_, validation_data=(test_X, test_Y),
                    verbose=2,
                    shuffle=False)

# 进行预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
yhat = scaler_Y.inverse_transform(yhat)
test_X = scaler_X.inverse_transform(test_X)
test_Y = scaler_Y.inverse_transform(test_Y)
y_difference = test_Y - yhat
yhatdf = pd.DataFrame(yhat)
y_differencedf = pd.DataFrame(y_difference)
# y_differencedf.to_excel('y_diff.xlsx')

yhat2 = yhat[:, 2]
pyplot.plot(yhat2, label='yhat_2')
pyplot.plot(test_Y[:, 2], label='yreal_2')
pyplot.legend()
pyplot.show()


def my_lstm_net(X, Y, start, end, middle, lookback=1,
                epochs_=10, batch_size_=20, units_=50,
                col_check=3, shuffle_=False):
    # 归一化输入和输出
    # integer encode direction
    encoder = LabelEncoder()
    # normalize features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_scaled = scaler_Y.fit_transform(Y)

    # 制作训练集和预测集(已经归一化)
    train_X = X_scaled[start:middle, :]
    train_X = train_X.reshape(int(train_X.shape[0] / lookback), lookback, train_X.shape[1])
    train_Y = Y_scaled[start:middle:lookback, :]

    test_X = X_scaled[middle:end, :]
    test_X = test_X.reshape(int(test_X.shape[0] / lookback), lookback, test_X.shape[1])
    test_Y = Y_scaled[middle:end:lookback, :]

    # 构造神经网络
    model = Sequential()
    model.add(LSTM(units_, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(Y.shape[1]))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_Y, epochs=epochs_, batch_size=batch_size_, validation_data=(test_X, test_Y),
                        verbose=2,
                        shuffle=shuffle_)

    # 进行预测
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    yhat = scaler_Y.inverse_transform(yhat)
    test_X = scaler_X.inverse_transform(test_X)
    test_Y = scaler_Y.inverse_transform(test_Y)

    yhat2 = yhat[:, col_check]
    pyplot.plot(yhat2, label='yhat2')
    pyplot.plot(test_Y[:, col_check], label='yreal2')
    pyplot.legend()
    pyplot.show()
    return yhat


def check_lstm_plot(yhat, test_Y, col_check):
    yhat2 = yhat[:, col_check]
    pyplot.plot(yhat2, label='yhat2')
    pyplot.plot(test_Y[:, col_check], label='yreal2')
    pyplot.legend()
    pyplot.show()


def plot_all(Y, x):
    pyplot.plot(Y[:, x])
    pyplot.legend()
    pyplot.show()
