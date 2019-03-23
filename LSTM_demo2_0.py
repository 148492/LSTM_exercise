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
'''此版本的目的是使用函数将整个LSTM封装起来,用于之后的调参'''

lookback = 1
n_train_and_test = 1500
size = 2000

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


def my_lstm_net(X, Y, lookback, n_train_and_test, size, epochs_=20, batch_size_=150, units_=50):
    # 归一化输入和输出
    # integer encode direction
    encoder = LabelEncoder()
    # normalize features
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X)
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    Y_scaled = scaler_Y.fit_transform(Y)

    # 制作训练集和预测集(已经归一化)
    train_X = X_scaled[:n_train_and_test, :]
    train_X = train_X.reshape(int(train_X.shape[0] / lookback), lookback, train_X.shape[1])
    train_Y = Y_scaled[:n_train_and_test:lookback, :]

    test_X = X_scaled[n_train_and_test:size, :]
    test_X = test_X.reshape(int(test_X.shape[0] / lookback), lookback, test_X.shape[1])
    test_Y = Y_scaled[n_train_and_test:size:lookback, :]

    # 构造神经网络
    model = Sequential()
    model.add(LSTM(units_, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(Y.shape[1]))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_Y, epochs=epochs_, batch_size=batch_size_, validation_data=(test_X, test_Y),
                        verbose=2,
                        shuffle=False)
    N = epochs_
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35  # the width of the bars: can also be len(x) sequence
    # 将history利用pyplot输出
    p1 = pyplot.bar(ind, history.history['loss'])
    p2 = pyplot.bar(ind, history.history['val_loss'])
    pyplot.legend((p1[0], p2[0]), ('train', 'test'))
    pyplot.show()

    # 进行预测
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    yhat = scaler_Y.inverse_transform(yhat)
    test_X = scaler_X.inverse_transform(test_X)
    test_Y = scaler_Y.inverse_transform(test_Y)
    rmse = sqrt(mean_squared_error(yhat[:, :], test_Y[:, :]))
    print('Test RMSE: %.3f' % rmse)
    return yhat

