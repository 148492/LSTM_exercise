import LSTM_demo2_1 as ll

'''这个就是采取方法找到参数之中最优的参数配置,这个版本准备先采取循环的方式,之后的版本准备采取tensorflow的方式'''
i = 100
a = ll.readfile1()
X = a.X
Y = a.Y
lookback = a.lookback
n_train_and_test = a.n_train_and_test
size = a.size

yyy=pd.DataFrame(yhat)
yyy.to_excel('yyy1.xlsx')
yhat[2]


#
# for epochs in range(20, 51):
#     for unit in range(10, 300):
#
#         z = ll.my_lstm_net(X, Y, lookback, n_train_and_test, size, epochs_=epochs, batch_size_=150, units_=unit)
#         if z < i:
#             i = z
#             print(z, i, epochs, unit)
