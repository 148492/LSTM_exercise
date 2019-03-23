import numpy as np

filename = '1.txt'

# 第一种方法

# f = open("1.txt", "r")   #设置文件对象
# data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
# f.close()             #关闭文件

str = 'abc 123 456'
with open('data.txt', 'w') as f:
    f.write(str)
with open('data.txt', 'r') as f:
    print(f.readline())

# datas = np.loadtxt("1.txt") #unnable

# for i in data:
#     print(i)

# with open('1.txt', 'w') as f:
#     print(1)


# f=open('1.txt', 'r')
# print(f.tell())

# line = f.readline()
# print(line)

# f.seek(0, 0)
