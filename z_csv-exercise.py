import csv
import pandas
import numpy as np
import excel_exercise1 as  aa1

# 先把输入的文件名用无限长列表装起来
names = ['压力.xlsx', '氧含量.xlsx', '流量.xlsx', '温度.xlsx']

# 再把xls_array的输出用列表装起来 可以用字典对应
a = []
for name in names:
    # 每个各读全部所有的数据列,1000行
    a.append(aa1.xls_arrays(name, 3, -1, 1003, -1))

# 按照列表的index,做一个循环,在这个循环里,把整个数据合并成一个大的列表
output = []
for elements in a:
    output = np.hstack((output, elements))

# 如果有可能的话,在每一列头顶都加一个这列的名字




