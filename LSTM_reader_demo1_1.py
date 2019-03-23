import pandas as pd

'''这个版本利用pandas读取所需的文件,存到csv或者不存主要看运行时间
测试一下运行时间和能不能在约定时间里正确读取到所需要的数据'''
'''这个文件不把空的格子删掉了,保留,之后训练的时候如果实在不行能和所需要的东西对齐,
这个先读4000行,能读多少读多少'''

yang_han_liang = pd.read_excel('氧含量.xlsx', header=[0, 1, 2], index_col=[0])
liu_liang = pd.read_excel('流量.xlsx', header=[0], index_col=[0])
ya_li = pd.read_excel('压力.xlsx', header=[0, 1, 2], index_col=[0])

wen_du = pd.read_excel('温度.xlsx', header=[0, 1, 2], index_col=[0])
'''不必转化为DataFrame,因为读进来的时候就是DataFrame'''
'''不用存储到csv里了,只需要像现在这样读,时间就很短,不会花太长时间'''

'''之后就是怎么按照名字读取一列的值'''
'''http://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html'''
# a = ya_li['y2j507-PI-11001']
# a.columns.levels[0][1]
# 可以获取第一行里所有的阀门的名字,到时候可以按照这个查找

'''截取一部分自己想要的,并且保留表头?
http://pandas.pydata.org/pandas-docs/version/0.22/generated/pandas.DataFrame.fillna.html#pandas.DataFrame.fillna'''
# dropna() 默认是删掉空洞所在的那行--对齐!

'''拼接一部分自己想要的,保留表头'''
yali_and_yanghanliang = ya_li.merge(right=yang_han_liang, how='inner',
                                    left_on=ya_li.index,
                                    right_on=yang_han_liang.index)
# ya_li.keys()

# 问题出在,合并之后会多出来一个index['key_0//'],是作为left_on的
# ya_yang_liu = yali_and_yanghanliang.merge(right=liu_liang, how='inner',
#                                           left_on=yali_and_yanghanliang.index,
#                                           right_on=liu_liang['key_0//'])
# 输出左边第一列,称之为keys
# liu_liang.shape,ya_li.shape,yang_han_liang.shape
# ((5834, 85), (5832, 56), (5832, 26))
output = pd.concat([ya_li, liu_liang, yang_han_liang], sort=False, axis=1, join='inner')


'''把这个东西里的nan用什么代替比较好?先用0代替'''
# fillna(0.)
output1 = output.dropna(how='all')
output2 = output1.fillna(0)
'''存储到一个csv里留作使用,存储的格式是保留表头还是不保留?必须要保留,不然全白费了
不存储了'''
# liu_liang.to_csv('DataFrame流量.csv')
# a=pd.read_csv('DataFrame流量.csv',engine='python')
# 不对,因为这个里面输出到csv里和输出到别的里,最后读取时间都一样,格式甚至都一样
'''为了保持时间的一致性,应该把东西拼起来,
这样互相之间没有对方某时间对应的行就没了,互相之间的行都是相同时间的'''
