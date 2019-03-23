import matplotlib.pyplot as plt
import numpy as np

x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5])
y1 = np.array([8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68])
y2 = np.array([9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74])
y3 = np.array([7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73])
x4 = np.array([8, 9, 8, 10, 8, 6, 8, 19, 3.8, 8, 4.8])
y4 = np.array([6.8, 5.6, 7.1, 8.84, 8.7, 7.04, 5.5, 1.50, 5.6, 7.9, 6.88])


# fit函数的意义就是画那条红线, 红线就是一个数据里最大最小值的连线,斜率是固定的
def fit(x):
    return 1 + 4 * x / 5


# 创建xfit数组，该数组包含两个元素：x的最大值和最小值
xfit = np.array([np.min(x), np.max(x)])

# 创建子图
plt.subplot(221)
# 在第一张子图上绘制两个图形
# x,y1的散点图，黑色方块

plt.plot(x, y1, 'ks', xfit, fit(xfit),
         'r-', lw=2)  # xfit,fit(xfit)红色实线，线宽为2
# 设置x轴范围：2到20
# 设置y轴范围：2到14
plt.axis([2, 20, 2, 14])

# plt.gca():获取当前子图
# plt.setp():设置图标实例的属性。
# 设置子图的xticklabels为空
# yticks显示为：4、8、12,，xticks显示：0、 10  、20
plt.setp(plt.gca(), xticklabels=['hasaki'], yticks=(4, 8, 12), xticks=(0, 10, 20))

# matplotlib.pyplot.text(x, y, s, fontdict=None, withdash=False, **kwargs)
# 第一个参数是x轴坐标
# 第二个参数是y轴坐标
# 第三个参数是要显式的内容
# fontsize设置显示字体大小
plt.text(3, 12, 'I', fontsize=20)

# 创建第二个子图
plt.subplot(222)
# 绘制两个图形
# x,y2的散点图，黑色方块
# xfit,fit(xfit)红色实线，线宽为2
plt.plot(x, y2, 'bs', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
# plt.gca():获取当前子图
# plt.setp():设置图标实例的属性。
# 设置子图的xticklabels、yticklabels为空
# yticks显示为：4, 8, 12,，xticks显示：0、 10  、20
plt.setp(plt.gca(), xticks=(0, 10, 20), xticklabels=[],
         yticks=(4, 8, 12), yticklabels=[], )
plt.text(3, 12, 'II', fontsize=20)

plt.subplot(223)
plt.plot(x, y3, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.setp(plt.gca(), yticks=(4, 8, 12), xticks=(0, 10, 20))
plt.text(3, 12, 'III', fontsize=20)

plt.subplot(224)
xfit = np.array([np.min(x4), np.max(x4)])
plt.plot(x4, y4, 'ks', xfit, fit(xfit), 'r-', lw=2)
plt.axis([2, 20, 2, 14])
plt.setp(plt.gca(), yticklabels=[], yticks=(4, 8, 12), xticks=(0, 10, 20))
plt.text(3, 12, 'IV', fontsize=20)

# 验证统计数据
pairs = (x, y1), (x, y2), (x, y3), (x4, y4)

# corrcoef函数
# 计算两组数的相关系数
# 返回结果为矩阵，第i行第j列的数据表示第i组数与第j组数的相关系数。对角线为1
for x, y in pairs:
    print(
        'mean=%1.2f, std=%1.2f, r=%1.2f' % (np.mean(y), np.std(y), np.corrcoef(x, y)[0][1])
    )
# 显示绘制图像
plt.show()
