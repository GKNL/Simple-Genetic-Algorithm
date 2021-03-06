import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import math

"""
绘制n=2的情况下的函数曲线图
"""

def fun(X, Y):
    """
    目标函数
    :param X_list: 染色体组（多个变量组成的向量）
    :return: 目标函数值大小
    """
    score = 0
    val1 = 0
    val2 = 0
    for j in range(1, 6, 1):
        tmp = j * np.cos((j + 1) * X + j)
        val1 += tmp
    for j in range(1, 6, 1):
        tmp = j * np.cos((j + 1) * Y + j)
        val2 += tmp
    score = val1 * val2
    return score

if __name__ == '__main__':
    fig1 = plt.figure()  # 创建一个绘图对象
    ax = Axes3D(fig1)  # 用这个绘图对象创建一个Axes对象(有3D坐标)
    X = np.arange(-10, 10, 0.05)
    Y = np.arange(-10, 10, 0.05)  # 创建了从-10到10，步长为0.05的arange对象
    # 至此X,Y分别表示了取样点的横纵坐标的可能取值
    # 用这两个arange对象中的可能取值一一映射去扩充为所有可能的取样点
    X, Y = np.meshgrid(X, Y)
    Z = fun(X, Y)  # 用取样点横纵坐标去求取样点Z坐标
    plt.title("This is main title")  # 总标题
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)  # 用取样点(x,y,z)去构建曲面
    ax.set_xlabel('x label', color='r')
    ax.set_ylabel('y label', color='g')
    ax.set_zlabel('z label', color='b')  # 给三个坐标轴注明
    plt.show()  # 显示模块中的所有绘图对象
