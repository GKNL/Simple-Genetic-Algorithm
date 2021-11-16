import matplotlib.pyplot as plt  # 绘图用的模块
from mpl_toolkits.mplot3d import Axes3D  # 绘制3D坐标的函数
import numpy as np
import math

"""
绘制n=1的情况下的函数曲线图
"""

def fun(X):
    """
    目标函数
    :param X_list: 染色体组（多个变量组成的向量）
    :return: 目标函数值大小
    """
    score = 0
    for j in range(1, 6, 1):
        tmp = j * np.cos((j + 1) * X + j)
        score += tmp
    return score

if __name__ == '__main__':
    fig1 = plt.figure()  # 创建一个绘图对象
    X = np.arange(-10, 10, 0.01)
    Y = fun(X)
    # plt.title("This is main title")  # 总标题
    plt.plot(X, Y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()  # 显示模块中的所有绘图对象
