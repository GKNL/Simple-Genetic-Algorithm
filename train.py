from question1.GA_decimal_global_local import *


def filter_different(choromo_list, fitness_list, max_diff):
    """
    过滤掉与目标值相差太大的结果
    :param choromo_list:
    :param fitness_list:
    :param max_diff:
    :return:
    """
    res_x = []
    res_fitness = []
    for i, choromo in enumerate(choromo_list):
        tmp_fitness = fitness_list[i]
        if abs(tmp_fitness - max_diff) < max_diff:
            res_x.append(choromo)
            res_fitness.append(tmp_fitness)
    return res_x, res_fitness


if __name__ == '__main__':
    EPOCHS = 1

    POP_SIZE = 100
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 1000  # 全局演化最大迭代次数
    iter_nums = N_GENERATION  # 实际迭代次数
    N_para = 3  # 变量个数
    M_parent = 10  # 即为M1，杂交时父体个数
    K_top = 1  # 精英杂交算法中，选取topK个最好的个体作为父体
    L_son = 1  # 在子空间中生成L_son个新个体，选取其中一个与上一代的最差个体进行比较
    optimization = -2709.093505572829

    # 全局-局部演化参数
    SUB_GENERATION = 20000  # 局部演化最大迭代次数
    alpha = 0.08
    epsilon = 0.05  # 两个个体之间的欧氏距离
    min_num = 4  # 最小最优解的个数
    max_num = 10  # 最大最优解的个数
    P = 0  # 局部演化的中心点数
    N_local = 100  # 局部演化的种群大小

    """迭代EPOCHS次，得到总体的计算结果"""
    res_x, res_fitness = [], []
    for i in range(EPOCHS):
        epoch_x, epoch_fitness = run_epoch()
        epoch_x, epoch_fitness - filter_different(epoch_x, epoch_fitness)
        res_x.extend(epoch_x)
        res_fitness.extend(epoch_fitness)

