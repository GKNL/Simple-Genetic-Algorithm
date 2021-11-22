import random
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import math

"""
题目一：
全局-局部演化算法
------------------
变量个数：n
编码方式：十进制编码
选择算子：~
杂交算子：1.多父体杂交  2.精英多父体杂交
变异算子：无
"""


def score_function(X_list):
    """
    目标函数
    :param X_list: 染色体组（多个变量组成的向量）
    :return: 目标函数值大小
    """
    score = 0
    x0 = X_list[0]
    for j in range(1, 6, 1):
        tmp = j * math.cos((j + 1) * x0 + j)
        score += tmp

    for n in range(1, N_para):
        x = X_list[n]
        y = 0
        for j in range(1, 6, 1):
            tmp = j * math.cos((j + 1) * x + j)
            y += tmp
        score *= y
    return score


def cal_euclidean(X_list, X_fitness, Y_list, Y_fitness):
    """
    计算两个个体之间的欧氏距离
    :param X_list:
    :param X_fitness:
    :param Y_list:
    :param Y_fitness:
    :return:
    """
    distance = 0
    for i in range(N_para):
        x1 = X_list[i]
        x2 = Y_list[i]
        distance += (x2 - x1) ** 2
    distance += (Y_fitness - X_fitness) ** 2
    return math.sqrt(distance)


def initial_population(pop_size):
    """
    初始化种群(一个种群内的每个个体由多个变量构成)
    :param pop_size: 种群个体数目
    :return:
    """
    population = []
    # 初始化生成范围在[-10,10]的pop_size个个体的十进制基因型种群
    for i in range(pop_size):
        per_chromosome = []
        for n in range(N_para):
            tmp_para = random.random() * 20 - 10
            per_chromosome.append(tmp_para)
        population.append(per_chromosome)
    return population


def cal_fitness(population):
    """
    输入N个变量的十进制染色体种群，计算种群中每个个体的适应值大小
    :param population:
    :return:
    """
    fitness = []
    for i in range(len(population)):
        x_list = population[i]
        tmp_fitness = score_function(x_list)
        fitness.append(tmp_fitness)
    return fitness


def find_max(population, fitness):
    max_fit = fitness[0]
    max_chromosome_idx = 0  # ！！！注意这里一定要是0
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal > max_fit:
            max_fit = tmpVal
            max_chromosome_idx = i
    return max_chromosome_idx, max_fit


def find_min(population, fitness):
    min_fit = fitness[0]
    min_chromosome_idx = 0
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal < min_fit:
            min_fit = tmpVal
            min_chromosome_idx = i
    return min_chromosome_idx, min_fit


def multi_parent_crossover(population, M_parent=5):
    """
    TODO: 不需要杂交概率？？？每一代进行一次即可？
    对种群中的个体，进行多父体杂交
    [直接对父母的染色体进行杂交更改]
    :param population:
    :param M_parent: 多父体杂交M参数
    :return:
    """
    parent_indexs = [random.randint(0, POP_SIZE - 1) for i in range(M_parent)]
    # 生成多父体杂交的alpha参数(随机生成M个参数)
    alphas = [random.uniform(-0.5, 1.5) for i in range(M_parent)]
    sum_alphas = np.sum(alphas)
    alphas = [i / sum_alphas for i in alphas]
    son = np.zeros(N_para)
    for i in range(M_parent):
        tmp_parent = np.array(population[parent_indexs[i]])
        son += tmp_parent * alphas[i]
    return son.tolist()


def excellent_multi_parent_crossover(population, fitness, M_parent=5, K_top=3, L_son=5):
    """
    对种群中的个体，进行精英多父体杂交
    [直接对父母的染色体进行杂交更改]
    :param population:
    :param M_parent: 多父体杂交M参数
    :param K_top: K个最好的个体
    :return:
    """
    # 挑选K个最好的个体
    sort_fit = copy.deepcopy(fitness)
    sort_fit.sort()
    parent_indexs = []
    for i in range(K_top):
        tmp_val = sort_fit[i]
        tmp_idx = fitness.index(tmp_val)
        parent_indexs.append(tmp_idx)
    # 随机挑选M-K个个体
    other_parent_indexs = [random.randint(0, POP_SIZE - 1) for i in range(M_parent - K_top)]
    parent_indexs.extend(other_parent_indexs)
    # 生成多父体杂交的alpha参数(随机生成M个参数)
    sons = []
    for k in range(L_son):
        alphas = [random.uniform(-0.5, 1.5) for i in range(M_parent)]
        sum_alphas = np.sum(alphas)
        alphas = [i / sum_alphas for i in alphas]
        son = np.zeros(N_para)
        for i in range(M_parent):
            tmp_parent = np.array(population[parent_indexs[i]])
            son += tmp_parent * alphas[i]
        sons.append(son.tolist())
    return sons


def multi_parent_select(population, fitness, new_chromo):
    """
    多父体杂交选择算子
    :param population:
    :param fitness:
    :param new_chromo:
    :return:
    """
    # 计算新生成子代的适应值
    son_score = score_function(new_chromo)
    # 找出种群中的最差适应度值
    max_idx, max_fitness = find_max(population, fitness)
    if son_score < max_fitness:
        population[max_idx] = new_chromo
        fitness[max_idx] = son_score


def excellent_multi_parent_select(population, fitness, new_chromos):
    """
    精英多父体选择算子
    :param population:
    :param fitness:
    :param new_chromos: 精英多父体杂交算法选出来的n个子代新个体
    :return:
    """
    # 计算新生成子代的适应值
    son_scores = cal_fitness(new_chromos)
    # 选择表现最好的个体作为最终新产生的子个体
    best_son_idx, best_son_score = find_min(new_chromos, son_scores)
    new_son = new_chromos[best_son_idx]
    # 找出种群中的最差适应度值
    max_idx, max_fitness = find_max(population, fitness)
    if best_son_score < max_fitness:
        population[max_idx] = new_son
        fitness[max_idx] = best_son_score


def choose_p_after_global(population, fitness, alpha):
    """
    从P(gen)中选择p个不同的个体
    迭代选择，直至p落在[min_num, max_num]之间
    :param population:
    :param alpha:
    :return:
    """
    global P
    global epsilon
    res = []
    while P < min_num or P > max_num:
        res = []
        # 对于任意不同的i和j，都有欧氏距离大于epsilon
        for i in range(POP_SIZE):
            flag = True  # 默认个体i与所有
            for j in range(POP_SIZE):
                distance = cal_euclidean(population[i], fitness[i], population[j], fitness[j])
                if i != j and distance <= epsilon:
                    flag = False
            if flag:
                res.append([population[i], fitness[i]])
        P = len(res)

        if P < min_num:
            epsilon = epsilon / (1 + alpha)
        else:
            epsilon = epsilon * (1 + alpha)

    # 对筛选之后的种群，按适应值降序排序
    def takeSecond(elem):  # 获取列表的第二个元素
        return elem[1]

    res.sort(key=takeSecond, reverse=False)  # 找最小值，因此升序排列即可
    res_pop = [i[0] for i in res]
    res_fitness = [i[1] for i in res]
    return res_pop, res_fitness


def sub_evolution(i_group, center_chromo):
    start = time.perf_counter()
    # Step 3.1: 定义搜索子空间
    bound_delta = math.sqrt(epsilon * (1 + (i_group - 1) / (P - 1)) / 2)
    BOUND_SUB = []
    for i in range(N_para):
        center_i = center_chromo[i]
        tmp_bound = [center_i - bound_delta, center_i + bound_delta]
        BOUND_SUB.append(tmp_bound)
    # Step 3.2: 在Di中随机生成N1个个体形成子种群Pi(0)
    population = []
    t = 0
    for i in range(N_local):
        per_chromosome = []
        for n in range(N_para):
            tmp_para = random.uniform(BOUND_SUB[n][0], BOUND_SUB[n][1])
            per_chromosome.append(tmp_para)
        population.append(per_chromosome)
    # Step 3.3: 演化进行与否判断
    min_fit = 0
    max_fit = 1
    while t<= SUB_GENERATION and min_fit != max_fit:
        # 计算种群个体适应度
        fitness = cal_fitness(population)
        _, max_fit = find_max(population, fitness)
        _, min_fit = find_min(population, fitness)
        # Step 3.4.1: 多父体精英演化
        new_son = excellent_multi_parent_crossover(population=population, fitness=fitness, M_parent=M_parent, K_top=K_top, L_son=L_son)
        # Step 3.4.2: 进行种群个体选择
        excellent_multi_parent_select(population, fitness, new_son)
        t += 1

    end = time.perf_counter()
    print('-----------------------Local {}--------------------------'.format(i_group+1))
    print('Local Space {} Running time: %s Seconds'.format(i_group) % (end - start))
    # Step 3.5: 选取最优的个体作为该局部演化的最优结果
    min_fitness_index = np.argmin(fitness)
    print("Local_Space_{}_min_fitness:".format(i_group), fitness[min_fitness_index])
    x = population[min_fitness_index]
    print("Local_Space_{}_min_x:".format(i_group), x)
    print('-----------------------Local {}--------------------------'.format(i_group+1))



def plot(results, iter_nums):
    """

    :param results: 其中每一个元素为[best_fitness, best_chromo, avg_fitness]
    :param iter_nums:
    :return:
    """
    X = []
    Y_best = []
    Y_avg = []

    for i in range(iter_nums):
        X.append(i)
        Y_best.append(results[i][0])
        Y_avg.append(results[i][2])

    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(X, Y_best, label="种群个体最优目标函数值")
    plt.plot(X, Y_avg, label="种群个体平均目标函数值")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    POP_SIZE = 100
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 500  # 全局演化最大迭代次数
    iter_nums = N_GENERATION  # 实际迭代次数
    N_para = 2  # 变量个数
    M_parent = 10  # 即为M1，杂交时父体个数
    K_top = 1  # 精英杂交算法中，选取topK个最好的个体作为父体
    L_son = 1  # 在子空间中生成L_son个新个体，选取其中一个与上一代的最差个体进行比较
    optimization = -2709.093505572829

    # 全局-局部演化参数
    SUB_GENERATION = 5000  # 局部演化最大迭代次数
    alpha = 0.08
    epsilon = 0.05  # 两个个体之间的欧氏距离
    min_num = 3  # 最小最优解的个数
    max_num = 10  # 最大最优解的个数
    P = 0  # 局部演化的中心点数
    N_local = 100  # 局部演化的种群大小

    # 1.初始化种群
    start = time.perf_counter()
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    for k in range(N_GENERATION):
        # 3.计算种群个体的适应度
        fitness = cal_fitness(pop)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        avg_fitness = np.sum(fitness) / POP_SIZE
        results.append([best_fitness, best_chromo, avg_fitness])
        # 当最优值与优化目标接近时，结束演化
        if abs(best_fitness - optimization) < 1e-5:
            print('Reach the optimization object!Total iteration time {}'.format(k + 1))
            iter_nums = k + 1
            break
        # 4.交叉
        new_son = multi_parent_crossover(population=pop, M_parent=M_parent)
        # 5.进行种群个体选择
        multi_parent_select(pop, fitness, new_son)

    end = time.perf_counter()

    print('-----------------------Global--------------------------')
    print('Global Running time: %s Seconds' % (end - start))
    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x = pop[min_fitness_index]
    print("min_x:", x)
    print('-----------------------Global--------------------------')

    plot(results, iter_nums)

    # Step 2：从P(gen)中选择p个不同的个体
    sub_centers, sub_fitness = choose_p_after_global(pop, fitness, alpha)
    # Step 3: 子空间演化
    for i in range(P):
        sub_evolution(i, sub_centers[i])
