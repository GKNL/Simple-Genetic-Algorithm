import random
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import math

"""
题目四：
多父体杂交遗传算法
------------------
变量个数：n
编码方式：十进制编码
选择算子：~
杂交算子：1.多父体杂交  2.精英多父体杂交
变异算子：无
约束限制：1. 种群初始化时进行限制限制
         2. 交叉变异后，通过if语句来进行判断
         3. 惩罚项
"""


def score_function(X_list):
    """
    目标函数
    :param X_list: 染色体组（多个变量组成的向量）
    :return: 目标函数值大小
    """
    score = 5.3578547 * math.pow(X_list[2], 2) + 0.8356891 * X_list[0] * X_list[4] + 37.293239 * X_list[0] - 40792.141
    return score


def punish_function(X_list):
    """
    计算单个个体的惩罚值
    :param X_list:
    :return:
    """
    cost = 0
    x = X_list
    x1, x2, x3, x4, x5 = x[0], x[1], x[2], x[3], x[4]
    # 不等式约束一
    e1_left = 0 - (85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022503 * x3 * x5)
    e1_right = 85.334407 + 0.0056858 * x2 * x5 + 0.0006262 * x1 * x4 - 0.0022503 * x3 * x5 - 92
    cost += max(0, e1_left) + max(0, e1_right)
    # 不等式约束二
    e2_left = 90 - (80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 * x3)
    e2_right = 80.51249 + 0.0071317 * x2 * x5 + 0.0029955 * x1 * x2 + 0.0021813 * x3 * x3 - 110
    cost += max(0, e2_left) + max(0, e2_right)
    # 不等式约束三
    e3_left = 20 - (9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4)
    e3_right = 9.300961 + 0.0047026 * x3 * x5 + 0.0012547 * x1 * x3 + 0.0019085 * x3 * x4 - 25
    cost += max(0, e3_left) + max(0, e3_right)
    return cost


def initial_population(pop_size):
    """
    初始化种群(一个种群内的每个个体由多个变量构成)
    :param pop_size: 种群个体数目
    :return:
    """
    population = []
    # 初始化生成范围在[LOWER_BOUND,UPPER_BOUND]的pop_size个个体的十进制基因型种群
    for i in range(pop_size):
        per_chromosome = []
        for n in range(N_para):
            tmp_para = random.uniform(LOWER_BOUND[n], UPPER_BOUND[n])
            per_chromosome.append(tmp_para)
        population.append(per_chromosome)
    return population


def cal_fitness(population, punishment):
    """
    输入N个变量的十进制染色体种群，计算种群中每个个体的适应值大小
    （适应度值=目标函数值+惩罚项）
    :param population:
    :param punishment: 种群对应的罚值矩阵
    :return:
    """
    fitness = []
    for i in range(len(population)):
        x_list = population[i]
        # 适应度值=目标函数值+惩罚项
        tmp_fitness = score_function(x_list) + 1000*punishment[i]
        fitness.append(tmp_fitness)
    return fitness


def cal_punishment(population):
    """
    计算种群中每个个体的惩罚项（<=0）
    :param population:
    :return:
    """
    punish = []
    for i in range(len(population)):
        x = population[i]
        tmp_cost = punish_function(x)
        punish.append(tmp_cost)
    return punish


def judge_legal(chromosome):
    """
    判断一条染色体中的各决策变量是否越界
    :param chromosome:
    :return:
    """
    for i in range(N_para):
        tmp_x = chromosome[i]
        low = LOWER_BOUND[i]
        high = UPPER_BOUND[i]
        if tmp_x > high or tmp_x < low:
            chromosome[i] = random.uniform(low, high)
    return chromosome


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
    # 对新生成的个体进行判断，对越界部分进行处理
    son = son.tolist()
    son = judge_legal(son)
    return son


def excellent_multi_parent_crossover(population, fitness, M_parent=5, K_top=3, L_son = 5):
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
        # 对新生成的个体进行判断，对越界部分进行处理
        son = son.tolist()
        son = judge_legal(son)
        sons.append(son)
    return sons


def multi_parent_select(population, fitness, new_chromo):
    """
    多父体杂交选择算子
    :param population:
    :param fitness:
    :param new_chromo:
    :return:
    """
    # 计算新生成子代的适应值和惩罚值
    son_score = score_function(new_chromo)
    son_punishment = punish_function(new_chromo)
    son_fitness = son_score + son_punishment
    # 找出种群中的最差个体的适应度值和惩罚值
    worst_idx, worst_fitness = find_max(population, fitness)
    worst_chromo = population[worst_idx]
    worst_punishment = punish_function(worst_chromo)

    # 规则1：如果child违反约束而parent没有违反约束，则取parent
    if worst_punishment < 0.0000001 and son_punishment >= 0.0000001:
        return
    # 规则2：如果parent违反约束而child没有违反约束，则取child
    elif worst_punishment >= 0.0000001 and son_punishment < 0.0000001:
        population[worst_idx] = new_chromo
        fitness[worst_idx] = son_fitness
    # 规则3：如果 parent 和 child 都没有违反约束（或都违反约束），则取适应度小的
    else:
        if worst_fitness <= son_fitness:  # 保留老个体
            return
        else:  # 新个体对老个体进行取代
            population[worst_idx] = new_chromo
            fitness[worst_idx] = son_fitness


def excellent_multi_parent_select(population, fitness, new_chromos):
    """
    精英多父体选择算子
    :param population:
    :param fitness:
    :param new_chromos: 精英多父体杂交算法选出来的n个子代新个体
    :return:
    """
    # 计算新生成子代的适应值(目标函数值+惩罚值)
    son_punishment = cal_punishment(new_chromos)
    son_fitness = cal_fitness(new_chromos, son_punishment)
    # 选择表现最好的个体作为最终新产生的子个体
    best_son_idx, best_son_fitness = find_min(new_chromos, son_fitness)
    new_son = new_chromos[best_son_idx]
    new_son_punishment = son_punishment[best_son_idx]
    new_son_fitness = best_son_fitness

    # 找出种群中的最差个体的适应度值和惩罚值
    worst_idx, worst_fitness = find_max(population, fitness)
    worst_chromo = population[worst_idx]
    worst_punishment = punish_function(worst_chromo)

    # 规则1：如果child违反约束而parent没有违反约束，则取parent
    if worst_punishment < 0.0000001 and new_son_punishment >= 0.0000001:
        return
    # 规则2：如果parent违反约束而child没有违反约束，则取child
    elif worst_punishment >= 0.0000001 and new_son_punishment < 0.0000001:
        population[worst_idx] = new_son
        fitness[worst_idx] = new_son_fitness
    # 规则3：如果 parent 和 child 都没有违反约束（或都违反约束），则取适应度小的
    else:
        if worst_fitness <= new_son_fitness:  # 保留老个体
            return
        else:  # 新个体对老个体进行取代
            population[worst_idx] = new_son
            fitness[worst_idx] = new_son_fitness

def plot(results):
    X = []
    Y = []

    for i in range(N_GENERATION):
        X.append(i)
        Y.append(results[i][0])

    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    POP_SIZE = 100
    LOWER_BOUND = [78, 33, 27, 27, 27]  # 决策变量下界
    UPPER_BOUND = [102, 45, 45, 45, 45]  # 决策变量上界
    N_GENERATION = 16000
    CROSS_PROB = 0.7
    N_para = 5  # 变量个数
    M_parent = 10  # 杂交时父体个数
    K_top = 5  # 精英杂交算法中，选取topK个最好的个体作为父体
    L_son = 2  # 在子空间中生成L_son个新个体，选取其中一个与上一代的最差个体进行比较

    # 1.初始化种群
    start = time.perf_counter()
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    for k in range(N_GENERATION):
        # 3.计算种群个体的罚值和适应度
        punishment = cal_punishment(pop)
        fitness = cal_fitness(pop, punishment)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        results.append([best_fitness, best_chromo])
        # 4.交叉
        new_son = excellent_multi_parent_crossover(population=pop, fitness=fitness, M_parent=M_parent, K_top=6, L_son=4)
        # 5.进行种群个体选择
        excellent_multi_parent_select(pop, fitness, new_son)

    end = time.perf_counter()

    print('Running time: %s Seconds' % (end - start))
    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x = pop[min_fitness_index]
    print("min_x:", x)

    plot(results)
