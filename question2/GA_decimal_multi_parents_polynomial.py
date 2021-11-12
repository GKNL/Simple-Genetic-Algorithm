import random
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import math

"""
题目二：
多父体杂交遗传算法
------------------
变量个数：n
编码方式：十进制编码
选择算子：~
杂交算子：1.多父体杂交  2.精英多父体杂交
变异算子：无
"""


def score_function(para_list):
    """
    目标函数
    :param para_list: 染色体组（多个变量组成的向量）
    :return: 目标函数值大小
    """
    x = para_list[0]
    y = para_list[1]
    z = para_list[2]
    equation1 = math.pow((math.sin(x)*math.cos(y) + math.sin(y)*math.cos(z) + math.sin(z)*math.cos(x) + math.sqrt(6)/2 - 1), 2)
    equation2 = math.pow((math.tan(x)*math.tan(y) + math.tan(y)*math.tan(z) + math.tan(z)*math.tan(x) - 4*math.sqrt(3)/3 -1), 2)
    equation3 = math.pow((math.sin(x)*math.tan(y)*math.tan(z) - 1/2), 2)
    score = equation1 + equation2 + equation3
    return score


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
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 3500
    CROSS_PROB = 0.7
    N_para = 3  # 变量个数
    M_parent = 10  # 杂交时父体个数
    K_top = 6  # 精英杂交算法中，选取topK个最好的个体作为父体
    L_son = 4  # 在子空间中生成L_son个新个体，选取其中一个与上一代的最差个体进行比较

    # 1.初始化种群
    start = time.perf_counter()
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    for k in range(N_GENERATION):
        # 3.计算种群个体的适应度
        fitness = cal_fitness(pop)  # 计算种群每个个体的适应值
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
