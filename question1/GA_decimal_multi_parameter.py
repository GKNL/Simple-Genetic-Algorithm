import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import math

"""
基础遗传算法
------------------
变量个数：n
编码方式：十进制编码
选择算子：轮盘赌
杂交算子：整体算术杂交（直接改变父母的基因位）   # another way：新生成的个体加入种群
变异算子：基因突变（将某些分量在其定义域内随机取值）
"""

def score_function(x):
    """
    目标函数
    :param x: 单个染色体（十进制表现型）
    :return: 目标函数值大小
    """
    y = 0
    for j in range(1, 6, 1):
        tmp = j * math.cos((j + 1) * x + j)
        y += tmp
    return y


def initial_population(pop_size):
    """
    初始化种群(为每个变量创建一个种群)
    :param pop_size: 种群个体数目
    :return:
    """
    population = []
    # 初始化生成范围在[-10,10]的pop_size个个体的十进制基因型种群
    for n in range(N_para):
        sub_pop = []
        for i in range(pop_size):
            chromosome = random.random() * 20 - 10
            sub_pop.append(chromosome)
        population.append(sub_pop)

    return population


def cal_fitness(population):
    """
    输入N个变量的十进制染色体种群，计算种群中每个个体的适应值大小
    :param population:
    :return:
    """
    fitness = []
    for n in range(N_para):
        sub_fit = []
        per_pop = population[n]
        for i in range(len(per_pop)):
            x = per_pop[i]
            tmp_fitness = score_function(x)
            sub_fit.append(tmp_fitness)
        fitness.append(sub_fit)
    return fitness


def find_max(population, fitness):
    max_fit = []
    max_chromosome = []
    for n in range(N_para):
        tmp_max_chrom = 0
        tmp_max_fit = 0
        sub_fit = fitness[n]
        for i in range(len(sub_fit)):
            tmpVal = sub_fit[i]
            if tmpVal > tmp_max_fit:
                tmp_max_fit = tmpVal
                tmp_max_chrom = population[n][i]
        max_fit.append(tmp_max_fit)
        max_chromosome.append(tmp_max_chrom)
    return max_chromosome, max_fit


def find_min(population, fitness):
    min_fit = []
    min_chromosome = []
    for n in range(N_para):
        tmp_min_chrom = 0
        tmp_min_fit = 0
        sub_fit = fitness[n]
        for i in range(len(sub_fit)):
            tmpVal = sub_fit[i]
            if tmpVal < tmp_min_fit:
                tmp_min_fit = tmpVal
                tmp_min_chrom = population[n][i]
        min_fit.append(tmp_min_fit)
        min_chromosome.append(tmp_min_chrom)
    return min_chromosome, min_fit


def mutation(population, mute_prob=0.05):
    """
    按一定的变异概率，将某些分量在其定义域内随机取值
    :param population:
    :param mute_prob:变异概率
    :return:
    """

    for n in range(N_para):
        sub_pop = population[n]
        for i, chrom in enumerate(sub_pop):
            choice = random.random()  # 0-1之间的随机数
            if choice < mute_prob:
                population[n][i] = random.random() * 20 - 10


# def crossover(population, cross_prob=0.4):
#     """
#     对种群中的个体，按照杂交概率进行杂交
#     [将新产生的个体作为新的个体加入种群]
#     :param population:
#     :param cross_prob:杂交概率
#     :return:
#     """
#     for male in population:
#         son = copy.deepcopy(male)
#         choice = random.random()  # 0-1之间的随机数
#         if choice < cross_prob:
#             # TODO: 杂交的结果是否应该算作新个体加入种群？还是直接替换父辈？
#             # 若杂交，从种群中随机选取一个个体进行杂交
#             female = population[random.randint(0, POP_SIZE - 1)]
#             cross_pos = random.randint(0, CHOROMOSOME_LENGTH - 1)
#             son[cross_pos:] = female[cross_pos:]
#             population.append(son)


def crossover(population, cross_prob=0.7, alpha=0.3):
    """
    对种群中的个体，按照杂交概率进行算术杂交
    [直接对父母的染色体进行杂交更改]
    :param population:
    :param cross_prob:杂交概率
    :return:
    """
    for n in range(N_para):
        sub_pop = population[n]
        for i, male in enumerate(sub_pop):
            son1 = 0
            son2 = 0
            choice = random.random()  # 0-1之间的随机数
            if choice < cross_prob:
                # 若杂交，从种群中随机选取一个个体进行杂交
                # TODO: 杂交的结果是否应该算作新个体加入种群？还是直接替换父辈？
                female_pos = random.randint(0, POP_SIZE - 1)
                female = sub_pop[female_pos]
                # cross_pos = random.randint(0, CHOROMOSOME_LENGTH - 1)
                son1 = alpha * male + (1-alpha) * female
                son2 = alpha * female + (1-alpha) * male
                population[n][i] = son1
                population[n][female_pos] = son2


def select(population, fitness):
    """
    对种群中的个体进行筛选，保留适应的个体
    【这里是求解最小值，因此需要将问题的适应度函数进行转换（如采用倒数或相反数）！！！】
    :param population: 二进制种群
    :param fitness: 种群适应度矩阵
    :return:
    """
    # 找出种群中的最大适应度值
    _, max_fitness = find_max(population, fitness)
    # 更新适应度值，使其大于0
    select_pop = []
    for n in range(N_para):
        per_fitness = fitness[n]
        per_pop = population[n]
        total = 0
        adj_fitness = []
        for i in range(len(per_fitness)):
            tmp_fit = per_fitness[i]
            adjust_fit = max_fitness[n] - tmp_fit + 1e-3  # 最后在加上一个很小的数防止出现为0的适应度
            adj_fitness.append(adjust_fit)
            total += adjust_fit
        # 对调整后的适应值进行正则化，得到选择概率矩阵
        probabilities = []
        for i in range(len(adj_fitness)):
            probabilities.append(adj_fitness[i] / total)
        # 依概率随机筛选下一代个体
        # TODO: np.arange里？应该是新的population的size吧
        index = np.random.choice(np.arange(len(per_pop)), size=POP_SIZE, replace=True, p=probabilities)
        per_selected_res = []
        for idx in index:
            per_selected_res.append(per_pop[idx])
        select_pop.append(per_selected_res)
    return select_pop


def plot(results):
    X = []
    Y = []

    for i in range(N_GENERATION):
        X.append(i)
        Y.append(results[i][0])

    plt.plot(X, Y)
    plt.show()


if __name__ == '__main__':
    POP_SIZE = 200
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 100
    CROSS_PROB = 0.7
    MUTE_PROB = 0.05
    N_para = 2  # 变量个数

    # 1.初始化种群
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    for k in range(N_GENERATION):
        # 3.交叉、变异
        crossover(population=pop, cross_prob=CROSS_PROB)
        mutation(population=pop, mute_prob=MUTE_PROB)
        # 4.计算种群个体的适应度
        fitness = cal_fitness(pop)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        results.append([best_fitness, best_chromo])
        # 5.进行种群个体选择
        pop = select(pop, fitness)

    for n in range(N_para):
        per_fitness = fitness[n]
        per_pop = pop[n]

        min_fitness_index = np.argmin(per_fitness)
        print("min_fitness_{}:".format(n+1), per_fitness[min_fitness_index])
        x = per_pop[min_fitness_index]
        print("min_x_{}:".format(n+1), x)

    # plot(results)