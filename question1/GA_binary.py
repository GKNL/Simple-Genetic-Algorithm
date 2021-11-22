import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import math

"""
基础遗传算法
------------------
变量个数：1
编码方式：二进制编码
选择算子：轮盘赌
杂交算子：单点杂交【直接改变父母的基因位】   # another way：新生成的个体加入种群
变异算子：基因突变
"""

def random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


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


def initial_population(pop_size, gene_length):
    """
    初始化种群
    :param pop_size: 种群个体数目
    :param gene_length: 单个个体染色体的大小
    :return:
    """
    population = []
    # 初始化生成gene_length大小的pop_size个个体的二进制基因型种群
    for i in range(pop_size):
        chromosome = []
        for j in range(gene_length):
            tmp_gene = random.randint(0, 1)
            chromosome.append(tmp_gene)
        population.append(chromosome)

    return population


def transform(population):
    """
    将种群中的所有染色体，从二进制转换为十进制表示
    :param population: 二进制种群
    :return:
    """

    trans_pop = []
    for i in range(len(population)):
        value = 0
        for j in range(CHOROMOSOME_LENGTH):
            # 对每位求2的幂，再求和
            value += population[i][j] * math.pow(2, j)
        # 将染色体表现型的值约束在[-10,10]范围内
        value = value / (math.pow(2, CHOROMOSOME_LENGTH) - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]  # 映射为x范围内的数
        trans_pop.append(value)

    return trans_pop


def cal_fitness(trans_pop):
    """
    输入十进制染色体种群，计算种群中每个个体的适应值大小
    :param trans_pop:
    :return:
    """
    fitness = []
    for i in range(len(trans_pop)):
        x = trans_pop[i]
        # # 将染色体表现型的值约束在[-10,10]范围内
        # x = tmp_val / (math.pow(2, CHOROMOSOME_LENGTH) - 1) * (X_BOUND[1] - X_BOUND[0]) + X_BOUND[0]  # 映射为x范围内的数
        tmp_fitness = score_function(x)
        fitness.append(tmp_fitness)
    return fitness


def find_max(population, fitness):
    max_fit = fitness[0]
    max_chromosome = []
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal > max_fit:
            max_fit = tmpVal
            max_chromosome = population[i]
    return max_chromosome, max_fit


def find_min(population, fitness):
    min_fit = fitness[0]
    min_chromosome = []
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal < min_fit:
            min_fit = tmpVal
            min_chromosome = population[i]
    return min_chromosome, min_fit


def mutation(population, mute_prob=0.05):
    """
    对种群中的个体的每一个基因位，按一定的概率进行变异
    :param population:
    :param mute_prob:变异概率
    :return:
    """
    # for chrom in population:
    #     choice = random.random()  # 0-1之间的随机数
    #     # 符合变异要求
    #     if choice < mute_prob:
    #         pos = random.randint(0, len(chrom)-1)
    #         val = chrom[pos]
    #         chrom[pos] = val ^ 1  # 取反

    for chrom in population:
        for j in range(len(chrom)):
            val = chrom[j]
            choice = random.random()  # 0-1之间的随机数
            if choice < mute_prob:
                chrom[j] = val ^ 1  # 取反


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


def crossover(population, cross_prob=0.4):
    """
    对种群中的个体，按照杂交概率进行杂交
    [直接对父母的染色体进行杂交更改]
    :param population:
    :param cross_prob:杂交概率
    :return:
    """
    for i, male in enumerate(population):
        son1 = []
        son2 = []
        choice = random.random()  # 0-1之间的随机数
        if choice < cross_prob:
            # 若杂交，从种群中随机选取一个个体进行杂交
            # TODO: 杂交的结果是否应该算作新个体加入种群？还是直接替换父辈？
            female_pos = random.randint(0, POP_SIZE - 1)
            female = population[female_pos]
            cross_pos = random.randint(0, CHOROMOSOME_LENGTH - 1)
            son1[:cross_pos] = male[:cross_pos]
            son1[cross_pos:] = female[cross_pos:]
            son2[:cross_pos] = female[:cross_pos]
            son2[cross_pos:] = male[cross_pos:]
            population[i] = son1
            population[female_pos] = son2


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
    total = 0
    adj_fitness = []
    for i in range(len(fitness)):
        tmp_fit = fitness[i]
        adjust_fit = max_fitness - tmp_fit + 1e-3  # 最后在加上一个很小的数防止出现为0的适应度
        adj_fitness.append(adjust_fit)
        total += adjust_fit

    # 对调整后的适应值进行正则化，得到选择概率矩阵
    probabilities = []
    for i in range(len(adj_fitness)):
        probabilities.append(adj_fitness[i] / total)
    # 依概率随机筛选下一代个体
    # TODO: np.arange里？应该是新的population的size吧
    index = np.random.choice(np.arange(len(population)), size=POP_SIZE, replace=True, p=probabilities)
    selected_res = []
    for idx in index:
        selected_res.append(population[idx])
    return selected_res


def plot(results):
    """

    :param results: 其中每一个元素为[best_fitness, best_chromo, avg_fitness]
    :param iter_nums:
    :return:
    """
    X = []
    Y_best = []
    Y_avg = []

    for i in range(N_GENERATION):
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
    CHOROMOSOME_LENGTH = 20  # 种群染色体大小
    POP_SIZE = 200
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 100
    CROSS_PROB = 0.7
    MUTE_PROB = 0.05

    # 1.初始化种群
    pop = initial_population(POP_SIZE, CHOROMOSOME_LENGTH)
    # 2.迭代N代
    results = []
    for k in range(N_GENERATION):
        # 3.交叉、变异
        crossover(population=pop, cross_prob=CROSS_PROB)
        mutation(population=pop, mute_prob=MUTE_PROB)
        # 4.计算种群个体的适应度
        trans_pop = transform(pop)  # 基因型转化为表现型
        fitness = cal_fitness(trans_pop)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        avg_fitness = np.sum(fitness) / POP_SIZE
        results.append([best_fitness, best_chromo, avg_fitness])
        # 5.进行种群个体选择
        pop = select(pop, fitness)

    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x = trans_pop[min_fitness_index]
    print("min_x:", x)

    plot(results)