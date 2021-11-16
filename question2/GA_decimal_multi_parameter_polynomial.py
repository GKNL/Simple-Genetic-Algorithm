import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import math

"""
题目二：
基础遗传算法
------------------
变量个数：n
编码方式：十进制编码
选择算子：轮盘赌
杂交算子：1.部分算术杂交  2.部分离散杂交
变异算子：基因突变（将某些分量在其定义域内随机取值）
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
    # 初始化生成范围在(0,10)的pop_size个个体的十进制基因型种群
    for i in range(pop_size):
        per_chromosome = []
        for n in range(N_para):
            tmp_para = random.random() * 10
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
    max_chromosome_list = []
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal > max_fit:
            max_fit = tmpVal
            max_chromosome_list = population[i]
    return max_chromosome_list, max_fit


def find_min(population, fitness):
    min_fit = fitness[0]
    min_chromosome_list = []
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal < min_fit:
            min_fit = tmpVal
            min_chromosome_list = population[i]
    return min_chromosome_list, min_fit


def mutation(population, mute_prob=0.05):
    """
    按一定的变异概率，将某些分量在其定义域内随机取值
    :param population:
    :param mute_prob:变异概率
    :return:
    """
    for i, chrom in enumerate(population):
        choice = random.random()  # 0-1之间的随机数
        if choice < mute_prob:
            # 随机取一个变量，在取值范围内进行变异
            mut_pos = random.randint(0, len(chrom) - 1)
            population[i][mut_pos] = random.random() * 20 - 10


def arithmetic_crossover_part(population, cross_prob=0.4):
    """
    对种群中的个体，按照杂交概率进行部分算术杂交
    [直接对父母的染色体进行杂交更改]
    :param population:
    :param cross_prob:杂交概率
    :return:
    """
    # 生成算术杂交的alpha参数
    alphas = [random.random() for i in range(N_para)]
    for i, male in enumerate(population):
        son1 = copy.deepcopy(male)
        son2 = []
        choice = random.random()  # 0-1之间的随机数
        if choice < cross_prob:
            # 部分算术杂交的杂交点位
            cross_pos = random.randint(0, N_para - 1)
            # 若杂交，从种群中随机选取一个个体进行杂交
            female_pos = random.randint(0, POP_SIZE - 1)
            female = population[female_pos]
            son2 = copy.deepcopy(female)
            for n in range(cross_pos, N_para):
                male_para = male[n]
                female_para = female[n]
                son1[n] = male_para * alphas[n] + female_para * (1 - alphas[n])
                son2[n] = female_para * alphas[n] + male_para * (1 - alphas[n])
            population[i] = son1
            population[female_pos] = son2



def disperse_crossover_part(population, cross_prob=0.4):
    """
    对种群中的个体，按照杂交概率进行部分离散杂交
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
            # 部分离散杂交的杂交点位
            cross_pos = random.randint(0, N_para - 1)
            # 若杂交，从种群中随机选取一个个体进行杂交
            female = population[random.randint(0, POP_SIZE - 1)]
            for n in range(cross_pos, N_para):
                son1[:cross_pos] = male[:cross_pos]
                son1[cross_pos:] = female[cross_pos:]
                son2[:cross_pos] = female[:cross_pos]
                son2[cross_pos:] = male[cross_pos:]
                population[i] = son1
                population[cross_pos] = son2

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


def champion_select(population, fitness, K_player):
    """
    锦标赛选择算子
    对种群中的个体进行筛选，保留适应的个体
    :param population: 二进制种群
    :param fitness: 种群适应度矩阵
    :return:
    """
    pop = np.array(population)
    scores = np.array(fitness)
    selected_res = []
    for i in range(POP_SIZE):
        # 从种群中随机选两个个体，进行锦标赛选择
        index = np.random.choice(np.arange(len(population)), size=K_player, replace=True)
        player_scores = scores[index]
        players = pop[index]
        min_idx = np.argmin(player_scores)
        selected_res.append(players[min_idx])
    return selected_res


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
    POP_SIZE = 50
    X_BOUND = [0, 10]  # x取值范围
    N_GENERATION = 1000
    iter_nums = N_GENERATION  # 实际迭代次数
    CROSS_PROB = 0.7
    MUTE_PROB = 0.05
    N_para = 3  # 变量个数
    players = 4  # 锦标赛算法每轮参赛选手数量
    optimization = 5.317713118060457e-12

    # 1.初始化种群
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    minest = 10
    for k in range(N_GENERATION):
        # 3.交叉、变异
        arithmetic_crossover_part(population=pop, cross_prob=CROSS_PROB)
        mutation(population=pop, mute_prob=MUTE_PROB)
        # 4.计算种群个体的适应度
        fitness = cal_fitness(pop)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        avg_fitness = np.sum(fitness) / POP_SIZE
        results.append([best_fitness, best_chromo, avg_fitness])
        # 当最优值与优化目标接近时，结束演化
        if abs(best_fitness - optimization) < 1e-8:
            print('Reach the optimization object!Total iteration num: {}'.format(k + 1))
            iter_nums = k + 1
            break
        if best_fitness<minest:
            minest = best_fitness
        # 5.进行种群个体选择
        pop = champion_select(pop, fitness, players)

    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x = pop[min_fitness_index]
    print("min_x:", x)
    print("minest_fit:", minest)

    plot(results, iter_nums)