import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import math
from alive_progress import alive_bar
from tqdm import tqdm

"""
基础遗传算法
------------------
变量个数：1
编码方式：十进制编码
选择算子：1.轮盘赌  2.锦标赛
杂交算子：整体算术杂交（直接改变父母的基因位）   # another way：新生成的个体加入种群
变异算子：基因突变（将某些分量在其定义域内随机取值）
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


def initial_population(pop_size):
    """
    初始化种群
    :param pop_size: 种群个体数目
    :return:
    """
    population = []
    # 初始化生成范围在[-10,10]的pop_size个个体的十进制基因型种群
    for i in range(pop_size):
        chromosome = random.random() * 20 - 10
        population.append(chromosome)

    return population


def cal_fitness(population):
    """
    输入十进制染色体种群，计算种群中每个个体的适应值大小
    :param population:
    :return:
    """
    fitness = []
    for i in range(len(population)):
        x = population[i]
        tmp_fitness = score_function(x)
        fitness.append(tmp_fitness)
    return fitness


def find_max(population, fitness):
    max_fit = fitness[0]
    max_chromosome = 0
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal > max_fit:
            max_fit = tmpVal
            max_chromosome = population[i]
    return max_chromosome, max_fit


def find_min(population, fitness):
    min_fit = fitness[0]
    min_chromosome = 0
    for i in range(len(population)):
        tmpVal = fitness[i]
        if tmpVal < min_fit:
            min_fit = tmpVal
            min_chromosome = population[i]
    return min_chromosome, min_fit


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
            population[i] = random.random() * 20 - 10


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
    for i, male in enumerate(population):
        son1 = 0
        son2 = 0
        choice = random.random()  # 0-1之间的随机数
        if choice < cross_prob:
            # 若杂交，从种群中随机选取一个个体进行杂交
            # TODO: 杂交的结果是否应该算作新个体加入种群？还是直接替换父辈？
            female_pos = random.randint(0, POP_SIZE - 1)
            female = population[female_pos]
            # cross_pos = random.randint(0, CHOROMOSOME_LENGTH - 1)
            son1 = alpha * male + (1-alpha) * female
            son2 = alpha * female + (1-alpha) * male
            population[i] = son1
            population[female_pos] = son2


def wheel_select(population, fitness):
    """
    轮盘赌选择算子
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
    POP_SIZE = 200
    X_BOUND = [-10, 10]  # x取值范围
    N_GENERATION = 100
    CROSS_PROB = 0.7
    MUTE_PROB = 0.05
    players = 3  # 锦标赛算法每轮参赛选手数量

    # 1.初始化种群
    pop = initial_population(POP_SIZE)
    # 2.迭代N代
    results = []
    for k in tqdm(range(N_GENERATION)):
        # 3.交叉、变异
        crossover(population=pop, cross_prob=CROSS_PROB)
        mutation(population=pop, mute_prob=MUTE_PROB)
        # 4.计算种群个体的适应度
        fitness = cal_fitness(pop)  # 计算种群每个个体的适应值
        best_chromo, best_fitness = find_min(population=pop, fitness=fitness)
        avg_fitness = np.sum(fitness) / POP_SIZE
        results.append([best_fitness, best_chromo, avg_fitness])
        # 5.进行种群个体选择
        pop = champion_select(pop, fitness, players)

    min_fitness_index = np.argmin(fitness)
    print("min_fitness:", fitness[min_fitness_index])
    x = pop[min_fitness_index]
    print("min_x:", x)

    plot(results)