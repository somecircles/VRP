import copy
import math
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

input_data = [[1, -1], [35, 35], [41, 49], [35, 17], [55, 45], [15, 30], [25, 30], [10, 43], [55, 60], [30, 60],
              [20, 65], [50, 35], [30, 25], [15, 10], [30, 5], [10, 20], [5, 30], [20, 40], [15, 60], [45, 65],
              [45, 20], [45, 10], [55, 5], [65, 35], [65, 20], [45, 30], [35, 40], [64, 42], [40, 60], [35, 69],
              [53, 52], [65, 55], [63, 65], [2, 60], [20, 20], [5, 5], [60, 12], [40, 25], [42, 7], [24, 12], [23, 3],
              [11, 14], [6, 38], [2, 48], [8, 56], [6, 68], [47, 47], [27, 43], [37, 31], [57, 29], [63, 23], [53, 12],
              [32, 12], [21, 24], [17, 34], [12, 24], [24, 58], [27, 69], [15, 77], [62, 77], [49, 73], [67, 5],
              [56, 39], [37, 47], [37, 56], [57, 68], [47, 16], [44, 17], [46, 13], [49, 11], [49, 42], [53, 43],
              [61, 52], [57, 48], [56, 37], [55, 54], [15, 47], [14, 37], [11, 31], [4, 18], [28, 18], [26, 52],
              [26, 35], [31, 67], [15, 19], [22, 22], [18, 24], [26, 27], [25, 24], [22, 27], [25, 21], [19, 21],
              [20, 26], [18, 18]]
demand = [0, 10, 10, 7, 13, 26, 3, 9, 16, 16, 12, 19, 23, 20, 8, 19, 2, 12, 17, 9, 11, 18, 29, 3, 6, 17, 16, 9, 21, 23,
          11, 14, 8, 5, 8, 16, 31, 9, 5, 5, 7, 18, 16, 1, 27, 30, 13, 9, 14, 18, 2, 6, 7, 28, 3, 13, 19, 10, 9, 20, 25,
          25, 36, 6, 5, 15, 25, 9, 8, 18, 13, 14, 3, 23, 6, 26, 16, 11, 7, 35, 26, 9, 15, 3, 1, 2, 22, 27, 20, 11, 12,
          10, 9, 17]
X=[]
Y=[]
capa=150
HUGE=999999999
population_size=60
vary_num=10
vary_possibility=0.1
temperature=1000
alpha=0.98

def get_distance(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2)

def get_cost(path):
    distance=0
    sum=0
    for i,ii in enumerate(path):
        if i<len(path)-1:
            distance+=get_distance(X[path[i]],Y[path[i]],X[path[i+1]],Y[path[i+1]])
            sum+=demand[path[i]]
            if sum>capa:
                distance+=HUGE
            if path[i]==0:
                sum=0
    return distance

def init_code(length):
    init_path=[]
    for i in range(1,length+1):
        init_path.append(i)
    random.shuffle(init_path)
    init_path.insert(0,0)
    init_path.append(0)
    init_path=insert_deport(init_path)
    return init_path

def insert_deport(path):
    sum=0
    for i,ii in enumerate(path):#ii=path[i],copy.deepcopy(path)
        sum+=demand[path[i]]
        if sum>capa:
            path.insert(i,0)
            sum = demand[path[i]]

    return path

def moveRandSubPathLeft(path):  # 任选一段路径放在最前面
    count = random.randrange(path.count(0) - 1)
    index = path.index(0, count + 1)
    locToInsert = 0
    path.insert(locToInsert, path.pop(index))
    index += 1
    locToInsert += 1
    while path[index] != 0:
        path.insert(locToInsert, path.pop(index))
        index += 1
        locToInsert += 1


def crossPair(path1, path2, crossed_paths):  # 交叉
    moveRandSubPathLeft(path1)  # 对染色体1，选择子路径，然后放前面
    moveRandSubPathLeft(path2)  # 对染色体2，选择子路径，然后放前面

    new_path1 = []
    new_path2 = []

    # 复制子路径1
    centers = 0
    first_pos1 = 1
    for pos in path1:
        first_pos1 += 1
        centers += (pos == 0)
        new_path1.append(pos)
        if centers >= 2:
            break
    # 复制子路径1
    centers = 0
    first_pos2 = 1
    for pos in path2:
        first_pos2 += 1
        centers += (pos == 0)
        new_path2.append(pos)
        if centers >= 2:
            break

    # 将父代染色体１的子路经Ａ作为子代染色体１的一部分，
    # 同时将父代染色体２中子路经Ａ没有的编码按照父代染色体２中的顺序添加到子路经Ａ的后面，
    # 在染色体的末尾添加编码０；
    for pos in path2:
        if pos not in new_path1:
            new_path1.append(pos)
    for pos in path1:
        if pos not in new_path2:
            new_path2.append(pos)
    # add center at end
    new_path1.append(0)
    new_path2.append(0)

    possible = []
    while path1[first_pos1] != 0:
        newGene = new_path1.copy()
        newGene.insert(first_pos1, 0)
        possible.append(newGene)
        first_pos1 += 1

    def key(path):
        return 1 / get_cost(path)

    possible.sort(reverse=True, key=key)  # 计算适应度最高的
    if len(possible) != 0:
        crossed_paths.append(possible[0])

    possible = []
    while path2[first_pos2] != 0:
        newGene = new_path2.copy()
        newGene.insert(first_pos2, 0)
        possible.append(newGene)
        first_pos2 += 1

    def key(path):
        return 1 / get_cost(path)

    possible.sort(reverse=True, key=key)  # 计算适应度最高的
    if len(possible) != 0:
        crossed_paths.append(possible[0])

def cross(solutions):  # 所有基因的交叉
    crossed_paths = []
    for i in range(0, len(solutions), 2):
        crossPair(solutions[i], solutions[i + 1], crossed_paths)
    return crossed_paths

def vary_one(path):  # 对特定染色体的变异操作
    varied_paths = []
    for i in range(vary_num):
        new_path = path.copy()
        p1, p2 = random.choices(list(range(1, len(path) - 2)), k=2)
        new_path[p1], new_path[p2] = new_path[p2], new_path[p1]  # 交换
        varied_paths.append(new_path)

    def key(path):
        return 1 / get_cost(path)

    varied_paths.sort(reverse=True, key=key)
    return varied_paths[0]


def vary(solutions):  # 变异
    new_solutions = []
    for j, jj in enumerate(solutions):
        if j < len(solutions) / 3:
            new_solutions.append(solutions[j])
            continue
        else:
            if random.random() <= vary_possibility:
                temp_solution = vary_one(solutions[j])
                delta = get_cost(temp_solution) - get_cost(solutions[j])
                if delta <= 0:
                    new_solutions.append(temp_solution)
                else:
                    p = math.exp(-delta / temperature)
                    r = random.random()
                    if r < p:
                        new_solutions.append(temp_solution)
                    else:
                        new_solutions.append(solutions[j])

            else:
                new_solutions.append(solutions[j])
    return new_solutions

def select_sol(solution):
    select_num = int(len(solution) / 3)
    temp_solution = copy.deepcopy(solution[:select_num])
    return temp_solution

def plot(path):
    x = []
    y = []
    for i, ii in enumerate(path):
        x.append(X[path[i]])
        y.append(Y[path[i]])
        if ii==0:
            plt.plot(x, y)
            x=[X[ii]]
            y=[Y[ii]]

    for i, ii in enumerate(path):
        plt.scatter(X[ii], Y[ii])
    plt.show()
    plt.clf()  # 清图。
    plt.cla()  # 清坐标轴。
    plt.close()  # 关窗口

def normalize_cost(cost):
    new_cost = []
    for i, ii in enumerate(cost):
        if ii < HUGE / 10:
            new_cost.append(ii)
            break
    for i, ii in enumerate(cost):
        if i > 0:
            # print(ii)
            if ii > HUGE / 10:
                new_cost.append(new_cost[i - 1])
            else:
                new_cost.append(ii)
    return new_cost

if __name__=='__main__':
    solutions=[]
    for i, ii in enumerate(input_data):
        X.append(input_data[i][0])
        Y.append(input_data[i][1])
    for i in range(population_size):
        path = init_code(93)
        solutions.append(path)

    def key(s):
        return 1 / get_cost(s)  # key为选择条件,按照选择概率进行排序
    solutions.sort(reverse=True, key=key)

    cost=[]
    for i in tqdm(range(1000)):
        temp_solutions = select_sol(solutions)
        temp_solutions = cross(temp_solutions)
        for j in range(len(temp_solutions)):
            delta = get_cost(temp_solutions[j])-get_cost(solutions[population_size - len(temp_solutions)+j])
            if delta<=0:
                solutions[population_size - len(temp_solutions)+j] = temp_solutions[j]
            else:
                p = math.exp(-delta / temperature)
                r = random.random()
                if r < p:
                    solutions[population_size - len(temp_solutions) + j] = temp_solutions[j]
        def key(s):
            return 1 / get_cost(s)  # key为选择条件,按照选择概率进行排序
        solutions.sort(reverse=True, key=key)
        solutions = vary(solutions)
        def key(s):
            return 1 / get_cost(s)  # key为选择条件,按照选择概率进行排序
        solutions.sort(reverse=True, key=key)
        cost.append(get_cost(solutions[0]))
    print(solutions[0])
    print(cost[-1])
    print(cost)
    plot(solutions[0])
    cost = normalize_cost(cost)
    plt.plot(cost)
    plt.show()
