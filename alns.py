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
score = [1 for i in range(4)]
best_cost=HUGE
omega1 = 10  # 全局最佳
omega2 = 6  # 比当前解好
omega3 = 4  # 被接受了
omega4 = 0.8  # 被拒绝了
update_lambda = 0.75
temperature=1000
alpha=0.99
population_size=20

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


def random_remove_2node(path):
    # print(path)
    # print('random_remove_2node')
    if len(path) > 3:
        temp_path = path.copy()
        remove_list = []
        for i in range(2):
            length = len(temp_path) - 2
            select_pos = random.randint(1, length)
            remove_list.append(temp_path[select_pos])
            del temp_path[select_pos]
        return temp_path, remove_list
    else:
        return path, []


def remove_2max_node(path):
    # print('remove_2max_node')
    if len(path) > 3:
        temp_path = path.copy()
        remove_list = []
        for i in range(2):
            longest = 0
            longest_pos = -1
            for j, jj in enumerate(temp_path):
                if j < len(temp_path) - 2 and temp_path[j + 1] != 0:
                    if temp_path[j] == 0:
                        l = get_distance(X[temp_path[j]],
                                         Y[temp_path[j]], X[temp_path[j + 1]],
                                         Y[temp_path[j + 1]])
                        if l > longest:
                            longest = l
                            longest_pos = j + 1
                            # print('*1*',temp_path[longest_pos])
                    else:
                        l = get_distance(X[temp_path[j]],
                                         Y[temp_path[j]], X[temp_path[j + 1]],
                                         Y[temp_path[j + 1]])
                        if l > longest:
                            longest = l
                            longest_pos = j + 1
                            # print('*2*', temp_path[longest_pos])
            remove_list.append(temp_path[longest_pos])
            del temp_path[longest_pos]
            # print(remove_list)
        return temp_path, remove_list
    else:
        return path, []


def random_insert(path, remove_list):
    # print('random_insert')
    temp_path = path.copy()
    for i in range(len(remove_list)):
        length = len(temp_path) - 1
        select_pos = random.randint(1, length)
        temp_path.insert(select_pos, remove_list[i])
    return temp_path


def greedy_insert(path, remove_list):
    # print('greedy_insert')
    temp_path = path.copy()
    for i in range(len(remove_list)):
        best_pos = -1
        best_cost = HUGE*100
        for j in range(len(temp_path)):
            if j > 0:
                temp_temp_path = temp_path.copy()
                temp_temp_path.insert(j, remove_list[i])
                c1 = get_cost(temp_temp_path)
                if c1 < best_cost:
                    best_pos = j
                    best_cost = c1
        temp_path.insert(best_pos, remove_list[i])
    return temp_path


def local_search(path):
    cost = get_cost(path)
    global best_cost
    if random.random() < score[0] / (score[0] + score[1]):
        remove_select = 0
    else:
        remove_select = 1
    if random.random() < score[2] / (score[2] + score[3]):
        insert_select = 0
    else:
        insert_select = 1
    remove = [random_remove_2node, remove_2max_node]
    insert = [random_insert, greedy_insert]

    temp_path, remove_list = remove[remove_select](path.copy())
    temp_path = insert[insert_select](temp_path.copy(), remove_list)
    c = get_cost(temp_path)
    if c < cost:
        if c < best_cost:
            score[remove_select] = score[remove_select] * update_lambda + (1 - update_lambda) * omega1
            score[2 + insert_select] = score[2 + insert_select] * update_lambda + (1 - update_lambda) * omega1
            best_cost = c
        else:
            score[remove_select] = score[remove_select] * update_lambda + (1 - update_lambda) * omega2
            score[2 + insert_select] = score[2 + insert_select] * update_lambda + (1 - update_lambda) * omega2
        return temp_path
    else:
        delta = c - cost
        p = math.exp(-delta / temperature)
        r = random.random()
        if r < p:
            score[remove_select] = score[remove_select] * update_lambda + (1 - update_lambda) * omega3
            score[2 + insert_select] = score[2 + insert_select] * update_lambda + (1 - update_lambda) * omega3
            return temp_path
        else:
            score[remove_select] = score[remove_select] * update_lambda + (1 - update_lambda) * omega4
            score[2 + insert_select] = score[2 + insert_select] * update_lambda + (1 - update_lambda) * omega4
            return path

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

def select_sol(solution):
    select_num = int(len(solution) / 3)
    temp_solution = copy.deepcopy(solution[:select_num])
    return temp_solution

if __name__=='__main__':
    solutions = []
    for i, ii in enumerate(input_data):
        X.append(input_data[i][0])
        Y.append(input_data[i][1])
    for i in range(population_size):
        path = init_code(93)
        solutions.append(path)
    def key(s):
        return 1 / get_cost(s)  # key为选择条件,按照选择概率进行排序
    solutions.sort(reverse=True, key=key)
    cost = []
    for i in tqdm(range(1000)):
        temp_solutions = select_sol(solutions)
        for j in range(len(temp_solutions)):
            temp_solutions[j]= local_search(temp_solutions[j])
        solutions[population_size - len(temp_solutions):] = temp_solutions
        def key(s):
            return 1 / get_cost(s)  # key为选择条件,按照选择概率进行排序
        solutions.sort(reverse=True, key=key)
        cost.append(get_cost(solutions[0]))
        temperature *= alpha
    print(solutions[0])
    print(cost[-1])
    print(cost)
    plot(solutions[0])
    cost = normalize_cost(cost)
    plt.plot(cost)
    plt.show()
