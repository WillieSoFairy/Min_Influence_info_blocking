import pandas as pd
from re import match
from queue import Queue
from copy import deepcopy
from random import randint


# 定义边的数据结构
class Edge:
    def __init__(self, to, prob):
        self.to = to  # 与该点相邻的点
        self.prob = prob  # 与该相邻点的传播概率


# 初始化模拟次数T
T = 100


# 读图
def load_graph(fileName):  # 文件名入参，返回邻接表、节点个数和边个数
    with open(fileName, "r") as f:
        str_list = f.readlines()

    # 读入第一行，获取节点和边的个数
    info = match(r"([0-9].*) ([0-9].*)\n", str_list[0])
    node_count = int(info.group(1))
    edge_count = int(info.group(2))

    # 读入图主体
    graph = [[] for i in range(node_count)]
    for x in str_list[1:]:
        match_str = match(r"([0-9].*) ([0-9].*) ([0-9].*)\n", x)
        start_p = int(match_str.group(1))
        dest_p = int(match_str.group(2))
        prob = float(match_str.group(3))
        graph[start_p].append(Edge(dest_p, prob))
    return (graph, node_count, edge_count)


# 读种子
def load_seeds(fileName):
    with open(fileName, "r") as f:
        str_list = f.readlines()
    return list(map(lambda x: int(match(r"([0-9]*)\n", x).group(1)), str_list))


# 传入概率的倒数取整，并生成1到该数的随机整数，则=1的概率即为该传播概率
def check(x):
    rand_num = randint(1, x)
    if rand_num == 1:
        return True
    return False


# 计算每个点的激活概率
def calc_node_prob(graph, node_count, seeds, T):
    activation_counts = [0 for i in range(node_count)]
    for i in range(T):
        actived_nodes = set(seeds)  # 初始化激活节点集合，包括种子节点
        # 迭代直到没有新的节点被激活
        while actived_nodes:
            new_activations = set()
            for node in list(actived_nodes):
                for neighbor in graph[node]:  # 遍历当前激活节点的邻居
                    # 如果邻居未被激活且存在对应的传播概率
                    if neighbor.to not in actived_nodes:
                        z = neighbor.prob
                        # 根据传播概率决定是否激活邻居
                        if (z != 0) and (check(int(1.0 / z))):
                            new_activations.add(neighbor.to)
                            activation_counts[neighbor.to] += 1
            actived_nodes = new_activations  # 更新激活节点集合
    activation_counts = map(lambda x: T if x >= T else x, activation_counts)
    activation_prob = [count / T for count in activation_counts]
    for seed in seeds:
        activation_prob[seed] = 1.0
    return activation_prob


# 生成节点表，包括id、点编号、是否种子顶点、激活概率
def create_node_dataframe(node_count, seeds, activation):
    node_idx = [i for i in range(node_count)]
    node_type = ["b" for i in range(node_count)]
    for seed in seeds:
        node_type[seed] = "a"
    nodes = pd.DataFrame(
        {
            "id": node_idx,
            "label": node_idx,
            "category": node_type,
            "PageRank": activation,
        }
    )
    nodes.loc[0, "label"] = node_count
    return nodes


# 生成边表，包括当前节点、相邻节点和边的传播概率
def create_edge_dataframe(graph):
    start_idx = []
    end_idx = []
    prob = []
    for idx, start in enumerate(graph):
        for dest in start:
            start_idx.append(idx)
            end_idx.append(dest.to)
            prob.append(dest.prob)
    edges = pd.DataFrame({"source": start_idx, "target": end_idx, "weight": prob})
    edges["type"] = "directed"
    return edges


# 计算传播期望
def calc_exp(graph: list[list[Edge]], seeds: list, node_count, T, block_nodes=[]):
    visit = [0 for i in range(node_count)]
    gain = 0
    for i in range(1, T + 1):
        gain_i = 1
        e_sample = [[] for j in range(node_count)]

        for j in range(node_count):
            for dest_n in graph[j]:
                x = j
                y = dest_n.to
                z = dest_n.prob
                if (z != 0) and check(int(1.0 / z)):
                    e_sample[x].append(y)
        Q = Queue()
        for seed in seeds:
            Q.put(seed)
            visit[seed] = i
        for remove in block_nodes:
            visit[remove] = i
        while not Q.empty():
            x = Q.get()
            for y in e_sample[x]:
                if visit[y] != i:
                    Q.put(y)
                    visit[y] = i
                    gain_i = gain_i + 1
        gain = gain + gain_i
    if gain:
        return gain / T
    else:
        return 0


# 按照预算找出使传播最小化的阻塞点
def seek_remove_nodes(graph, seeds, node_count, budget):
    remove_nodes = []
    remove_flag = [0 for i in range(node_count)]
    for seed in seeds:
        remove_flag[seed] = 1
    for i in range(budget):
        Min = float(node_count)
        for j in range(node_count):
            if remove_flag[j] == 0:
                remove_nodes.append(j)
                res = calc_exp(graph, seeds, node_count, int(T / 100), remove_nodes)
                if res < Min:
                    Min = res
                    remove_node = j
                remove_nodes.pop()
                remove_flag[i] = 0
        remove_flag[remove_node] = 1
        remove_nodes.append(remove_node)
    return remove_nodes


# 阻塞图中的点
def block_node(graph: list[list[Edge]], nodes):
    G = deepcopy(graph)
    for node in nodes:
        for i in G[node]:
            i.prob = 0
    return G


if __name__ == "__main__":
    # 读取图和种子
    graph_fileName = r"facebook_in.txt"
    seeds_fileName = r"ICSeed=facebook_in.txt"
    graph, node_count, edge_count = load_graph(graph_fileName)
    seeds = load_seeds(seeds_fileName)

    # 被阻塞前的传播期望
    begin_exp = calc_exp(graph, seeds, node_count, T)

    # 被阻塞前点的激活概率和边的传播概率
    begin_activation = calc_node_prob(graph, node_count, seeds, T)
    begin_nodes = create_node_dataframe(node_count, seeds, begin_activation)
    begin_edges = create_edge_dataframe(graph)

    # 设定预算并阻塞预算数量的点
    budget = 50
    block = seek_remove_nodes(graph, seeds, node_count, budget)

    # 被阻塞后的传播期望
    after_exp = calc_exp(graph, seeds, node_count, T, block)

    # 在图中阻塞点，并计算被阻塞后点的激活概率和边的传播概率
    G = block_node(graph, block)
    after_activation = calc_node_prob(G, node_count, seeds, T)
    after_nodes = create_node_dataframe(node_count, seeds, begin_activation)
    after_edges = create_edge_dataframe(G)

    # 导出数据表
    begin_nodes.to_csv("./begin_nodes.csv", index=False)
    begin_edges.to_csv("./begin_edges.csv", index=False)

    after_nodes.to_csv("./after_nodes.csv", index=False)
    after_edges.to_csv("./after_edges.csv", index=False)
