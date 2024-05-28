# 社交网络中的影响力最小化

## 实验目标

本实验旨在利用Gephi软件对Facebook社交网络图进行可视化，并研究种子顶点在社交网络中的传播效果以及阻塞对传播的影响。

## 实验准备

**准备实验数据：**

* Facebook社交网络图数据 `facebook_in.txt`
* 种子顶点数据 `ICSeed=facebook_in.txt`

## 实验要求

### 一、数据准备

1. 修改实验代码（前几次实验的代码），使其能够读取`ICSeed=facebook_in.txt`中的种子，并导出每个顶点被种子顶点影响的概率以及阻塞后每个顶点被种子顶点影响的概率。
2. 将所有数据转换为CSV格式文件。

### 二、可视化过程

1. 利用Gephi软件可视化Facebook图结构。
2. 根据给定的种子顶点数据展示种子顶点在社交网络中的传播结果。
3. 展示阻塞顶点阻塞后种子顶点传播的结果。
4. 调整节点大小、颜色等属性，以突出显示不同的顶点受种子顶点影响的概率。

### 三、结果分析与总结

1. 分析可视化结果，观察种子顶点在社交网络中的传播效果以及阻塞对传播的影响。
2. 总结实验结果，提出可能的改进和进一步研究方向。

## 实验详情（使用python实现）

### 1. 导入相关的库


```python
import pandas as pd
from re import match
from queue import Queue
from copy import deepcopy
from random import randint
```

### 2. 定义边的数据结构，并初始化模拟次数`T`


```python
class Edge:
    def __init__(self, to, prob):
        self.to = to  # 与该点相邻的点
        self.prob = prob  # 与该相邻点的传播概率


T = 1000
```

### 3. 从文件中读入图和种子节点

#### 函数实现：读图和种子（正则表达式匹配文本）


```python
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


def load_seeds(fileName):
    with open(fileName, "r") as f:
        str_list = f.readlines()
    return list(map(lambda x: int(match(r"([0-9]*)\n", x).group(1)), str_list))
```

#### 读取图和种子


```python
graph_fileName = r"facebook_in.txt"
seeds_fileName = r"ICSeed=facebook_in.txt"
graph, node_count, edge_count = load_graph(graph_fileName)
seeds = load_seeds(seeds_fileName)
```

### 4. 计算未被阻塞前，每个点的激活概率

#### 函数实现：概率检查和计算激活概率


```python
def check(x):  # 传入概率的倒数取整，并生成1到该数的随机整数，则=1的概率即为该传播概率
    rand_num = randint(1, x)
    if rand_num == 1:
        return True
    return False
```


```python
def calc_node_prob(graph, node_count, seeds, T):  # 计算每个点的激活概率
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
```

#### 函数实现：生成适用于Gephi导入的节点表和边表


```python
def create_node_dataframe(node_count, seeds, activation):  # 生成节点表，包括id、点编号、是否种子顶点、激活概率
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


def create_edge_dataframe(graph):  # 生成边表，包括当前节点、相邻节点和边的传播概率
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
```


```python
begin_activation = calc_node_prob(graph, node_count, seeds, T)
begin_nodes = create_node_dataframe(node_count, seeds, begin_activation)
begin_edges = create_edge_dataframe(graph)
```

#### 初始的节点和边

* 节点


```python
begin_nodes
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>id</th>
      <th>label</th>
      <th>category</th>
      <th>PageRank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4039</td>
      <td>a</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4034</td>
      <td>4034</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4035</td>
      <td>4035</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4036</td>
      <td>4036</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4037</td>
      <td>4037</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4038</td>
      <td>4038</td>
      <td>b</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
<p>4039 rows × 4 columns</p>



* 边


```python
begin_edges
```




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>source</th>
      <th>target</th>
      <th>weight</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>4</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4024</td>
      <td>4038</td>
      <td>0.1111</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4027</td>
      <td>4031</td>
      <td>0.0526</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>1830</td>
      <td>0.1000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>4032</td>
      <td>0.5000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>4038</td>
      <td>0.1111</td>
      <td>directed</td>
    </tr>
  </tbody>
</table>
<p>88234 rows × 4 columns</p>



### 5. 计算被阻塞前传播期望

#### 函数实现：计算传播期望


```python
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
```

#### 被阻塞前的传播期望


```python
begin_exp = calc_exp(graph, seeds, node_count, T)
begin_exp
```




    2207.36



### 6. 根据预算计算被阻塞的点，以及其传播期望和各点的激活概率

#### 函数实现：按照预算找出使传播最小化的阻塞点


```python
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
```

#### 函数实现：阻塞图中的点


```python
def block_node(graph: list[list[Edge]], nodes):
    G = deepcopy(graph)
    for node in nodes:
        for i in G[node]:
            i.prob = 0
    return G
```

#### 计算阻塞点


```python
budget = 50
block = seek_remove_nodes(graph, seeds, node_count, budget)
block
```




    [3504,
     644,
     2315,
     3439,
     1984,
     401,
     1057,
     3218,
     1745,
     582,
     423,
     1418,
     1962,
     1393,
     2373,
     1087,
     1372,
     188,
     302,
     2117,
     1013,
     1325,
     682,
     795,
     2585,
     2269,
     1255,
     402,
     3416,
     3651,
     950,
     619,
     3542,
     651,
     1236,
     3532,
     1193,
     1741,
     3283,
     1361,
     641,
     53,
     1869,
     32,
     706,
     2287,
     1353,
     2121,
     2010,
     1948]



#### 阻塞上述点，并计算传播期望和各点的激活概率


```python
G = block_node(graph, block)
after_activation = calc_node_prob(G, node_count, seeds, T)
after_nodes = create_node_dataframe(node_count, seeds, begin_activation)
after_edges = create_edge_dataframe(G)
```

##### 被阻塞后的传播期望


```python
after_exp = calc_exp(graph, seeds, node_count, T, block)
after_exp
```




    2137.97



##### 被阻塞后的节点和边

* 节点


```python
after_nodes
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>id</th>
      <th>label</th>
      <th>category</th>
      <th>PageRank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>4039</td>
      <td>a</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>3</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>4</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4034</td>
      <td>4034</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4035</td>
      <td>4035</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4036</td>
      <td>4036</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4037</td>
      <td>4037</td>
      <td>b</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>4038</td>
      <td>4038</td>
      <td>b</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
<p>4039 rows × 4 columns</p>



* 边


```python
after_edges
```



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>source</th>
      <th>target</th>
      <th>weight</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>2</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>4</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>0</td>
      <td>5</td>
      <td>1.0000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>4024</td>
      <td>4038</td>
      <td>0.1111</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4027</td>
      <td>4031</td>
      <td>0.0526</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>1830</td>
      <td>0.1000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>4032</td>
      <td>0.5000</td>
      <td>directed</td>
    </tr>
    <tr>
      <td>4028</td>
      <td>4038</td>
      <td>0.1111</td>
      <td>directed</td>
    </tr>
  </tbody>
</table>
<p>88234 rows × 4 columns</p>



### 7. 导出 `.csv` 文件，并导入Gephi处理

#### 导出数据表


```python
begin_nodes.to_csv("./begin_nodes.csv", index=False)
begin_edges.to_csv("./begin_edges.csv", index=False)

after_nodes.to_csv("./after_nodes.csv", index=False)
after_edges.to_csv("./after_edges.csv", index=False)
```

#### 数据可视化

* 被阻塞前

  ![begin](./begin.png)

* 被阻塞后

  ![after](./after.png)

## 实验总结

### 为什么使用python？

想要细化实验的步骤，使用解释型语言，搭配Jupyter Notebook，将每一个步骤拆分成一个单独的函数，并输出该步骤的结果，能更灵活地研究每一步变量的输入和输出对实验整体的影响。

### 使用python的局限

非常慢！可能也是算法设计的问题，寻找50个最佳的阻塞点（且模拟次数仅为10次），耗时210分钟。

使用python的局限，也是整个实验的局限。只能寻找少量的点，否则耗时将会爆表。**所以，本次实验，阻塞的效果并不理想**，阻塞50个点，只减弱了约60个点的传播效果。

### 还有一些踩过的坑

1. 计算阻塞后的传播期望，本想直接在图中阻塞点（把该点的传播概率定义为0）以后，直接计算该图的期望，但是复现的结果总是偏大。发现论文的阻塞，是将该点出和入的传播路径都阻塞（即完全不计算该点）；而我的阻塞，是将出的路径设为0，但是计算期望时，依然还会将入的传播概率纳入计算范围。
2. 计算每个点的激活概率，本来设想是通过计算种子到该点的所有路径，并将其所有路径可通达的概率合计出A点到B点的概率。但是，写到后面，突然发现种子顶点不止一个，最后还是用了模拟的方式。

### 未来改进

* 算法角度：寻找阻塞点，目前只采用贪心算法，考虑未来改进使用支配树算法；
* 工具角度：部分算法代码使用C++重构；
* 运算角度：改进运行调度，使用多进程，并行计算；
* 硬件角度：考虑租用云服务器，用更高性能的计算机计算。

### 有待探索

阻塞点的数量和其边际成本的关系，寻找阻塞效果与阻塞成本的平衡点。

设计实验从阻塞50个点开始，每次递增50个，直至阻塞所有的点，计算出每一次的传播期望，并求出与初始未阻塞的比值。

绘制图表，x轴为阻塞点数，y轴为控制传播比。
