import numpy as np
import matplotlib.pyplot as plt

#代价函数为vec1与vec2之间距离
def price_cn(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# distance为距离矩阵，path为路径，计算某一路径的距离
def calc_distance(path, distance):
    dis = 0.0
    for i in range(len(path) - 1):
        dis += distance[path[i]][path[i+1]]
    return dis

#根据城市坐标，得到两城市之间的距离矩阵
def get_distance(citys):
    N = len(citys)
    distance = np.zeros((N, N))
    for i, row in enumerate(citys):
        line = []
        for j, col in enumerate(citys):
            if i != j:
                line.append(price_cn(row, col))
            else:
                line.append(0.0)
        distance[i] = line
    return distance

#动态方程计算微分方程du
def calc_du(V, distance, citys, A, D):
    N = len(citys)
    a = np.sum(V, axis=0) - 1 
    b = np.sum(V, axis=1) - 1 
    t1 = np.zeros((N, N))
    t2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            t1[i, j] = a[j]
            t2[j, i] = b[j]
    #将第一列移动至最后一列
    c_1 = V[ : , 1:N]
    c_0 = np.zeros((N, 1))
    c_0[ : , 0] = V[ : ,0]
    c = np.concatenate((c_1, c_0), axis=1)
    c = np.dot(distance, c)
    return - A * (t1 + t2) - D * c

#更新神经网络的输入电压U
def calc_U(U, du, step):
    return U + du * step

#更新神经网络的输出电压V
def calc_V(U, U0):
    return 1 / 2 * (1 + np.tanh(U / U0))

#计算能量函数
def calc_energy(V, distance, citys, A, D):
    N = len(citys)
    t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
    t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
    idx = [i for i in range(1, N)]
    idx = idx + [0]
    Vt = V[ : , idx]
    t3 = distance * Vt
    t3 = np.sum(np.sum(np.multiply(V, t3)))
    e = 1 / 2 * (A * (t1 + t2) + D * t3)
    return e

#检查路径的正确性
def check_path(V, citys):
    N = len(citys) 
    newV = np.zeros((N, N))
    route = []
    for i in range(N):
        mm = np.max(V[ : , i])
        for j in range(N):
            if V[j, i] == mm:
                newV[j, i] = 1
                route += [j]
                break
    return route, newV

#绘制哈密顿回路和能量趋势
def draw_H_and_E(citys, H_path, energys):
    fig = plt.figure(figsize=(16, 9))
    #绘制哈密顿回路
    ax1 = fig.add_subplot(211)
    for (from_, to_) in H_path:
        p1 = plt.Circle(citys[from_], 0.2, color='red')
        p2 = plt.Circle(citys[to_], 0.2, color='cyan')
        ax1.add_patch(p1)
        ax1.add_patch(p2)
        ax1.plot((citys[from_][0], citys[to_][0]), (citys[from_][1], citys[to_][1]), color='cyan')
        ax1.annotate(text=chr(97 + to_), xy=citys[to_], xytext=(-8, -4), textcoords='offset points', fontsize=20)
    ax1.axis('equal')
    ax1.grid()
    #绘制能量趋势图
    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(0, len(energys), 1), energys, color='red')
    plt.savefig('pic/tsp')
 
#各城市坐标
citys = np.array([[9,6],
                  [2, 6],
                  [2, 0],
                  [1, 4],
                  [4, 5],
                  [5, 8],
                  [6, 9],
                  [3, 2]])
# 计算各城市距离
distance = get_distance(citys) 
N = len(citys)
# 设置初始值
A = N * N
D = N / 2
U0 = 0.0009 # 初始电压
step = 0.0001 # 步长
num_iter = 10000 # 迭代次数
# 初始化神经网络输入状态（电路的输出电压为V）
U = 1 / 2 * U0 * np.log(N - 1) + (2 * (np.random.random((N, N))) - 1)
# 初始化神经网络输出状态（电路的输出电压为V）
V = calc_V(U, U0)
energys = np.array([0.0 for x in range(num_iter)]) # 每次迭代的能量
best_distance = np.inf # 初始化最优距离
best_route = [] # 最优路线
H_path = [] # 哈密顿回路

# 开始迭代训练网络
for n in range(num_iter):
    # 利用动态方程计算du
    du = calc_du(V, distance, citys, A, D)
    # 由一阶欧拉法更新下一个时间的输入状态（电路的输入电压U）
    U = calc_U(U, du, step)
    # 由sigmoid函数更新下一个时间的输出状态（电路的输出电压V）
    V = calc_V(U, U0)
    # 计算当前网络的能量E
    energys[n] = calc_energy(V, distance, citys, A, D)
    # 检查路径是否为次最优解
    route, newV = check_path(V, citys)
    if len(np.unique(route)) == N:
        route.append(route[0])
        dis = calc_distance(route, distance)
        if dis < best_distance:
            H_path = []
            best_distance = dis
            best_route = route
            for i in range(len(route) - 1):
                H_path.append((route[i], route[i+1]))
            print(f"第{n}次迭代找到的次最优解距离：{best_distance:.4f}，能量：{energys[n]:.4f}，路径：")
            [print(chr(97 + v), end=',' if i < len(best_route) - 1 else '\n') for i, v in enumerate(best_route)]
if len(H_path) > 0:
    draw_H_and_E(citys, H_path, energys)
else:
    print("没有找到最优解")