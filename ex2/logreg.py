import numpy as np
import sympy as sp

def run(data):

    # data[0]为1，表示偏差，剩下n-1个样本维度，以及最后一个为y值
    data = np.hstack((np.ones([data.shape[0], 1]), data))
    [sampleNum, thetaNum] = data.shape
    y = sp.symbols('y')
    # X=sp.Matrix([thetaNum-1])
    X = sp.symbols('X')
    # 设立线性模型参数向量
    # theta对应1个偏差，thetaNum-1个样本维度参数

    theta = sp.symbols('theta:' + str(thetaNum))
    # theta = sp.Matrix([thetaNum])

    # 线性模型

    f = theta * X
    cost = (f - y) ** 2 / 2
    g = sp.diff(cost, theta)
    print(g)
    f
    # 引入数据，计算代价，根据代价函数梯度下降调整参数
    for i in range(sampleNum):
        sample = data[i]
        # 计算代价函数
        cost(theta, sample)
        # 计算梯度
        grad()
    # 可以考虑在代价函数引入正则化项
