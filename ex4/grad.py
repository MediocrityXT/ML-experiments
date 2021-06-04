import numpy as np


class LossFunc:
    def __init__(self, w_init):
        # 初始化参数，没有数据
        self.w = w_init  # 785 * 10 每个类别都有一个785*1的参数列向量
        # 用v保存动量
        self.v = 0
        self.m = 0
        self.iterTime = 0

    def inputData(self, x, y):
        # 输入x,y的数据，并计算每个样本分到各类的概率
        self.x = x
        self.num = x.shape[0]
        self.dim = x.shape[1]
        # y把它从1-10的数字转化为one-hot encoder矩阵
        y_matrix = np.zeros([self.num, 10])
        y_matrix[range(y.size), np.transpose(y)] = 1
        self.y = y_matrix  # batchSize * 10
        # softmax得分，且通过冗余性将最大值置零
        score = np.dot(x, self.w)
        z = np.exp(score)  # batchSize * 10 每个样本在不同类别上未归一化的概率值
        s = np.sum(z, axis=1).reshape([self.num, 1])
        self.p = z / s

    def func(self):
        res = np.dot(np.log(self.p), np.transpose(self.y))
        func = - np.sum(np.trace(res)) / self.num
        return func

    def grad(self):
        # 直接利用微积分推导公式计算梯度
        x = self.x
        y = self.y
        p = self.p
        grad = np.dot(np.transpose(x), (p - y)) / self.num  # dim*10
        return grad

    def learn(self, x, y):
        self.inputData(x, y)
        self.iterTime += 1
        #
        # 选择迭代方式
        # iter/iter2/iter3
        #
        self.iter2()

    def iter(self):
        # 固定步长梯度法
        learningRate = 3 * (10 ** -7)
        self.w -= learningRate * self.grad()

    def iter2(self):
        # 动量下降法 v也要和w一起迭代
        beta = 0.9
        learningRate = 1 * (10 ** -6)
        self.v = beta * self.v + self.grad()  # (1 - beta)#
        # 本质是将所有训练批次计算出的梯度进行指数加权平均,作为迭代方向
        self.w -= learningRate * self.v

    def precision(self):
        # 多分类问题就只有分类准确率了
        p = self.p
        y = self.y
        cnt = 0
        for i in range(p.shape[0]):
            index = np.argmax(p[i])
            if y[i][index] == 1:
                cnt += 1
        return cnt / y.shape[0]
