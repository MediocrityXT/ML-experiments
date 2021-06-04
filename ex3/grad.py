import numpy as np


class LossFunc:
    def __init__(self, w_init):
        self.w = w_init
        # 用v保存动量
        self.v = 0
        self.m = 0
        self.v_hat = 0
        self.iterTime = 0

    def func(self, w):
        # 计算函数值
        x = self.x  # 1000x785
        y = self.y  # 1000x1
        z = np.dot(x, w)  # 1000x1
        # sig = self.sigmoid(z)
        # 直接根据推导得到下面的代价函数公式
        func = np.sum(np.log(1 + np.exp(z))) - np.sum(np.multiply(y, z))
        return func

    def grad(self, w):
        # 直接利用微积分推导公式计算梯度
        x = self.x
        y = self.y
        z = np.dot(x, w)
        # p1 = np.true_divide(np.exp(z), 1 + np.exp(z))  # 其实就是sigmoid(-z)
        func1d = -np.dot(np.transpose(x), y - z)  # 785*1 jacobian矩阵
        return func1d

    def inputData(self, x, y):
        # 输入x,y的数据
        self.x = x
        self.y = y.reshape([y.size])
        self.num = x.shape[0]
        self.dim = x.shape[1]

    def learn(self, x, y):
        self.inputData(x, y)
        # 选择迭代方式
        self.iterTime += 1
        new_theta = self.iter3(self.w)
        self.w = new_theta

    def iter(self, w):
        # 固定步长梯度法
        learningRate = 1 * (10 ** -6)
        next_w = w - learningRate * self.grad(w)
        return next_w

    def iter2(self, w):
        # 动量下降法 v也要和w一起迭代
        beta = 0.9
        learningRate = 1 * (10 ** -6)
        self.v = beta * self.v + self.grad(w) # (1 - beta)#
        # 本质是将所有训练批次计算出的梯度进行指数加权平均,作为迭代方向
        next_w = w - learningRate * self.v
        return next_w

    def iter3(self, w):
        # Adam算法
        beta1, beta2 = 0.9, 0.999
        epi = 10 ** -8
        learningRate = 10 * (10 ** -4)
        grad = self.grad(w)
        self.m = beta1 * self.m + (1 - beta1) * grad
        squaredGrad = np.dot(grad, grad)
        self.v = beta2 * self.v + (1 - beta2) * squaredGrad
        m_hat = self.m / (1 - beta1 ** self.iterTime)
        v_hat = self.v / (1 - beta2 ** self.iterTime)
        next_w = w - learningRate * m_hat / (np.sqrt(v_hat) + epi)
        return next_w

    def iter4(self, w):
        # AMSGrad算法
        beta1, beta2 = 0.9, 0.999
        epi = 10 ** -8
        learningRate = 90 * (10 ** -4)
        grad = self.grad(w)
        self.m = beta1 * self.m + (1 - beta1) * grad
        squaredGrad = np.dot(grad, grad)
        self.v = beta2 * self.v + (1 - beta2) * squaredGrad
        self.v_hat = max(self.v_hat, self.v)
        next_w = w - learningRate * self.m / (np.sqrt(self.v_hat) + epi)
        return next_w

    def prec_recall(self, w):
        # 以1为指标，precision是检测为1中确实为1的比例,recall是真实1检测出是1的比例
        x = self.x
        y = self.y
        z = np.dot(x, w)  # z的正负对应1，0

        TP, FN, FP, TN = 0, 0, 0, 0
        for i in range(y.size):
            if y[i] == 1:
                if z[i] >= 0:  # sigmoid(z)>0.5
                    TP += 1
                else:
                    FN += 1
            else:
                if z[i] >= 0:
                    FP += 1
                else:
                    TN += 1
        # python3除法自动保留小数
        try:
            precision = TP / (TP + FP)
        except ZeroDivisionError:
            precision = 0
        try:
            recall = TP / (TP + FN)
        except ZeroDivisionError:
            recall = 0
        return precision, recall
