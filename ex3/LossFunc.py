import numpy as np


class LossFunc:
    def __init__(self, x, y, w):
        # 需要直接对自变量——参数向量w求函数，求一阶导，二阶导，保存在一个实例里
        self.x = x  # x为常量，全体数据已经直接带入
        self.num = x.shape[0]
        self.dim = x.shape[1]
        self.y = y
        self.w = w
        self.last_w = w / 2


    def sigmoid(self, z):
        sig = np.zeros_like(z)
        for i in range(z.shape[0]):
            sig[i] = 1.0 / (1 + np.exp(-z[i]))
            # if z[i] <= 0:
            #     sig[i] = 1.0 / (1 + np.exp(-z[i]))
            # else:
            #     sig[i] = np.exp(-z[i]) / (1 + np.exp(-z[i]))
        return sig

    def func(self, w):
        # 计算函数值
        x = self.x
        y = self.y
        z = np.dot(x, w)
        sig = self.sigmoid(z)

        # func = 0
        # for i in range(self.num):
        #     func += np.log(1+np.exp(z[i])) - y[i] * z[i]
        # print(func)
        func = np.sum(np.log(1+np.exp(z))) - np.sum(np.multiply(y, z))

        #func = -np.sum(np.dot(y, np.log(sig)) + np.dot(1 - y, np.log(1 - sig)))
        return func

    def func1d(self, w, epi=10 ** -4):
        # （差分方法/直接利用微积分推导得到计算公式）计算一阶偏导，最终得到jacobian矩阵
        func1d = np.zeros_like(w)
        for i in range(self.dim):
            delta = np.zeros_like(w)
            delta[i] = epi
            forth_w = w + delta
            back_w = w - delta
            func1d[i] = (self.func(forth_w) - self.func(back_w)) / (2 * epi)
        return func1d

    def func2d(self, w, epi=10 ** -4):
        # 差分方法计算二阶导,黑塞矩阵
        func2d = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            delta = np.zeros_like(w)
            delta[i] = epi
            forth_w = w + delta
            back_w = w - delta
            temp = (self.func1d(forth_w) - self.func1d(back_w)) / (2 * epi)
            func2d[i] = temp
        return func2d

    def iter(self):
        # 牛顿法计算下一个w,黑塞矩阵求逆容易出现问题
        # 计算二阶导和一阶导 进行了太多重复计算 可以考虑存储一个函数值字典、导数字典、
        next_w = self.w - np.linalg.inv(self.func2d(self.w)) * self.func1d(self.w)
        return next_w

    def iter2(self):
        # 割线法比较合理
        next_w = self.last_w - self.w
        self.last_w = self.w
        return next_w
