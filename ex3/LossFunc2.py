import numpy as np

#直接使用公式带入计算一阶导二阶导
class LossFunc:
    def __init__(self, x, y, w):
        # 需要直接对自变量——参数向量w求函数，求一阶导，二阶导，保存在一个实例里
        self.x = x  # x为常量，全体数据已经直接带入
        self.num = x.shape[0]
        self.dim = x.shape[1]
        self.y = y
        self.w = w

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
        x = self.x  # 1000x785
        y = self.y  # 1000x1
        z = np.dot(x, w)  #1000x1
        #sig = self.sigmoid(z)
        func = np.sum(np.log(1+np.exp(z))) - np.sum(np.multiply(y, z))
        return func

    def func1d(self, w, epi=10 ** -4):
        # （差分方法/直接利用微积分推导得到计算公式）计算一阶偏导，最终得到jacobian矩阵
        x = self.x
        y = self.y
        z = np.dot(x, w)
        # 需要按照每个样本来单独计算吗
        p1 = np.true_divide(np.exp(z),1+np.exp(z))  # 其实就是sigmoid(-z)
        func1d = -np.dot(np.transpose(x), y - z)#785*1 jacobian矩阵
        return func1d

    def func2d(self, w, epi=10 ** -4):
        # 计算二阶导,黑塞矩阵
        x = self.x
        y = self.y
        z = np.dot(x, w)
        p1 = np.true_divide(np.exp(z), 1 + np.exp(z))  # 其实就是sigmoid(-z)  1000*1
        t1 = np.dot(np.transpose(x), (1-p1)).reshape([self.dim, 1])
        t2 = np.dot(np.transpose(p1) , x).reshape([1, self.dim])
        func2d = t1 * t2 # 785*n  n*1 1*n n*785
        return func2d


    def iter(self):
        # 牛顿法计算下一个w,黑塞矩阵求逆容易出现问题
        # 计算二阶导和一阶导 进行了太多重复计算 可以考虑存储一个函数值字典、导数字典、
        next_w = self.w - np.linalg.inv(self.func2d(self.w)) * self.func1d(self.w)
        return next_w
