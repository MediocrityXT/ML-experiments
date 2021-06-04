import numpy as np
import matplotlib.pyplot as plt


def run():
    data = np.loadtxt('ex2/housing.data')
    # data[0]为1，表示偏差，剩下13个样本维度，以及最后一个为y值
    data = np.hstack((np.ones([data.shape[0], 1]), data))
    [sampleNum, thetaNum] = data.shape
    thetaNum -= 1  # 减去最后y

    # 设立线性模型参数向量
    # theta对应1个偏差，13个样本维度参数

    theta = np.zeros([thetaNum])
    # theta = np.random.rand(thetaNum)

    # 划分训练集和测试集
    batchSize = 90
    batchNum = 5
    trainingSetSize = batchSize * batchNum
    # 可以打乱数据顺序
    np.random.shuffle(data)
    train = data[:trainingSetSize]
    test = data[trainingSetSize:]

    theta = np.mat(theta).T
    epochNum = 10000
    learningRate = 0.000001
    for it in range(epochNum):
        np.random.shuffle(train)
        # 计算每个批次累计代价函数，计算累计梯度，更新theta
        for i in range(batchNum):
            theta = Matrixlearn(train[i * batchSize:(i + 1) * batchSize], theta, learningRate)

    testErrorMat(test, theta)

    # 绘制最终预测点与目标点
    printPrediction(data, theta)


def printPrediction(data, theta):
    data = np.mat(data)
    y = data[:, -1]
    x = data[:, 0:-1]
    fx = x * theta
    y = np.array(y).reshape([y.size])
    fx = np.array(fx).reshape([fx.size])
    y_indices = np.argsort(y)
    plt.plot(y[y_indices], 'xg', label='Real Price')
    plt.plot(fx[y_indices], 'xr', label='Predicted Price')
    plt.xlabel('House #')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('ex2/PredictionResult.png')
    plt.show()


def Matrixlearn(train, theta, learningRate):
    [batchSize, thetaNum] = train.shape
    thetaNum -= 1
    train = np.mat(train)
    cost = 0
    grad = np.zeros([thetaNum])
    xMat = train[:, 0:thetaNum]
    yMat = train[:, -1]

    fxMat = xMat * theta
    error = fxMat - yMat
    cost = 0.5 * error.T * error / error.size
    print('J cost:{}'.format(cost))
    grad = xMat.T * error / error.size

    theta = theta - learningRate * grad
    return theta


def testErrorMat(test, theta):
    # 检验模型在测试集上的LMS
    [testSize, thetaNum] = test.shape
    thetaNum -= 1
    test = np.mat(test)
    cost = 0
    grad = np.zeros([thetaNum])
    xMat = test[:, 0:thetaNum]
    yMat = test[:, -1]

    error = xMat * theta - yMat
    LMS = error.T * error / error.size
    print("LMS Error of TestSet:{}".format(LMS))
    return LMS
