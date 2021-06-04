import numpy as np
import pandas
import grad
import lossImage


def run():
    # extractData()
    # 该函数从MNIST原始数据里提取出所需的01图片数据,data增加一列常数偏差，保存为txt
    # 提取data之后进行归一化
    data = np.loadtxt("ex3/data.txt", dtype=int) / 255
    labels = np.loadtxt("ex3/labels.txt", dtype=int)
    # labels是列向量
    labels = labels.reshape([labels.size, 1])
    # 设立回归模型参数初始值（如果使用割线法优化需要两个初值）
    theta = np.ones([data.shape[1]]) * 1

    # (随机)划分训练集和测试集
    # data, labels = shuffle(data, labels)
    # 设定训练集占比
    trainRate = 0.7
    batchSize = 64
    batchNum = int((trainRate * labels.size) / batchSize)
    #batchNum = 1
    trainingSetSize = batchSize * batchNum
    train = data[:trainingSetSize, :]
    train_labels = labels[:trainingSetSize, :]
    test = data[trainingSetSize:, :]
    test_labels = labels[trainingSetSize:, :]

    img = lossImage.lossImg()
    # 向梯度下降模型中输入theta的初始值
    loss = grad.LossFunc(theta)
    epochNum = 1000
    for it in range(epochNum):
        # 可以在每次迭代中打乱数据，使得每次迭代的批次不同
        train, train_labels = shuffle(train, train_labels)
        # 输入每个批次数据，学习并更新theta
        for i in range(batchNum):
            loss.learn(train[i * batchSize:(i + 1) * batchSize], train_labels[i * batchSize:(i + 1) * batchSize])
        theta = loss.w
        # 每次迭代更新loss曲线点
        trainLoss(train, train_labels, theta, img)
        testLoss(test, test_labels, theta, img)

    # 绘制loss曲线
    img.plot()

def trainLoss(train, train_labels, theta, img):
    # 检验模型在训练集上的LOSS
    loss = grad.LossFunc(theta)
    loss.inputData(train, train_labels)
    lossValue = loss.func(theta)
    img.loss_train(lossValue)
    #print("loss On TrainSet:{}".format(lossValue))
    # prec, recall = loss.prec_recall(theta)
    # print("precision:{}\nrecall:{}".format(prec, recall))

def testLoss(test, test_labels, theta, img):
    # 检验模型在测试集上的LOSS
    loss = grad.LossFunc(theta)
    loss.inputData(test, test_labels)
    lossValue = loss.func(theta)
    img.loss_test(lossValue)
    print("loss On TestSet:{}".format(lossValue))
    precision, recall = loss.prec_recall(theta)
    print("precision:{}".format(precision))
    # print("recall:{}".format(recall))


def extractData():
    mnist = pandas.read_csv('ex3/mnist_784.csv')
    labels = mnist.loc[:, ['class']].values
    data = mnist.iloc[:, 0:-1].values
    l = []
    for i in range(data.shape[0]):
        if labels[i] > 1:
            l.append(i)
    data = np.delete(data, l, axis=0)
    labels = np.delete(labels, l, axis=0)
    # x[0]为1，表示偏差，剩下28*28列是像素值，每一行表示不同样本图片
    # y值提取出来,因为数据集只包括0,1，所以是二分类问题
    data = np.hstack((np.ones([data.shape[0], 1]) * 255, data))
    np.savetxt("ex3/data.txt", data, fmt="%d")
    np.savetxt("ex3/labels.txt", labels, fmt="%d")


def shuffle(data, labels):
    # labels只有一列
    l = np.hstack((data, labels))
    np.random.shuffle(l)
    data = l[:, :-1]
    labels = l[:, -1]
    labels = labels.reshape([labels.size, 1])
    return data, labels
