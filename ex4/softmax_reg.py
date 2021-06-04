import numpy as np
import grad
import lossImage
from mlxtend.data import loadlocal_mnist


def run():
    # 设定mini-batch大小和在训练集整体上迭代次数
    batchSize = 20000
    epochNum = 50

    # 该函数从MNIST原始数据里提取出所需的训练集和测试集,进行了归一化并附加了截距1
    print("Loading Dataset")
    train, train_labels, test, test_labels = load_mnist()
    batchNum = int(train_labels.size / batchSize)
    # labels是列向量
    train_labels = train_labels.reshape([train_labels.size, 1])
    test_labels = test_labels.reshape([test_labels.size, 1])

    print("Initializing Model")
    # 设立回归模型参数初始值
    theta = np.random.rand(train.shape[1], 10) * 0.001
    # 向梯度下降模型中输入theta的初始值
    loss = grad.LossFunc(theta)
    print("Initializing Image")
    img = lossImage.lossImg()
    print("Start Iterating")
    for it in range(epochNum):
        # 可以在每次迭代中打乱数据，使得每次迭代的批次不同
        train, train_labels = shuffle(train, train_labels)
        # 输入每个批次数据，学习并更新theta
        print("--------------")
        print("Epoch {}".format(it + 1))
        for i in range(batchNum):
            upBound = min((i + 1) * batchSize, train.shape[0])
            loss.learn(train[i * batchSize:upBound], train_labels[i * batchSize:upBound])
        # 每轮迭代更新loss曲线点
        testLoss(test, test_labels, loss, img)
    print("Precision:{}%".format(loss.precision() * 100))
    # 绘制loss曲线
    img.plot()


def testLoss(test, test_labels, loss, img):
    # 检验模型在测试集上的LOSS
    loss.inputData(test, test_labels)
    lossValue = loss.func()
    img.loss_test(lossValue)
    print("loss On TestSet:{}".format(lossValue))


def load_mnist():
    x, y = loadlocal_mnist(
        images_path='ex4/train-images-idx3-ubyte',
        labels_path='ex4/train-labels-idx1-ubyte')
    m, n = loadlocal_mnist(
        images_path='ex4/t10k-images-idx3-ubyte',
        labels_path='ex4/t10k-labels-idx1-ubyte')
    train = np.c_[x, y]
    train, train_labels = train[:, :-1], train[:, -1]
    train = np.hstack((np.ones([train.shape[0], 1]), train))
    test = np.c_[m, n]
    test, test_labels = test[:, :-1], test[:, -1]
    test = np.hstack((np.ones([test.shape[0], 1]), test))

    return train, train_labels, test, test_labels


def shuffle(data, labels):
    # labels只有一列
    l = np.hstack([data, labels]).astype(np.int)
    np.random.shuffle(l)
    data = l[:, :-1]
    labels = l[:, -1].reshape([labels.size, 1])
    return data, labels
