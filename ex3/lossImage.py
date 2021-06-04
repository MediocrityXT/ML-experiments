import matplotlib.pyplot as plt


class lossImg:
    def __init__(self):
        self.loss_trainSet = []
        self.loss_testSet = []

    def loss_train(self, loss):
        self.loss_trainSet.append(loss)

    def loss_test(self, loss):
        self.loss_testSet.append(loss)

    def plot(self):
        plt.plot(self.loss_trainSet, 'g', label='loss on trainSet')
        plt.plot(self.loss_testSet, 'r', label='loss on testSet')
        plt.legend()
        plt.show()
        plt.savefig('ex3/Loss.png')
