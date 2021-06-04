import matplotlib.pyplot as plt


class lossImg:
    def __init__(self):
        self.loss_trainSet = []
        self.loss_testSet = []

    def loss_test(self, loss):
        self.loss_testSet.append(loss)

    def plot(self):
        plt.plot(self.loss_testSet, 'r')
        plt.title('Loss on Test Set')
        plt.savefig('ex4/Loss.png')
        plt.show()
