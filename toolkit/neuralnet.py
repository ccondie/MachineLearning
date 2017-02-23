from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from random import uniform
import math
from .node import Node


class NeuralNetLearner(SupervisedLearner):
    def __init__(self):
        self.LR = 0.1

        # init hidden layer
        self.hid_lays = []

        # init out layer
        self.out_lay = self.genTestOutputLayer()

        # fill the hidden layer
        self.hid_lays.append(self.genTestHiddenLayer())

    # creates a new list of "weights" representing a newly initialized hidden layer
    def genHiddenLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node(uniform(-0.1, 0.1)))
        return layer

    def genTestHiddenLayer(self):
        layer = [Node(1), Node(1)]
        return layer

    def genOutputLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node(uniform(-0.1, 0.1)))
        return layer

    def genTestOutputLayer(self):
        layer = [Node(1)]
        return layer

    def train(self, features, labels):
        # get the input set
        inVals = features.row(0)
        target = labels.row(0)

        print("FORWARD PROPAGATING ...")
        print("\tinput values: " + str(inVals))

        # calculate the error(net) of the hidden layer nodes
        print("Processing HIDDEN Layers")
        cur_hl = 0

        # Calculate the net values of the hidden layers
        for node in self.hid_lays[cur_hl]:
            for inVal in inVals:
                node.net += node.weight * inVal
            # handle the input bias
            node.net += node.weight * 1

        print("\thidden layer NET values: ", end='')
        for node in self.hid_lays[cur_hl]:
            print(node.net, end=' ')
        print()

        # Calculate the out values of the hidden layers
        for node in self.hid_lays[cur_hl]:
            node.out = 1 / (1 + math.exp(-node.net))

        print("\thidden layer OUT values: ", end='')
        for node in self.hid_lays[cur_hl]:
            print(node.out, end=' ')
        print()

        print("Processing Output Layers")

        # Calculate the net values of the output nodes
        for node in self.out_lay:
            for hid_node in self.hid_lays[-1]:
                node.net += node.weight * hid_node.out
            # handle bias
            node.net += node.weight * 1

        print("\toutput layer NET values: ", end=' ')
        for node in self.out_lay:
            print(node.net, end=' ')
        print()

        # Calculate the out  values of the output nodes
        for node in self.out_lay:
            node.out = 1 / (1 + math.exp(-node.net))

        print("\toutput layer OUT values: ", end='')
        for node in self.out_lay:
            print(node.out, end=' ')
        print()

        print("BACK PROPAGATING ...")

        # Calculate the sigma of each output node
        for node in self.out_lay:
            # sigma = (TARGETj - OUTPUTj)OUTPUTj(1-OUTPUTj)
            node.sigma = (target[0] - node.out) * node.out * (1 - node.out)

        print("\toutput layer SIGMA values: ", end=' ')
        for node in self.out_lay:
            print(node.sigma, end=' ')
        print()

        #
        #
        #
        #
        #

        print()
        pass

    def predict(self, features, labels):
        pass
