from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from random import uniform
import math


class Node(object):
    def __init__(self, out=0):
        self.uid = NeuralNetLearner.node_uid_count
        NeuralNetLearner.node_uid_count += 1

        self.net = 0
        self.out = out
        self.sigma = None
        self.bias_weight = 1


class NeuralNetLearner(SupervisedLearner):
    node_uid_count = 0

    def __init__(self):
        self.LR = 0.1
        self.hid_count = 2

        # init in layer
        self.in_lay = []

        # init hidden layer
        self.hid_lays = []

        # init out layer
        self.out_lay = []

        # init weightMap
        self.wm = dict()

    def fillInputLayer(self, data):
        layer = []
        for val in data:
            layer.append(Node(val))
        return layer

    def genHiddenLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node())
            self.node_uid_count += 1
        return layer

    def genTestHiddenLayer(self):
        layer = [Node(), Node()]
        return layer

    def genOutputLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node())
        return layer

    def genTestOutputLayer(self):
        layer = [Node()]
        return layer

    def buildWeightMap(self):
        pass

    def train(self, features, labels):
        target = labels.row(0)

        # Create Nodes
        self.in_lay = self.fillInputLayer(features.row(0))
        self.hid_lays.append(self.genTestHiddenLayer())
        self.out_lay = self.genTestOutputLayer()

        # Build weight map
        for in_node in self.in_lay:
            for h_node in self.hid_lays[0]:
                w_uid = ''.join([str(in_node.uid), "-", str(h_node.uid)])
                self.wm[w_uid] = uniform(-0.1, 0.1)
                self.wm[w_uid] = 1

        for h_node in self.hid_lays[0]:
            for o_node in self.out_lay:
                w_uid = ''.join([str(h_node.uid), "-", str(o_node.uid)])
                self.wm[w_uid] = uniform(-0.1, 0.1)
                self.wm[w_uid] = 1

        print("\tinput values: ", end=' ')
        for node in self.in_lay:
            print(node.out, end=' ')
        print()

        print("\tinitial weights: ")
        for weightKey in self.wm:
            print("\t\t" + str(weightKey) + " : " + str(self.wm[weightKey]))

        print("FORWARD PROPAGATING ...")

        # calculate the error(net) of the hidden layer nodes
        print("Processing HIDDEN Layers")
        cur_hl = 0

        # Calculate the net values of the hidden layers
        for node in self.hid_lays[cur_hl]:
            # for each node in the hidden layer
            for in_node in self.in_lay:
                # look up the weight between it and each input node
                node.net += self.wm[''.join([str(in_node.uid), '-', str(node.uid)])] * in_node.out
            node.net += 1 * node.bias_weight

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
        for out_node in self.out_lay:
            for hid_node in self.hid_lays[-1]:


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
