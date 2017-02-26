from .supervised_learner import SupervisedLearner
from .matrix import Matrix
from random import uniform
from random import shuffle
import math


class Node(object):
    node_uid_count = 0

    def __init__(self, out=0):
        self.uid = Node.node_uid_count
        Node.node_uid_count += 1

        self.net = 0
        self.out = out
        self.sigma = None
        self.bias_weight = 1


class NeuralNetLearner(SupervisedLearner):
    def __init__(self):
        self.LR = 0.1
        # number of nodes in the hidden layer
        self.hid_count = 2
        # train/validate split
        self.train_percent = 0.75

        # init in layer
        self.in_lay = []

        # init hidden layer
        self.hid_lays = []

        # init out layer
        self.out_lay = []

        # init weightMap
        self.wm = dict()

    def fillInputLayer(self, instance):
        for i in range(len(instance)):
            self.in_lay[i].out = instance[i]

    def genInputLayer(self, size):
        layer = []
        for i in size:
            layer.append(Node())
        return layer

    def genHiddenLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node())
        return layer

    def genOutputLayer(self, size):
        layer = []
        for i in range(size):
            layer.append(Node())
        return layer

    # Calculates the new sigma values for output layer j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    def updateWeights_out(self, i, j, target):
        print('********************************************************** TARGET: ' + str(target))
        for j_node in j:
            # calculate the output node's new sigma
            print('calculating output sigmas ...')
            j_node.sigma = (target - j_node.out) * j_node.out * (1 - j_node.out)
            print('\t' + str(j_node.uid) + ' - newSig: ' + str(j_node.sigma))

            # use that sigma to update the weights feeding into this output node
            print('updating output weights')
            for i_node in i:
                w_uid = self.gen_w_uid(i_node, j_node)
                delta_w = self.LR * j_node.sigma * i_node.out
                print('\t' + w_uid + ' : ' + str(delta_w))
                self.wm[w_uid] += delta_w

            # update the output node's bias weight
            delta_w = self.LR * j_node.sigma * 1
            print(str(j_node.uid) + ' delta_bias : ' + str(delta_w), end='   ')
            j_node.bias_weight += delta_w
            print('new bias: ' + str(j_node.bias_weight))

    # Calculates the new sigma values for j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    # param k: list of nodes in the following node layer
    def updateWeights(self, i, j, k):
        for j_node in j:
            sigSum = 0
            # use the nodes in the layer ahead of this node to calculate the sigma
            for k_node in k:
                w_uid = self.gen_w_uid(j_node, k_node)
                sigSum += self.wm[w_uid] * k_node.sigma
            net_prime = j_node.out * (1 - j_node.out)
            j_node.sigma = net_prime * sigSum

            # use that sigma to update the weights feeding into this hidden layer node
            for i_node in i:
                w_delta = self.LR * j_node.sigma * i_node.out
                w_uid = self.gen_w_uid(i_node, j_node)
                self.wm[w_uid] += w_delta

            # update the hidden nodes bias weight
            delta_w = self.LR * j_node.sigma * 1
            print(str(j_node.uid) + ' delta_bias : ' + str(delta_w), end='   ')
            j_node.bias_weight += delta_w
            print('new bias: ' + str(j_node.bias_weight))

    def gen_w_uid(self, node1, node2):
        return ''.join([str(node1.uid), '-', str(node2.uid)])

    def train(self, features, labels):
        # Create Nodes
        # fill the input layer with the first entry in the instances, this will be overwritten
        self.in_lay = self.genInputLayer(features.row(0))
        self.hid_lays.append(self.genHiddenLayer(self.hid_count))
        self.out_lay = self.genOutputLayer(labels.value_count(0))

        # Build weight map - the nodes in place are dummies that will develop as the program runs
        # the weights established here will persist through the program's training
        for in_node in self.in_lay:
            for h_node in self.hid_lays[0]:
                w_uid = self.gen_w_uid(in_node, h_node)
                self.wm[w_uid] = uniform(-0.1, 0.1)

        for h_node in self.hid_lays[0]:
            for o_node in self.out_lay:
                w_uid = self.gen_w_uid(h_node, o_node)
                self.wm[w_uid] = uniform(-0.1, 0.1)

        epoch_count = 0

        while epoch_count < 1:
            epoch_count += 1

            # randomly split the features into train and validation sets
            features.shuffle(labels)
            train_set = []
            train_set_targets = []
            valid_set = []
            valid_set_targets = []

            for feat_index in range(features.rows):
                if feat_index < math.floor(features.rows * self.train_percent):
                    train_set.append(features.row(feat_index))
                    train_set_targets.append(labels.row(feat_index))
                else:
                    valid_set.append(features.row(feat_index))
                    valid_set_targets.append(labels.row(feat_index))

            print("train set... ")
            for el in train_set:
                print('\t' + str(el))
            print("valid set...")
            for el in valid_set:
                print('\t' + str(el))

            # adjust weights using the train set
            for instance_index in range(len(train_set)):
                self.propagate(train_set[instance_index], train_set_targets[instance_index])
            # check validity using validation set
            # repeat while accuracy is increasing

    def propagate(self, instance, target):
        # load the instance into the input nodes
        print('loading instance into input nodes...')
        self.fillInputLayer(instance)
        print('\tcurrent instance: ',end='')
        for i_node in self.in_lay:
            print(i_node.out, end=' ')
        print()

        print("FORWARD PROPAGATING ...")
        cur_hl = 0

        # calculate the error(net) of the hidden layer nodes
        print("calculating hidden layer " + str(cur_hl) + " values ...")

        # Calculate the net values of the hidden layers
        for h_node in self.hid_lays[cur_hl]:
            # for each node in the hidden layer
            for in_node in self.in_lay:
                # look up the weight between it and each input node
                h_node.net += self.wm[self.gen_w_uid(in_node, h_node)] * in_node.out
            h_node.net += 1 * h_node.bias_weight

        print("\thidden layer NET values ... ")
        for node in self.hid_lays[cur_hl]:
            print('\t\t' + str(node.net))

        # Calculate the out values of the hidden layers
        for node in self.hid_lays[cur_hl]:
            node.out = 1 / (1 + math.exp(-node.net))

        print("\thidden layer OUT values ...")
        for node in self.hid_lays[cur_hl]:
            print('\t\t' + str(node.out))

        print("calculating output values ...")

        # Calculate the net values of the output nodes
        for o_node in self.out_lay:
            for h_node in self.hid_lays[-1]:
                o_node.net += self.wm[self.gen_w_uid(h_node, o_node)] * h_node.out
            o_node.net += 1 * o_node.bias_weight

        print("\toutput layer NET values ...")
        for node in self.out_lay:
            print('\t\t' + str(node.uid) + ' : ' + str(node.net))

        # Calculate the out  values of the output nodes
        for node in self.out_lay:
            node.out = 1 / (1 + math.exp(-node.net))

        print("\toutput layer OUT values ...")
        for node in self.out_lay:
            print('\t\t' + str(node.uid) + ' : ' + str(node.out))

        print("BACK PROPAGATING ...")

        # Calculate sigma and update weights for the output layer
        self.updateWeights_out(self.hid_lays[-1], self.out_lay, target[0])

        # Calculate sigma and update weights for the hidden layer
        self.updateWeights(self.in_lay, self.hid_lays[0], self.out_lay)

        print("\tending weights: ")
        for weightKey in self.wm:
            print("\t\t" + str(weightKey) + " : " + str(self.wm[weightKey]))

    def predict(self, features, labels):
        pass
