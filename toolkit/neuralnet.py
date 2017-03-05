from supervised_learner import SupervisedLearner
from matrix import Matrix
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
        self.sigma = 0
        self.bias_weight = 1
        # used to specify target for an output node using categorical data
        self.target = 0

    def print(self):
        print('\t' + str(self.uid) + '\tnet: ' + '{:.10f}'.format(self.net) + '\tout: ' + '{:.10f}'.format(
            self.out) + '\tsigma: ' + '{:.10f}'.format(self.sigma) + '\tbias_weight: ' + '{:.10f}'.format(
            self.bias_weight) + '\ttarget: ' + '{:.10f}'.format(self.target))


class NeuralNetLearner(SupervisedLearner):
    def __init__(self):
        self.debug = False
        self.LR = 0.1
        # number of nodes in the hidden layer
        self.hid_count = 8
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

    def genOutputLayer(self, labels):
        layer = []
        for i in range(labels.value_count(0)):
            dumNode = Node()
            dumNode.target = i
            layer.append(dumNode)

        return layer

    # Calculates the new sigma values for output layer j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    def updateWeights_out(self, i, j, target):
        for j_node in j:
            node_target = None
            if target == j_node.target:
                node_target = 1
            else:
                node_target = 0

            # calculate the output node's new sigma
            j_node.sigma = (node_target - j_node.out) * j_node.out * (1 - j_node.out)

            # use that sigma to update the weights feeding into this output node
            for i_node in i:
                w_uid = self.gen_w_uid(i_node, j_node)
                delta_w = self.LR * j_node.sigma * i_node.out
                self.wm[w_uid] += delta_w

            # update the output node's bias weight
            delta_w = self.LR * j_node.sigma * 1
            j_node.bias_weight += delta_w

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
            j_node.bias_weight += delta_w

    def gen_w_uid(self, node1, node2):
        return ''.join([str(node1.uid), '-', str(node2.uid)])

    def print_status(self):
        if self.debug:
            print('INPUT LAYER Nodes: ')
            for node in self.in_lay:
                node.print()

            print('HIDDEN LAYER Nodes: ')
            for layer in self.hid_lays:
                for node in layer:
                    node.print()

            print('OUTPUT LAYER Nodes: ')
            for node in self.out_lay:
                node.print()

            print('WEIGHTS: ')
            for key in self.wm:
                print(str(key) + ': ' + str(self.wm[key]))
            print('\n\n\n')

    def train(self, features, labels):
        # Create Nodes
        # fill the input layer with the first entry in the instances, this will be overwritten
        self.in_lay = self.genInputLayer(features.row(0))
        self.hid_lays.append(self.genHiddenLayer(self.hid_count))
        self.out_lay = self.genOutputLayer(labels)

        if self.debug:
            print('START TRAIN')
            print('------------------------------------------------------------------------------------------')

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

        self.print_status()

        epoch_count = 0
        learning = True
        delta_accur = 1.0
        prev_accur = 0

        # while learning:
        while epoch_count < 1000:
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

            if self.debug:
                print("train set... ")
                for i in range(len(train_set)):
                    print('\t' + str(train_set[i]) + ' ---> ' + str(train_set_targets[i]))

                print("valid set...")
                for i in range(len(valid_set)):
                    print('\t' + str(valid_set[i]) + ' ---> ' + str(valid_set_targets[i]))
                print('\n\n')

                # adjust weights using the train set
            for instance_index in range(len(train_set)):
                self.propagate(train_set[instance_index], train_set_targets[instance_index])

            # check validity using validation set
            correct = 0
            incorrect = 0

            if self.debug:
                print('STARTING VALIDATION CHECK')

            for instance_index in range(len(valid_set)):
                instance_prediction = []
                self.predict(valid_set[instance_index],instance_prediction)

                if self.debug:
                    print('expect: ' + str(valid_set_targets[instance_index]) + '\t', end='')
                    print('actual: ' + str(instance_prediction[0]))

                if instance_prediction == valid_set_targets[instance_index]:
                    correct += 1
                else:
                    incorrect += 1

            accuracy = correct / (correct + incorrect)
            delta_accur = abs(prev_accur - accuracy)
            prev_accur = accuracy

            if self.debug:
                print('CORRECT: ' + str(correct))
                print('INCORRECT: ' + str(incorrect))
                print('ACCURACY: ' + str(accuracy))
                print('CHANGE IN ACCURACY: ' + str(delta_accur))

            if accuracy > .80:
                learning = False

        self.debug = True
        self.print_status()

    def calc_output(self, in_layer, layer):
        # calculate net values
        for node in layer:
            node.net = 0
            for in_node in in_layer:
                node.net += self.wm[self.gen_w_uid(in_node, node)] * in_node.out
            node.net += 1 * node.bias_weight

        for node in layer:
            node.out = 1 / (1 + math.exp(-node.net))

    def propagate(self, instance, target):
        # load the instance into the input nodes
        self.fillInputLayer(instance)

        cur_hl = 0

        # Calculate the net values of the hidden layers
        self.calc_output(self.in_lay, self.hid_lays[cur_hl])
        # Calculate the net values of the output nodes
        self.calc_output(self.hid_lays[-1], self.out_lay)

        self.print_status()

        # Calculate sigma and update weights for the output layer
        self.updateWeights_out(self.hid_lays[-1], self.out_lay, target[0])

        # Calculate sigma and update weights for the hidden layer
        self.updateWeights(self.in_lay, self.hid_lays[0], self.out_lay)

        self.print_status()

    def predict(self, features, labels):
        self.fillInputLayer(features)

        self.calc_output(self.in_lay, self.hid_lays[0])
        self.calc_output(self.hid_lays[-1], self.out_lay)

        prediction = -1
        highest = 0

        for node in self.out_lay:
            if node.out > highest:
                highest = node.out
                prediction = node.target

        if len(labels) == 0:
            labels.append(prediction)
        else:
            labels[0] = prediction
