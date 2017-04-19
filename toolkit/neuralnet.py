import math
from datetime import datetime
from random import uniform
from random import shuffle
import sys
import copy

from supervised_learner import SupervisedLearner


class Node(object):
    node_uid_count = 0

    def __init__(self, out=0):
        self.uid = Node.node_uid_count
        Node.node_uid_count += 1
        self.net = 0
        self.out = out
        self.sigma = 0
        self.bias_weight = 1
        self.bias_last_delta = 0
        # used to specify target for an output node using categorical data
        self.target = 0


def gen_hidden_layer(size):
    layer = []
    for i in range(size):
        layer.append(Node())
    return layer


def gen_output_layer(labels):
    layer = []
    for i in range(labels.value_count(0)):
        dum_node = Node()
        dum_node.target = i
        layer.append(dum_node)

    return layer


def gen_input_layer(size):
    layer = []
    for _ in size:
        layer.append(Node())
    return layer


def update_sigma_out(j, target):
    for j_node in j:
        if target == j_node.target:
            node_target = 1
        else:
            node_target = 0

        # calculate the output node's new sigma
        j_node.sigma = (node_target - j_node.out) * j_node.out * (1 - j_node.out)


def gen_w_uid(node1, node2):
    return ''.join([str(node1.uid), '-', str(node2.uid)])


class NeuralNetLearner(SupervisedLearner):
    def __init__(self):
        self.debug = False

        self.LR = 0.3

        # number of nodes in the hidden layer
        self.hid_count = 20
        # train/validate split
        self.train_percent = 0.75

        self.momentum = False
        self.momentumCo = 0.9

        # init in layer
        self.in_lay = []

        # init hidden layer
        self.hid_lays = []

        # init out layer
        self.out_lay = []

        # init weightMap
        self.wm = dict()
        self.last_delta = dict()

    def fill_input_layer(self, instance):
        for i in range(len(instance)):
            self.in_lay[i].out = instance[i]

    # calculate the error on the output layer given an expected target
    # Calculates the new sigma values for output layer j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    def update_weights_out(self, i, j, target):
        # update the sigma values in the forward layer
        update_sigma_out(j, target)

        for j_node in j:
            # use that sigma to update the weights feeding into this output node
            for i_node in i:
                w_uid = gen_w_uid(i_node, j_node)
                delta_w = self.LR * j_node.sigma * i_node.out
                if self.momentum:
                    self.wm[w_uid] += delta_w + self.momentumCo * self.last_delta[w_uid]
                else:
                    self.wm[w_uid] += delta_w
                self.last_delta[w_uid] = delta_w

            # update the output node's bias weight
            delta_w = self.LR * j_node.sigma * 1
            if self.momentum:
                j_node.bias_weight += delta_w + self.momentumCo * j_node.bias_last_delta
            else:
                j_node.bias_weight += delta_w
            j_node.bias_last_delta = delta_w

    # Calculates the new sigma values for j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    # param k: list of nodes in the following node layer
    def update_weights(self, i, j, k):
        for j_node in j:
            sig_sum = 0
            # use the nodes in the layer ahead of this node to calculate the sigma
            for k_node in k:
                w_uid = gen_w_uid(j_node, k_node)
                sig_sum += self.wm[w_uid] * k_node.sigma
            net_prime = j_node.out * (1 - j_node.out)
            j_node.sigma = net_prime * sig_sum

            # use that sigma to update the weights feeding into this hidden layer node
            for i_node in i:
                delta_w = self.LR * j_node.sigma * i_node.out
                w_uid = gen_w_uid(i_node, j_node)
                if self.momentum:
                    self.wm[w_uid] += delta_w + self.momentumCo * self.last_delta[w_uid]
                else:
                    self.wm[w_uid] += delta_w
                self.last_delta[w_uid] = delta_w

            # update the hidden nodes bias weight
            delta_w = self.LR * j_node.sigma * 1
            if self.momentum:
                j_node.bias_weight += delta_w + self.momentumCo * j_node.bias_last_delta
            else:
                j_node.bias_weight += delta_w
            j_node.bias_last_delta = delta_w

    def train(self, features, labels, lr=None):
        if lr is not None:
            self.LR = lr

        out_file = open('{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()) + str('.csv'), 'a')
        out_file.write('epoch,train_mse,vs_mse,vs_accuracy\n')

        # Create Nodes
        # fill the input layer with the first entry in the instances, this will be overwritten
        self.in_lay = gen_input_layer(features.row(0))
        self.hid_lays.append(gen_hidden_layer(self.hid_count))
        self.out_lay = gen_output_layer(labels)

        # Build weight map - the nodes in place are dummies that will develop as the program runs
        # the weights established here will persist through the program's training
        for in_node in self.in_lay:
            for h_node in self.hid_lays[0]:
                w_uid = gen_w_uid(in_node, h_node)
                self.wm[w_uid] = uniform(-0.1, 0.1)
                self.last_delta[w_uid] = 0

        for h_node in self.hid_lays[0]:
            for o_node in self.out_lay:
                w_uid = gen_w_uid(h_node, o_node)
                self.wm[w_uid] = uniform(-0.1, 0.1)
                self.last_delta[w_uid] = 0

        # ******************************************************************************************
        # Split out VS
        # ******************************************************************************************
        # randomly split the features into train and validation sets
        features.shuffle(labels)
        train_set = []
        train_set_targets = []
        train_order = []

        valid_set = []
        valid_set_targets = []

        # allot the train and valid sets
        for feat_index in range(features.rows):
            if feat_index < math.floor(features.rows * self.train_percent):
                train_set.append(features.row(feat_index))
                train_set_targets.append(labels.row(feat_index))
            else:
                valid_set.append(features.row(feat_index))
                valid_set_targets.append(labels.row(feat_index))

        for i in range(len(train_set)):
            train_order.append(i)

        # ******************************************************************************************
        # Start Training
        # ******************************************************************************************
        learning = True

        epoch_count = 0
        epochs_without_improvement = 0

        best_vs_accuracy = 0
        last_vs_accuracy = 0

        best_vs_mse = sys.maxsize
        last_vs_mse = sys.maxsize

        best_mse = sys.maxsize
        last_mse = sys.maxsize

        best_at_epoch = 0
        best_hl = []
        best_out = []
        best_weights = []

        # while learning:
        while epoch_count < 200:

            print(epoch_count, end=' - ', flush=True)

            # ******************************************************************************************
            # Shuffle the Training Set
            # ******************************************************************************************
            shuffle(train_order)

            # ******************************************************************************************
            # Training Step
            # ******************************************************************************************
            # adjust weights using the train set
            sse = 0
            for instance_index in train_order:
                self.propagate(train_set[instance_index], train_set_targets[instance_index])
                sse += self.net_mse()
            train_mse = sse / len(train_set)
            # train_mse = self.net_mse()

            out_file.write(str(epoch_count) + ',' + str(train_mse) + str(','))

            # ******************************************************************************************
            # Validation Step
            # ******************************************************************************************
            correct = 0
            vs_count = 0
            vs_sse = 0

            # test the validation instances
            for instance_index in range(len(valid_set)):
                vs_count += 1
                instance_prediction = []
                self.vs_predict(valid_set[instance_index], instance_prediction, valid_set_targets[instance_index])
                # vs_sse += self.net_mse()

                if instance_prediction == valid_set_targets[instance_index]:
                    correct += 1

            # vs_mse = vs_sse / len(valid_set)
            vs_mse = self.net_mse()
            vs_accuracy = correct / vs_count

            # vs mse termination
            print(str(train_mse) + ' - ' + str(vs_mse) + ' - ' + str(vs_accuracy))
            if epoch_count > 10:
                if vs_mse < best_vs_mse:
                    print('BEST FOUND')
                    best_vs_mse = vs_mse
                    best_at_epoch = epoch_count
                    epochs_without_improvement = 0
                    # assign the "best" hidden layer set
                    best_hl = copy.deepcopy(self.hid_lays)
                    best_out = copy.deepcopy(self.out_lay)
                    best_weights = copy.deepcopy(self.wm)
                else:
                    epochs_without_improvement += 1

            # # mse termination
            # print(str(train_mse) +' - ' + str(vs_mse) + ' - ' + str(vs_accuracy))
            # if train_mse < last_mse:
            #     if train_mse < best_mse:
            #         best_mse = train_mse
            #         best_at_epoch = epoch_count
            #         epochs_without_improvement = 0
            #         # assign the "best" hidden layer set
            #         best_hl = copy.deepcopy(self.hid_lays)
            #         best_out = copy.deepcopy(self.out_lay)
            # else:
            #     epochs_without_improvement += 1
            #
            # last_mse = train_mse

            # # vs accuracy termination
            # if vs_accuracy > last_vs_accuracy:
            #     if vs_accuracy > best_vs_accuracy:
            #         best_vs_accuracy = vs_accuracy
            #         epochs_without_improvement = 0
            #         # assign the "best" hidden layer set
            #         best_hl = copy.deepcopy(self.hid_lays)
            #         best_out = copy.deepcopy(self.out_lay)
            # else:
            #     epochs_without_improvement += 1
            # last_vs_accuracy = vs_accuracy

            out_file.write(str(vs_mse) + ',' + str(vs_accuracy) + '\n')
            epoch_count += 1

            if epochs_without_improvement > 10:
                learning = False

        # self.hid_lays = best_hl
        # self.out_lay = best_out
        # self.wm = best_weights

        out_file.close()
        print('Best Epoch at: ' + str(best_at_epoch))

    def calc_output(self, in_layer, layer):
        # calculate net values
        for node in layer:
            node.net = 0
            for in_node in in_layer:
                node.net += self.wm[gen_w_uid(in_node, node)] * in_node.out
            node.net += 1 * node.bias_weight

        for node in layer:
            node.out = 1 / (1 + math.exp(-node.net))

    def propagate(self, instance, target):
        # load the instance into the input nodes
        self.fill_input_layer(instance)

        cur_hl = 0

        # Calculate the net values of the hidden layers
        self.calc_output(self.in_lay, self.hid_lays[cur_hl])

        # Calculate the net values of the output nodes
        self.calc_output(self.hid_lays[-1], self.out_lay)

        # Calculate sigma and update weights for the output layer
        self.update_weights_out(self.hid_lays[-1], self.out_lay, target[0])

        # Calculate sigma and update weights for the hidden layer
        self.update_weights(self.in_lay, self.hid_lays[0], self.out_lay)

    def predict(self, features, labels):
        self.fill_input_layer(features)

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

    def vs_predict(self, features, labels, expected):
        self.fill_input_layer(features)

        self.calc_output(self.in_lay, self.hid_lays[0])
        self.calc_output(self.hid_lays[-1], self.out_lay)
        update_sigma_out(self.out_lay, expected)

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

    def net_mse(self):
        # calculates the mse of the neural net as it is now
        error_sum = 0
        node_tally = 0
        for node in self.out_lay:
            error_sum += node.sigma ** 2
            node_tally += 1
        return error_sum / node_tally
