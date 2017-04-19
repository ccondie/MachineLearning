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
        self.error = 0
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


def gen_w_uid(node1, node2):
    return ''.join([str(node1.uid), '-', str(node2.uid)])


class NeuralNetLearner(SupervisedLearner):
    def __init__(self):
        self.debug = False
        self.write_to_file = False

        self.LR = 0.1

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

        self.train_mse = 0
        self.vs_mse = 0
        self.vs_accuracy = 0
        self.best_epoch = 0

    def fill_input_layer(self, instance):
        for i in range(len(instance)):
            self.in_lay[i].out = instance[i]

    def calculate_target_error(self, target):
        for out_node in self.out_lay:
            if target == out_node.target:
                node_target = 1
            else:
                node_target = 0
            out_node.error = node_target - out_node.out

    def calculate_output_sigma(self):
        for out_node in self.out_lay:
            out_node.sigma = out_node.error * out_node.out * (1 - out_node.out)

    # calculate the error on the output layer given an expected target
    # Calculates the new sigma values for output layer j and updates the weights between i and j
    # param i: list of nodes in the preceding node layer
    # param j: list of nodes in the layer to update sigma
    def update_weights_out(self, i, j, target):
        # update the sigma values in the forward layer
        self.calculate_target_error(target)
        self.calculate_output_sigma()

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

    def train(self, features, labels, lr=None, hid_count=None, momCof=None):
        if lr is not None:
            self.LR = lr

        if hid_count is not None:
            self.hid_count = hid_count

        if momCof is not None:
            self.momentumCo = momCof

        # clear previous train data
        self.wm = dict()
        self.hid_lays = []

        if self.write_to_file:
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

        best_vs_mse = sys.maxsize

        best_at_epoch = 0

        best_hl = []
        best_out = []
        best_weights = []

        while learning:
            # print('{:<4d}'.format(epoch_count), end=' - ', flush=True)

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
                sse += self.net_sse()
            train_mse = sse / len(train_set)
            self.train_mse = train_mse

            if self.write_to_file:
                out_file.write(str(epoch_count) + ',' + str(train_mse) + str(','))

            # ******************************************************************************************
            # Validation Step
            # ******************************************************************************************
            correct = 0
            vs_sse = 0

            # test the validation instances
            for instance_index in range(len(valid_set)):
                instance_prediction = []
                self.vs_predict(valid_set[instance_index], instance_prediction, valid_set_targets[instance_index])
                vs_sse += self.net_sse()

                if instance_prediction == valid_set_targets[instance_index]:
                    correct += 1

            vs_mse = vs_sse / len(valid_set)
            vs_accuracy = correct / len(valid_set)
            self.vs_accuracy = vs_accuracy

            # vs mse termination
            # print('{:.10f}'.format(train_mse) + ' - ' + '{:.10f}'.format(vs_mse) + ' - ' + '{:.10f}'.format(vs_accuracy))
            if vs_mse < best_vs_mse:
                best_vs_mse = vs_mse
                self.vs_mse = best_vs_mse
                best_at_epoch = epoch_count
                epochs_without_improvement = 0

                # assign the "best" hidden layer set
                best_hl = copy.deepcopy(self.hid_lays)
                best_out = copy.deepcopy(self.out_lay)
                best_weights = copy.deepcopy(self.wm)
            else:
                epochs_without_improvement += 1

            if self.write_to_file:
                out_file.write(str(vs_mse) + ',' + str(vs_accuracy) + '\n')
            epoch_count += 1

            if epochs_without_improvement > 5:
                learning = False

        self.hid_lays = best_hl
        self.out_lay = best_out
        self.wm = best_weights

        if self.write_to_file:
            out_file.close()
        self.best_epoch = best_at_epoch

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
        self.calculate_target_error(expected[0])

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

    def net_sse(self):
        # calculates the sse of the neural net as it is now
        error_sum = 0
        for node in self.out_lay:
            error_sum += math.pow(node.error, 2)
        return error_sum

    def measure_accuracy(self, features, labels, confusion=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: Matrix
        :type labels: Matrix
        :type confusion: Matrix
        :rtype float
        """

        if features.rows != labels.rows:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.cols != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.rows == 0:
            raise Exception("Expected at least one row")

        label_values_count = labels.value_count(0)
        if label_values_count == 0:
            # print('CONTINUOUS ACCURACY MEASURE')
            # label is continuous
            pred = [0]
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred[0] = 0.0  # make sure the prediction is not biased by a previous prediction
                # print("Target: " + str(targ[0]))
                self.predict(feat, pred)
                delta = targ[0] - pred[0]
                sse += delta ** 2
            return math.sqrt(sse / features.rows)

        else:
            # print('NOMINCAL ACCURACY MEASURE')
            # label is nominal, so measure predictive accuracy
            if confusion:
                confusion.set_size(label_values_count, label_values_count)
                confusion.attr_names = [labels.attr_value(0, i) for i in range(label_values_count)]

            correct_count = 0
            prediction = []
            for i in range(features.rows):
                feat = features.row(i)
                targ = int(labels.get(i, 0))
                if targ >= label_values_count:
                    raise Exception("The label is out of range")
                self.vs_predict(feat, prediction, [targ])

                pred = int(prediction[0])
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred) + 1)

                if pred == targ:
                    correct_count += 1

            return correct_count / features.rows
