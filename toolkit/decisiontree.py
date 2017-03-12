from supervised_learner import SupervisedLearner
import math
import sys
from matrix import Matrix
import copy

debug = False


class Node(object):
    node_uuid_count = 0

    def __init__(self, features, labels, ignore):
        self.uuid = Node.node_uuid_count
        Node.node_uuid_count += 1

        # if the node is a branch, it will have children
        # key: the enumerated value of the feature_value associated with this path - value: node that decides the next feature
        self.paths = dict()
        self.path_counts = dict()
        # if the node is a leaf node, this will hold it's classification
        self.classification = None
        # if the node is a decision node, this variable will hold the index of the feature it chooses on
        self.index_of_deciding_feature = None

        self.features = features
        self.labels = labels
        self.ignore = copy.deepcopy(ignore)

        self.feature_split = None
        self.label_split = None

        self.build_node()

    def clean_path(self):
        for key in self.paths:
            if self.paths[key] is None:
                self.paths.pop(key)

        delete_these_keys = []
        for key in self.path_counts:
            if key not in self.paths:
                delete_these_keys.append(key)

        while len(delete_these_keys) > 0:
            self.path_counts.pop(delete_these_keys[0])
            del delete_these_keys[0]

    def predict(self, instance):
        if self.classification is not None:
            # if this is a leaf node, return this node's classification
            return self.classification

        self.clean_path()

        if len(self.paths) == 0:
            # if this is not a classification node AND there are no paths (result of pruning), return -1 as an immediate classification failure
            return -1

        if instance[self.index_of_deciding_feature] not in self.paths:
            # if this node does note have a path for the feature it decides, take the path with the biggest weight
            biggest_path = -1
            biggest_path_count = - sys.maxsize
            self.clean_path()

            for key in self.path_counts:
                if self.path_counts[key] > biggest_path_count:
                    biggest_path = key
                    biggest_path_count = self.path_counts[key]
            return self.paths[biggest_path].predict(instance)
        else:
            # otherwise, take the path intended by this node
            return self.paths[instance[self.index_of_deciding_feature]].predict(instance)

    # features: the instances from the training set
    # labels: the classes from the training set
    # ignore: a set of col_indices to ignore, meaning they have been split on previously, the nature of how the attributes are named by column index makes this necessary
    def build_node(self):
        # check for single class in labels
        num_classifications = count_classifications(self.labels)
        # if there is a single class
        #   set this node's classification to that class and stop
        if num_classifications == 1:
            self.classification = self.labels.row(0)[0]
            if debug:
                print('ARRIVED AT LEAF NODE - classification: ' + str(self.classification))
                print_features_and_labels(self.features, self.labels)
                print()

            return

        # else
        #   calc info gain for all indices not in the feature ignore set - return a feature index
        self.index_of_deciding_feature = calc_next_feature(self.features, self.labels, self.ignore)

        if self.index_of_deciding_feature is None:
            return
        self.ignore.add(self.index_of_deciding_feature)

        #   split on feature with highest info gain
        # feature_value: matrix of terms
        self.feature_split, self.label_split = split_matrix_by_feature(self.features, self.labels,
                                                                       self.index_of_deciding_feature)

        if debug:
            print('SPLITTING ON: ' + str(self.index_of_deciding_feature) + '\t\t' + str(self.feature_split))
            for key in self.feature_split:
                print_features_and_labels(self.feature_split[key], self.label_split[key])
                print('-------')
            print('=================================================================================================')

        # create new children nodes from each split
        for key in self.feature_split:
            self.paths[key] = Node(self.feature_split[key], self.label_split[key], self.ignore)
            self.path_counts[key] = self.feature_split[key].rows


# truncates the print_me string to a specific width, adds trailing spaces to the end of short strings
def string_to_width(print_me, width):
    print_me_list = list(print_me)
    if len(print_me_list) > width:
        return ''.join(print_me_list[:width])
    else:
        while len(print_me_list) < width:
            print_me_list.append(' ')
        return ''.join(print_me_list[:width])


def print_features_and_labels(features, labels):
    for row_index in range(features.rows):
        for col_index in range(features.cols):
            print('\t' + string_to_width(features.enum_to_str_access(row_index, col_index), 10), end='\t')
        print(labels.enum_to_str_access(row_index, 0))


def count_classifications(labels):
    class_set = set()
    for row_index in range(labels.rows):
        instance = labels.row(row_index)
        if instance[0] not in class_set:
            class_set.add(instance[0])
    return len(class_set)


def split_matrix_by_feature(features, labels, feature_col_index):
    feature_split = dict()
    label_split = dict()

    for row_index in range(features.rows):
        instance = features.row(row_index)
        feature_value = instance[feature_col_index]

        if feature_value in feature_split:
            feature_split[feature_value].add_row(features, row_index, 0, features.cols)
            label_split[feature_value].add_row(labels, row_index, 0, labels.cols)
        else:
            feature_split[feature_value] = Matrix(features, row_index, 0, 1, features.cols)
            label_split[feature_value] = Matrix(labels, row_index, 0, 1, labels.cols)

    return feature_split, label_split


# given a feature/label set, calculate the info data and return the index of the next col to split on
def calc_next_feature(features, labels, ignore):
    class_count_map = dict()
    partition_info_gain = dict()

    # evaluate the classifications to calculate InfoS
    for label in labels.col(0):
        if label in class_count_map:
            class_count_map[label] += 1
        else:
            class_count_map[label] = 1

    info_s = 0
    for key in class_count_map:
        p = class_count_map[key] / labels.rows
        info_s -= p * math.log(p, 2)
    # info_s is now calculated

    # feature_entry -> {feature_option1: {class1:count, class2: count, class3: count}, feature_option2: ...}
    # meat[y/n] -> {y:{great:0, good:0, bad:0}, n:{great:0, good:0, bad:0}}
    for col_index in range(features.cols):
        if col_index in ignore:
            continue

        data = dict()
        # move through each instance, create a map where the key is each feature entry and the value is a map of
        # their classification counts
        for row_index in range(features.rows):
            instance = features.row(row_index)
            feature_entry = instance[col_index]
            classification = labels.row(row_index)[0]

            if feature_entry in data:
                data[feature_entry][classification] += 1
            else:
                data[feature_entry] = dict()
                for class_type in class_count_map:
                    data[feature_entry][class_type] = 0
                data[feature_entry][classification] += 1

        # calculate their info values ... feature_entry is the 'N' or 'Y' for 'MEAT'
        info_feature = -1
        for feature_entry in data:
            feature_count = 0
            for class_count in data[feature_entry]:
                feature_count += data[feature_entry][class_count]

            # calc the sum of the individual entries
            log_comp = 0
            for class_count in data[feature_entry]:
                log_comp -= log_calc(data[feature_entry][class_count], feature_count)
            info_feature += (feature_count / features.rows) * log_comp

        # partition info now contains the information of each feature mapped col_index to value
        partition_info_gain[col_index] = info_s - info_feature
    next_split = max_index(partition_info_gain)
    return next_split


# helper function
def log_calc(numer, denom):
    if (numer / denom == 0):
        return 0
    return (numer / denom) * math.log((numer / denom), 2)


# helper function
def max_index(d):
    max_value_index = None
    max_value = -sys.maxsize
    for key in d:
        if d[key] > max_value:
            max_value = d[key]
            max_value_index = key
    return max_value_index


def count_tree_nodes(root_node):
    count = 1
    for key in root_node.paths:
        count += count_tree_nodes(root_node.paths[key])
    return count


def count_branch_nodes(root_node):
    count = 0
    if len(root_node.paths) > 0:
        count = 1

    for node_key in root_node.paths:
        count += count_branch_nodes(root_node.paths[node_key])

    return count


def count_tree_depth(root_node):
    highest_depth = 0
    for key in root_node.paths:
        path_depth = count_tree_depth(root_node.paths[key])
        if path_depth > highest_depth:
            highest_depth = path_depth
    return highest_depth + 1


def collect_branch_node_uuids(root_node):
    return_me = []
    # if this node is not a leaf node
    if len(root_node.paths) > 0:
        # add it to the list of nodes to return
        if root_node.uuid != 0:
            return_me.append(root_node.uuid)

        # check it's children
        for key in root_node.paths:
            return_me += collect_branch_node_uuids(root_node.paths[key])

    return return_me


def remove_node_by_uuid(uuid, root_node):
    for node_key in root_node.paths:
        child_uuid = root_node.paths[node_key].uuid
        if child_uuid == uuid:
            root_node.paths.pop(node_key)
            return
        else:
            remove_node_by_uuid(uuid, root_node.paths[node_key])


class DecisionTreeLearner(SupervisedLearner):
    REP = True
    valid_set_percent = 0.25

    def __init__(self):
        self.root_node = None

    def train(self, features, labels):
        print('----------------------------------------------------------------------------------------')

        features.shuffle(labels)

        train_set_instances = Matrix(features, 0, 0, 0, features.cols)
        train_set_labels = Matrix(labels, 0, 0, 0, labels.cols)
        valid_set_instances = Matrix(features, 0, 0, 0, features.cols)
        valid_set_labels = Matrix(labels, 0, 0, 0, labels.cols)

        # split features into valid and train sets
        for row_index in range(features.rows):
            if (row_index * DecisionTreeLearner.valid_set_percent) % 1 == 0:
                train_set_instances.add_row(features, row_index, 0, features.cols)
                train_set_labels.add_row(labels, row_index, 0, labels.cols)
            else:
                valid_set_instances.add_row(features, row_index, 0, features.cols)
                valid_set_labels.add_row(labels, row_index, 0, labels.cols)

        self.root_node = Node(train_set_instances, train_set_labels, set())
        print('TOTAL NODES: ' + str(count_tree_nodes(self.root_node)))
        print('BRANCH NODES: ' + str(count_branch_nodes(self.root_node)))
        print('TREE DEPTH: ' + str(count_tree_depth(self.root_node)))
        print('FULL TREE VALID ACCURACY: ' + str(self.measure_accuracy(valid_set_instances, valid_set_labels)))
        print()

        # Reduce error prune the tree
        if DecisionTreeLearner.REP:
            best_pruned_root = copy.deepcopy(self.root_node)
            best_accuracy = self.measure_accuracy(valid_set_instances, valid_set_labels)
            pruning = True

            while pruning:
                # while we have found a node which, when pruned, increases  accuracy
                pruning = False
                # map the nodes of the current best tree
                node_uuids = collect_branch_node_uuids(best_pruned_root)
                current_best_root = best_pruned_root
                current_best_accuracy = best_accuracy

                # for each node in the current best
                for node_uuid in node_uuids:
                    # deep copy the best as to not destroy it
                    pruned_tree_root = copy.deepcopy(best_pruned_root)
                    # map the copy's nodes
                    pruned_node_map = collect_branch_node_uuids(pruned_tree_root)

                    # delete a node at the current test index
                    # del pruned_node_map[node_index]
                    remove_node_by_uuid(node_uuid, pruned_tree_root)
                    # print('JUST PRUDED ... NODE COUNT: ' + str(len(collect_branch_node_uuids(pruned_tree_root))))

                    # test accuracy
                    pruned_accuracy = self.measure_accuracy(valid_set_instances, valid_set_labels, confusion=None,
                                                            root_node=pruned_tree_root)

                    # print('prune accuracy: ' + str(pruned_accuracy))
                    # if the new accuracy is better, keep this tree as the new current
                    if pruned_accuracy > best_accuracy:
                        print('PRUNING A NODE - new accuracy: ' + str(pruned_accuracy))
                        best_pruned_root = copy.deepcopy(pruned_tree_root)
                        best_accuracy = pruned_accuracy
                        pruning = True

                if True:
                    pass

            # set the pruned tree as the DT's root node
            self.root_node = best_pruned_root
            print('PRUNED TOTAL NODES: ' + str(count_tree_nodes(self.root_node)))
            print('PRUNED BRANCH NODES: ' + str(count_branch_nodes(self.root_node)))
            print('PRUNED TREE DEPTH: ' + str(count_tree_depth(self.root_node)))
            print()

    def predict(self, features, labels, root_node=None):
        if root_node is None:
            guess = self.root_node.predict(features)
        else:
            guess = root_node.predict(features)

        if len(labels) == 0:
            labels.append(guess)
        else:
            labels[0] = guess

    def measure_accuracy(self, features, labels, confusion=None, root_node=None):
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
            pred = []
            sse = 0.0
            for i in range(features.rows):
                feat = features.row(i)
                targ = labels.row(i)
                pred[0] = 0.0  # make sure the prediction is not biased by a previous prediction
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
                self.predict(feat, prediction, root_node)

                pred = int(prediction[0])
                if confusion:
                    confusion.set(targ, pred, confusion.get(targ, pred) + 1)

                if pred == targ:
                    correct_count += 1

            return correct_count / features.rows
