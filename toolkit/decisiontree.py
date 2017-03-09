from supervised_learner import SupervisedLearner
import math
import sys


class Node(object):
    def __init__(self):
        # if the node is a branch, it will have children
        self.children = []
        # if the node is a leaf node, this will hold it's classification
        self.classification = None

    # features: the instances from the training set
    # labels: the classes from the training set
    # ignore: a set of col_indices to ignore, meaning they have been split on previously, the nature of how the attributes are named by column index makes this necessary
    def build_node(self, features, labels, ignore):
        # check for single class in labels
        num_classifications = labels.value_count(0)
        # if there is a single class
        #   set this node's classification to that class and stop
        if num_classifications == 1:
            self.classification = labels.row(0)[0]
            return

        # else
        #   calc info gain for all indices not in the feature ignore set - return a feature index
        next_split_col_index = calc_next_feature(features, labels, ignore)
        #   split on feature with highest info gain
        #   create new children nodes from each split
        pass

    # creates a series of nodes attached to this node representing decisions in the tree
    # entries: a list of possible entries for a specific feature
    def gen_nodes(self, entries):
        for el in entries:
            pass
        pass

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
        info_feature = 0
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
    print('spitting on ' + str(next_split))
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
    index = 0
    for key in d:
        if d[key] > max_value:
            max_value = d[key]
            max_value_index = index
        index += 1
    return max_value_index


class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        self.root_node = None

    def train(self, features, labels):
        self.root_node = Node()
        self.root_node.build_node(features, labels, set())

    def predict(self, features, labels):
        pass
