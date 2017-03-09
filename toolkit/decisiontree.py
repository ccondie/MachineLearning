from supervised_learner import SupervisedLearner
import math


class Node(object):
    def __init__(self):
        # if the node is a branch, it will have children
        self.children = []
        # if the node is a leaf node, this will hold it's classification
        self.classification = None

    # creates a series of nodes attached to this node representing decisions in the tree
    # entries: a list of possible entries for a specific feature
    def gen_nodes(self, entries):
        for el in entries:
            pass
        pass


class DecisionTreeLearner(SupervisedLearner):
    def __init__(self):
        pass

    def train(self, features, labels):
        # split training set
        class_count_map = dict()
        class_info = dict()

        # evaluate the classifications to calculate InfoS
        for label in labels.col(0):
            if label in class_count_map:
                class_count_map[label] += 1
            else:
                class_count_map[label] = 1

        infoS = 0
        for key in class_count_map:
            p = class_count_map[key] / labels.rows
            infoS -= p * math.log(p, 2)

        # feature_entry -> {feature_option1: {class1:count, class2: count, class3: count}, feature_option2: ...}
        # meat[y/n] -> {y:{great:0, good:0, bad:0}, n:{great:0, good:0, bad:0}}
        for col_index in range(features.cols):
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
                    log_comp -= self.log_calc(data[feature_entry][class_count], feature_count)
                info_feature += (feature_count / features.rows) * log_comp

            print(info_feature)

    def log_calc(self, numer, denom):
        if(numer/denom == 0):
            return 0
        return (numer / denom) * math.log((numer / denom), 2)

    def predict(self, features, labels):
        pass
