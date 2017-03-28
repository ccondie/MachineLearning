from supervised_learner import SupervisedLearner
import math
import operator
from matrix import Matrix


class InstanceBasedLearner(SupervisedLearner):
    def __init__(self):
        self.weightFlag = True
        self.k = 15
        self.instances = None
        self.classes = None
        self.predict_count = 1
        self.train_max_size = 2000

    def train(self, instances, classes):
        self.instances = Matrix(instances, 0, 0, 0, instances.cols)
        self.classes = Matrix(classes, 0, 0, 0, classes.cols)

        instances.shuffle(classes)

        # build the train set
        if instances.rows < self.train_max_size:
            instance_count = instances.rows
        else:
            instance_count = self.train_max_size

        for i in range(instance_count):
            self.instances.add_row(instances, i, 0, instances.cols)
            self.classes.add_row(classes, i, 0, classes.cols)

    def predict(self, instance, classif):
        if self.predict_count % 100 == 0:
            print(self.predict_count, end=', ', flush=True)
        if self.predict_count % 1000 == 0:
            print()
        self.predict_count += 1
        # distance_map[train_index] = distance_value
        distance_map = dict()
        # calculate the distances from the inputted distance ... map each distance to train_instance index
        for train_index in range(self.instances.rows):
            distance_map[train_index] = self.dist(instance, self.instances.row(train_index))

        # sort the distance map, index 0 is smallest distance
        distance_sorted = sorted(distance_map.items(), key=operator.itemgetter(1))

        if self.classes.value_count(0) == 0:
            # classification is continuous
            reg_members = list()
            for k in range(self.k):
                dist_value = distance_sorted[k][1]
                cont_value = self.classes.row(distance_sorted[k][0])[0]
                # print("dist: " + str(dist_value) + " - point: " + str(cont_value), flush=True)
                reg_members.append([cont_value, dist_value])

            numerator = 0
            denominator = 0
            if self.weightFlag:
                for reg_member in reg_members:
                    numerator += reg_member[0]/math.pow(reg_member[1],2)
                    denominator += 1/math.pow(reg_member[1],2)
            else:
                for reg_member in reg_members:
                    numerator += reg_member[0]
                    denominator += 1

            reg_value = numerator / denominator
            # print("Prediction: " + str(reg_value))
            # print()

            # set the "return" value
            if len(classif) == 0:
                classif.append(reg_value)
            else:
                classif[0] = reg_value

        else:
            # classification is nominal
            vote_map = dict()
            for vote_index in range(self.k):
                # classification is nominal
                # vote_map[nominal_value] = vote_count
                if self.weightFlag:
                    # weighted
                    nominal_value = self.classes.row(distance_sorted[vote_index][0])[0]
                    denom = math.pow(distance_sorted[vote_index][1], 2)
                    if nominal_value in vote_map:
                        if denom == 0:
                            vote_map[nominal_value] += 1 / 0.0000000000000000000000000000000001
                        else:
                            vote_map[nominal_value] += 1 / denom
                    else:
                        if denom == 0:
                            vote_map[nominal_value] = 1 / 0.0000000000000000000000000000000001
                        else:
                            vote_map[nominal_value] = 1 / denom
                else:
                    # unweighted
                    nominal_value = self.classes.row(distance_sorted[vote_index][0])[0]
                    if nominal_value in vote_map:
                        vote_map[nominal_value] += 1
                    else:
                        vote_map[nominal_value] = 1
            # sort the votes and return the nominal value with the most votes
            vote_sorted = sorted(vote_map.items(), key=operator.itemgetter(1))

            # set the "return" value
            if len(classif) == 0:
                classif.append(vote_sorted[-1][0])
            else:
                classif[0] = vote_sorted[-1][0]

    def dist(self, test_row, train_row):
        sum_value = 0
        for col_index in range(len(train_row)):
            if self.instances.value_count(col_index) == 0:
                # value is continuous
                sum_value += math.pow(train_row[col_index] - test_row[col_index], 2)
            else:
                # value is nominal
                if test_row[col_index] == train_row[col_index]:
                    sum_value += 0
                else:
                    sum_value += 1
        return math.sqrt(sum_value)
