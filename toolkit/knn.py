from supervised_learner import SupervisedLearner
import math


class InstanceBasedLearner(SupervisedLearner):
    def __init__(self):
        self.weightFlag = False
        self.k = 3
        self.instances = None
        self.classes = None

    def train(self, instances, classes):
        # NOT SURE THIS IS ACTUALLY NEEDED
        self.instances = instances
        self.classes = classes

    def predict(self, instances, classes):

        pass

    def dist(self, inst1, inst2):
        for row_index in range(inst1.rows):
            row = inst1.row(row_index)
            for col_index in range(inst1.cols):
                print(str(row[col_index]) + ':' + str(inst1.value_count(col_index)), end=', ')
            print()
