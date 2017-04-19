from __future__ import (absolute_import, division, print_function, unicode_literals)

from datetime import datetime

from supervised_learner import SupervisedLearner
from baseline_learner import BaselineLearner
from neuralnet import NeuralNetLearner
from decisiontree import DecisionTreeLearner
from knn import InstanceBasedLearner
from matrix import Matrix
import random
import argparse
import time


class MLSystemManager:
    def get_learner(self, model):
        """
        Get an instance of a learner for the given model name.

        To use toolkitPython as external package, you can extend this class (MLSystemManager)
        with your own custom class located outside of this package, and override this method
        to return your custom learners.

        :type model: str
        :rtype: SupervisedLearner
        """
        modelmap = {
            "baseline": BaselineLearner(),
            # "perceptron": PerceptronLearner(),
            "neuralnet": NeuralNetLearner(),
            "decisiontree": DecisionTreeLearner(),
            "knn": InstanceBasedLearner()
        }
        if model in modelmap:
            return modelmap[model]
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def main(self):
        # parse the command-line arguments
        args = self.parser().parse_args()
        file_name = args.arff
        learner_name = args.L
        eval_method = args.E[0]
        eval_parameter = args.E[1] if len(args.E) > 1 else None
        print_confusion_matrix = args.verbose
        normalize = args.normalize
        random.seed(args.seed)  # Use a seed for deterministic results, if provided (makes debugging easier)

        # load the model
        learner = self.get_learner(learner_name)
        learner = NeuralNetLearner()

        # load the ARFF file
        data = Matrix()
        data.load_arff(file_name)
        if normalize:
            print("Using normalized data")
            data.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(file_name, data.rows, data.cols, learner_name, eval_method))

        if eval_method == "random":

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            # print("Percentage used for training: {}".format(train_percent))
            # print("Percentage used for testing: {}".format(1 - train_percent))

            output_file = open('learningRate_{:%Y-%m-%d_%H-%M-%S}.csv'.format(datetime.now()), 'a')
            output_file.write('hid_count,train_mse,vs_mse,test_mse,vs_accuracy,test_accuracy\n')

            learning_rates = [0.11, .12, .13, .14, .15, .16, .17, .18, .19]
            hid_counts = [1, 2, 4, 8, 16, 32, 64]

            test_count = 3
            for hid_count in hid_counts:
                train_mse_sum = 0
                vs_mse_sum = 0
                test_mse_sum = 0
                vs_accuracy_sum = 0
                test_accuracy_sum = 0

                for i in range(test_count):
                    print('hid_count: ' + str(hid_count))
                    data.shuffle()

                    train_size = int(train_percent * data.rows)
                    train_features = Matrix(data, 0, 0, train_size, data.cols - 1)
                    train_labels = Matrix(data, 0, data.cols - 1, train_size, 1)

                    test_features = Matrix(data, train_size, 0, data.rows - train_size, data.cols - 1)
                    test_labels = Matrix(data, train_size, data.cols - 1, data.rows - train_size, 1)

                    learner.train(train_features, train_labels, 0.1, hid_count)
                    vs_accuracy = learner.vs_accuracy

                    train_mse_sum += learner.train_mse
                    vs_mse_sum += learner.vs_mse

                    test_accuracy = learner.measure_accuracy(test_features, test_labels)
                    test_mse = learner.net_sse() / (data.rows - train_size)

                    test_mse_sum += test_mse
                    vs_accuracy_sum += vs_accuracy
                    test_accuracy_sum += test_accuracy

                train_mse_avg = train_mse_sum / test_count
                vs_mse_avg = vs_mse_sum / test_count
                test_mse_avg = test_mse_sum / test_count
                vs_accuracy_avg = vs_accuracy_sum / test_count
                test_accuracy_avg = test_accuracy_sum / test_count

                output_file.write(
                    str(hid_count) + ',' + str(train_mse_avg) + ',' + str(vs_mse_avg) + ',' + str(test_mse_avg) + ',' + str(
                        vs_accuracy_avg) + ',' + str(test_accuracy_avg) + '\n')
            output_file.close()

    def parser(self):
        parser = argparse.ArgumentParser(description='Machine Learning System Manager')

        parser.add_argument('-V', '--verbose', action='store_true',
                            help='Print the confusion matrix and learner accuracy on individual class values')
        parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
        parser.add_argument('-R', '--seed', help="Random seed")  # will give a string
        parser.add_argument('-L', required=True, choices=['baseline', 'perceptron', 'neuralnet', 'decisiontree', 'knn'],
                            help='Learning Algorithm')
        parser.add_argument('-A', '--arff', metavar='filename', required=True, help='ARFF file')
        parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+',
                            help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)")

        return parser


if __name__ == '__main__':
    MLSystemManager().main()
