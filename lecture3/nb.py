import argparse
import codecs
from collections import defaultdict
from scipy import log
import math

__author__ = 'dirkhovy, bplank'

import codecs

class NaiveBayes(object):

    def __init__(self):
        self.label_counts = defaultdict(lambda : 1)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: 1))


    def fit(self, file_name):
        """
        read in a tagged file in CoNLL format:
        label    feature1 feature2 ... featureN
        and collect counts for the relevant model parameters
        i.e.
           label_counts: count how often each label (class) appears in the data
           feature_counts: for every label, count how often each feature is on in the data
        :param file_name:
        :return:
        """
        #Read in a featurized text:


        for (label, features) in self.read_conll_file(file_name):

            ##########################
            # *** YOUR CODE HERE *** #
            # populate self.label_counts and self.feature_counts
            ##########################
            self.label_counts[label] += 1
            for i in xrange(len(features)):
                self.feature_counts[label][i] += int(features[i])



            # End your code


    def predict(self, features):
        """
        predicts the most likely class label under the model given the features
        :param features: list of 0s and 1s
        :return: predicted label
        """

        best_class = None
        best_prob = float('-Inf')

        ##########################
        # *** YOUR CODE HERE *** #
        # implement
        #     argmax_y log P(y) sum log P(x_i|y) => argmax_y log P(y) + sum log P(x_i|y) [see sheet]
        ##########################


        for label in self.label_counts.keys():

            logp = math.log(self.P_y(label))

            for i in xrange(len(features)):
                prob = self.P_x_given_y(label, i)
                logp += math.log(prob) if features[i] else math.log(1-prob)

            if logp > best_prob:
                best_class = label
                best_prob = logp


        return best_class


    def P_y(self, label):
        """
        compute prior probability of the label under the model
        :param label:
        :return: P(y)
        """

        ##########################
        # *** YOUR CODE HERE *** #
        # extract label (prior) probabilities from label_counts
        ##########################



        return self.label_counts[label]/float(sum(self.label_counts.values()))


    def P_x_given_y(self, label, feature_id):
        """
        compute probability of the feature given the label under the model
        :param label:
        :return: P(x|y)
        """
        P_x_given_y = 0.0

        ##########################
        # *** YOUR CODE HERE *** #
        # initialize P_x_given_y to the probability of feature with feature_id given class label
        ##########################

        return self.feature_counts[label][feature_id]/float(self.label_counts[label]+1)


    def evaluate(self, test_file):
        """
        evaluate performance of the model by testing on a held-out file
        :param test_file:
        :return:
        """
        correct = 0.0
        total = 0.0

        for (label, features) in self.read_conll_file(test_file):
            ##########################
            # ***    OPTIONAL:   *** #
            # *** YOUR CODE HERE *** #
            ##########################

            total += 1.0

            prediction = self.predict(features)
            if prediction == label:
                correct += 1.0

        accuracy = correct/total
        #cprecision =

        return correct/total


    def read_conll_file(self, file_name):
        """
        read in a file with label\tfeaturelist
        :param file_name:
        :return: generator of (label, features) pair
        """

        for line in codecs.open(file_name, encoding='utf-8'):
            line = line.strip()

            if line:
                label, features = line.split('\t')
                features = map(int, features.split(' '))

                yield (label, features)




# if script is run from command line, automatically execute the following
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Creates a Naive Bayes Model from text""")
    parser.add_argument("train", help="corpus to estimate NB from")
    parser.add_argument("--test", help="test model on a file (one sentence per line)", required=False)
    args = parser.parse_args()

    # create new model
    nb= NaiveBayes()

    # fit model to supplied file
    nb.fit(args.train)

    # check whether to show predictions
    if args.test:
        print nb.evaluate(args.test)
