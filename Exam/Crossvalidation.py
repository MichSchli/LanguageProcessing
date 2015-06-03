__author__ = 'Michael'

from itertools import chain
import RelationExtraction
from matplotlib import pyplot as plt
import numpy as np
import PartOfSpeechTagger as POS
import sys

def cross_validate(data, labels, k, create, params):
    #Split into evenly sized chunks
    samples_per_fold = len(data)/k
    data_folds = [list(t) for t in zip(*[iter(data)]*samples_per_fold)]
    label_folds = [list(t) for t in zip(*[iter(labels)]*samples_per_fold)]

    #Distribute the remainder evenly over the folds
    leftover_data = data[samples_per_fold*k:]
    leftover_labels = labels[samples_per_fold*k:]
    for i in xrange(len(leftover_data)):
        data_folds[i%k].append((leftover_data[i]))
        label_folds[i%k].append((leftover_labels[i]))

    acc = 0

    #Do the experiments
    for i in xrange(len(data_folds)):
        #Get a view of the data
        train_data = data_folds[:]
        train_labels = label_folds[:]

        #Construct training and test sets
        validate_data = train_data.pop(i)
        validate_labels = train_labels.pop(i)
        train_data = list(chain.from_iterable(train_data))
        train_labels = list(chain.from_iterable(train_labels))

        #Create a classifier
        model = create(*params)

        #Fit the classifier to the data:
        model.fit_sentences(train_data, train_labels)

        #Evaluate accuracy
        acc += model.evaluate_sentences(validate_data, validate_labels)[-1]

    return acc/float(k)


def do_grid_search(data, labels, range1, range2, cval_func):
    results = np.ndarray((len(range1),len(range2)))
    print >> sys.stderr, "Doing grid search..."

    best = None
    best_result = 1.0
    for i,C in enumerate(range1):
        for j,gamma in enumerate(range2):
            results[i][j] = cval_func(data, labels, C, gamma)
            print >> sys.stderr, C, gamma, results[i][j]

            if results[i][j] < best_result:
                best_result = results[i][j]
                best = (C, gamma)

    # draw heatmap of accuracy as a function of gamma and C
    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.2, top=0.95)
    plt.imshow(results, interpolation='nearest', cmap=plt.cm.spectral)
    plt.clim(0,1)
    plt.title('Heatmap of f1 score')
    plt.xlabel('Value 2')
    plt.ylabel('Value 1')
    plt.colorbar()
    plt.xticks(np.arange(len(range2)), range2, rotation=45)
    plt.yticks(np.arange(len(range1)), range1)

    plt.show()

    return best[0], best[1], best_result



def find_best_svm_params_detector(data, labels):
    def create(mode, p):
        return RelationExtraction.RelationDetector(mode, p)

    scale = [10**v for v in xrange(-6,6)]

    def cval_func(data, labels, param1, param2):
        return cross_validate(data, labels, 5, create, ['SVM', [param1, param2]])

    do_grid_search(data, labels, scale, scale, cval_func)


def evaluate_pos_tagger(data, labels):
    def create():
        return POS.StructuredPerceptron()

    return cross_validate(data, labels, 5, create, [])