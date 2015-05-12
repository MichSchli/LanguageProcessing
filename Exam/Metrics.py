__author__ = 'Michael'

import numpy as np


def true_pos_and_neg(predictions, gold, n_classes, count_negative=False):
    true_positive = [0] * n_classes
    true_negative = [0] * n_classes
    false_positive = [0] * n_classes
    false_negative = [0] * n_classes

    sp = 0 if count_negative else 1

    for i in xrange(sp, len(true_positive)):
        true_positive[i] = sum([int(predictions[j] == i and gold[j] == i) for j in xrange(len(predictions))])
        true_negative[i] = sum([int(predictions[j] != i and gold[j] != i) for j in xrange(len(predictions))])
        false_positive[i] = sum([int(predictions[j] == i and gold[j] != i) for j in xrange(len(predictions))])
        false_negative[i] = sum([int(predictions[j] != i and gold[j] == i) for j in xrange(len(predictions))])

    if not count_negative:
        true_positive.pop(0)
        true_negative.pop(0)
        false_negative.pop(0)
        false_positive.pop(0)

    return true_positive, true_negative, false_positive, false_negative


def precision(predictions, gold, n_classes, average='micro'):
    true_pos, true_neg, false_pos, false_neg = true_pos_and_neg(predictions, gold, n_classes)

    p = None

    if average == 'micro':
        if (sum(true_pos) + sum(false_pos)) == 0 and sum(false_neg) != 0:
            return 0.0
        p = float(sum(true_pos)) / (sum(true_pos) + sum(false_pos))
    elif average == 'macro':
        p = np.mean([float(true_pos[i]) / (true_pos[i] + false_pos[i]) for i in xrange(len(true_pos))])

    return p


def recall(predictions, gold, n_classes, average='micro'):
    true_pos, true_neg, false_pos, false_neg = true_pos_and_neg(predictions, gold, n_classes)

    r = None

    if average == 'micro':
        r = float(sum(true_pos)) / (sum(true_pos) + sum(false_neg))
    elif average == 'macro':
        r = np.mean([float(true_pos[i]) / (true_pos[i] + false_neg[i]) for i in xrange(len(true_pos))])

    return r


def f1(predictions, gold, n_classes, average='micro'):
    p = precision(predictions, gold, n_classes, average=average)
    r = recall(predictions, gold, n_classes, average=average)

    ##TODO: ask about this hack
    if p + r == 0:
        return 0.0

    return 2*p*r/(p+r)


if __name__ == '__main__':
    print f1([0,0,0,0,0,0,0,0], [1, 2, 3, 3, 3, 2, 1, 0], average='macro')