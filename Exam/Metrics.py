__author__ = 'Michael'

import numpy as np


def true_pos_and_neg(predictions, gold):
    true_positive = [0] * len(set(predictions))
    true_negative = [0] * len(set(predictions))
    false_positive = [0] * len(set(predictions))
    false_negative = [0] * len(set(predictions))

    for i in xrange(len(true_positive)):
        true_positive[i] = sum([int(predictions[j] == i and gold[j] == i) for j in xrange(len(predictions))])
        true_negative[i] = sum([int(predictions[j] != i and gold[j] != i) for j in xrange(len(predictions))])
        false_positive[i] = sum([int(predictions[j] == i and gold[j] != i) for j in xrange(len(predictions))])
        false_negative[i] = sum([int(predictions[j] != i and gold[j] == i) for j in xrange(len(predictions))])

    return true_positive, true_negative, false_positive, false_negative


def precision(predictions, gold, average='micro'):
    true_pos, true_neg, false_pos, false_neg = true_pos_and_neg(predictions, gold)

    p = None

    if average == 'micro':
        p = float(sum(true_pos)) / (sum(true_pos) + sum(false_pos))
    elif average == 'macro':
        p = np.mean([float(true_pos[i]) / (true_pos[i] + false_pos[i]) for i in xrange(len(true_pos))])

    return p


def recall(predictions, gold, average='micro'):
    true_pos, true_neg, false_pos, false_neg = true_pos_and_neg(predictions, gold)

    r = None

    if average == 'micro':
        r = float(sum(true_pos)) / (sum(true_pos) + sum(false_neg))
    elif average == 'macro':
        r = np.mean([float(true_pos[i]) / (true_pos[i] + false_neg[i]) for i in xrange(len(true_pos))])

    return r


def f1(predictions, gold, average='micro'):
    p = precision(predictions, gold, average=average)
    r = recall(predictions, gold, average=average)
    return 2*p*r/(p+r)


if __name__ == '__main__':
    print f1([0,0,0,0,0,0,0,0], [1, 2, 3, 3, 3, 2, 1, 0], average='macro')