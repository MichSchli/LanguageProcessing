from collections import defaultdict
import codecs
import argparse
import sys


parser = argparse.ArgumentParser(description="""Evaluate relations""")
parser.add_argument("--gold", help="gold file (CoNLL format)", required=True)
parser.add_argument("--prediction", help="prediction file (CoNLL format)", required=True)
args = parser.parse_args()


def read_RE_file(file_name):
    """
    read a ConLL file
    # FORMAT:
    #ARG    NER     ID  chunk   POS         token       ???     ?   ?

    #START_ENTITY    END_ENTITY    RELATION

    :param file_name:
    :return: generator of entity/relation triples
    """
    instance=[]
    rel_output = False

    for line in codecs.open(file_name, encoding='utf-8'):
        line = line.strip()

        if line:
            fields = line.split("\t")

            if len(fields) not in [3,4]:
                print>>sys.stderr, "File {} not in right format!".format(file_name)
                exit()

            if len(fields) == 3:
                # print fields

                arg0 = fields[0]
                if ':' in arg0:
                    i, j = map(int, arg0.split(':'))
                    arg0 = range(i, j+1)
                else:
                    arg0 = [int(arg0)]

                arg1 = fields[1]
                if ':' in arg1:
                    i, j = map(int, arg1.split(':'))
                    arg1 = range(i, j+1)
                else:
                    arg1 = [int(arg1)]

                rel = fields[2]

                # print arg0, arg1, rel
                # print
                instance.append((arg0, arg1, rel))

        else:
            # after relation output, commit
            if rel_output:
                rel_output = False
                yield instance
                instance = []
            else:
                rel_output = True

    if rel_output and instance:
        yield instance


def overlap(gold_arg0, gold_arg1, gold_rel, prediction_arg0, prediction_arg1, prediction_rel):
    """
    check whether the two spans overlap
    :param gold:
    :param prediction:
    :return:
    """
    # arg0_overlap = len(set(gold[0]).intersection(set(prediction[0]))) / float(len(gold[0]))
    # arg1_overlap = len(set(gold[1]).intersection(set(prediction[1]))) / float(len(gold[1]))
    arg0_overlap = set(gold_arg0).intersection(set(prediction_arg0)) != set()
    arg1_overlap = set(gold_arg1).intersection(set(prediction_arg1)) != set()

    return arg0_overlap and arg1_overlap


# read in files
predictions = read_RE_file(args.prediction)
gold = read_RE_file(args.gold)


# collect raw counts for true and false cases
true_positives = defaultdict(float)
false_positives = defaultdict(float)
false_negatives = defaultdict(float)


for gold_instance, prediction_instance in zip(gold, predictions):
    matched_predictions = [False] * len(prediction_instance)
    print gold_instance, prediction_instance
    for (gold_arg0, gold_arg1, gold_rel) in gold_instance:
        match = False
        for i, (prediction_arg0, prediction_arg1, prediction_rel) in enumerate(prediction_instance):
            if overlap(gold_arg0, gold_arg1, gold_rel, prediction_arg0, prediction_arg1, prediction_rel):
                match = True
                matched_predictions[i] = True

                if gold_rel == prediction_rel:
                    true_positives[gold_rel] += 1.0
                else:
                    false_positives[prediction_rel] += 1.0
                    false_negatives[gold_rel] += 1.0

        if not match:
            false_negatives[gold_rel] += 1.0


    for unmatched in [i for i, value in enumerate(matched_predictions) if not value]:
        prediction_arg0, prediction_arg1, prediction_rel = prediction_instance[unmatched]
        false_positives[prediction_rel] += 1.0

relations = set(false_positives.keys()).union(set(false_negatives)).union(set(true_positives))

debug=0
if debug:
    print "TP", true_positives
    print "FP", false_positives
    print "FN", false_negatives

print
print "RELATION\tPREC\tREC\tF1"
print '-' * 45

for relation in relations:
    # compute precision, recall and F1 from the raw counts
    try:
        precision = true_positives[relation] / (true_positives[relation] + false_positives[relation])
    except ZeroDivisionError:
        precision = 0.0

    try:
        recall = true_positives[relation] / (true_positives[relation] + false_negatives[relation])
    except ZeroDivisionError:
        recall = 0.0

    try:
        f1 = 2.0 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = 0.0

    if len(relation) <= 4:
        relation += '    '
    if len(relation) < 8:
        relation += '\t'

    print "%s\t%.4f\t%.4f\t%.4f" % (relation, precision, recall, f1)
print '-' * 45

try:
    precision = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_positives.values()))
except ZeroDivisionError:
    precision = 0.0

try:
    recall = sum(true_positives.values()) / (sum(true_positives.values()) + sum(false_negatives.values()))
except ZeroDivisionError:
    recall = 0.0

try:
    f1 = 2.0 * ((precision * recall) / (precision + recall))
except ZeroDivisionError:
    f1 = 0.0

print "MICRO-AVG.\t%.4f\t%.4f\t%.4f" % (precision, recall, f1)
print
