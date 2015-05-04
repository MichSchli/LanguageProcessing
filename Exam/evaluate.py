import sys
import codecs
import argparse
from collections import defaultdict


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

            if len(fields) == 3:
                rel_output = True
                start = fields[0]
                end = fields[1]
                rel = fields[2]
                instance.append((start, end, rel))

        else:
            # after relation output, commit
            if rel_output:
                yield instance
                rel_output = False
                instance = []

    if rel_output and instance:
        yield instance


# read in files
predictions = read_RE_file(args.prediction)
gold = read_RE_file(args.gold)

# collect raw counts for true and false cases
# TODO: use dictionaries instead of floats to count for each relation
true_positives = defaultdict(int)
false_positives = defaultdict(int)
false_negatives = defaultdict(int)

for gold_instance, prediction_instance in zip(gold, predictions):
    for prediction in prediction_instance:
        # TODO: introduce soft matches
        if prediction in gold_instance:
            true_positives[prediction[2]] += 1.0
        else:
            false_positives[prediction[2]] += 1.0

    for gold in gold_instance:
        if gold not in prediction_instance:
            false_negatives[gold[2]] += 1.0

for label in true_positives.iterkeys():
   print '=============================\n'
   print label+':'
   print '-----------------------------'
   # compute precision, recall and F1 from the raw counts
   precision = true_positives[label] / (true_positives[label] + false_positives[label])
   recall = true_positives[label] / (true_positives[label] + false_negatives[label])
   f1 = 2.0 * ((precision * recall) / (precision + recall))

   print "precision: %.4f" % (precision)
   print "recall: %.4f" % (recall)
   print "f1: %.4f" % (f1)

print '=============================\n'
print 'total:'
print '-----------------------------'
precision = sum(true_positives.itervalues()) / (sum(true_positives.itervalues()) + sum(false_positives.itervalues()))
recall = sum(true_positives.itervalues()) / (sum(true_positives.itervalues()) + sum(false_negatives.itervalues()))
f1 = 2.0 * ((precision * recall) / (precision + recall))
print "precision: %.4f" % (precision)
print "recall: %.4f" % (recall)
print "f1: %.4f" % (f1)