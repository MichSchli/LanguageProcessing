__author__ = 'Michael'

import argparse
import Preprocessing
import itertools
import Metrics

parser = argparse.ArgumentParser(description="""Evaluate relations""")
parser.add_argument("--gold", help="gold file (CoNLL format)", required=True)
parser.add_argument("--prediction", help="prediction file (CoNLL format)", required=True)
args = parser.parse_args()


_, _, gold_ner, _ = Preprocessing.parse_full_re_file(args.gold, zip_ne_to_dictionary=False)
gold_ner = list(itertools.chain(*gold_ner))

predict_ner = Preprocessing.parse_processed_sentence_file(args.prediction)
predict_ner = list(itertools.chain(*predict_ner))
c = len(set(gold_ner))
indexing = list(set(gold_ner))

gold_ner = [indexing.index(g) for g in gold_ner]
predict_ner = [indexing.index(g) for g in predict_ner]

print "Precision:", Metrics.precision(gold_ner, predict_ner, c)
print "Recall:", Metrics.recall(gold_ner, predict_ner, c)
print "F1:", Metrics.f1(gold_ner, predict_ner, c)