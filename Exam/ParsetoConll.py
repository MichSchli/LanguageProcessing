__author__ = 'Michael'

import Preprocessing
import argparse

parser = argparse.ArgumentParser(description="Contains implementation for formatting to conll.")
parser.add_argument("--sentences", help="Read a preprocessed sentence file.", required=True)
parser.add_argument("--pos", help="Read a POS file.", required=True)
args = parser.parse_args()

sentences = Preprocessing.parse_processed_sentence_file(args.sentences)
poss = Preprocessing.parse_processed_sentence_file(args.pos)

for sentence, pos in zip(sentences, poss):
    j = 0
    for w, p in zip(sentence, pos):
        print '\t'.join([str(j), w, '_', p, '_', '_', '0', 'PLACEHOLDER', '_', '_'])
        j += 1
    print ''