#! /usr/bin/python

__author__ = "Barbara Plank, Dirk Hovy"
__date__  = "$March 23,  2015"

import argparse
import json
from collections import defaultdict


class Parser:
    """
    Simple CKY Parser
    """

    def __init__(self):
        """
        Constructor
        :param countfile:
        :return:
        """
        # structures
        self.unary_counts = defaultdict(int)
        self.binary_counts = defaultdict(int)
        self.nonterminal_counts = defaultdict(int)
        self._RARE_  =  "_RARE_"
        self.vocab  =  set()


    def fit(self, file):
        """
        read in a file with rule type, rule, and counts
        :param file:
        :return:
        """
        # read out the counts from the files
        for line in open(file):
            line = line.strip()
            fields = line.strip().split()

            type = fields[1]
            tcount = int(fields[0])
            if type  ==  "NONTERMINAL":
                nt = fields[2]
                self.nonterminal_counts[nt] += tcount
            elif type  ==  "UNARYRULE":
                nt = fields[2]
                word = fields[3]
                self.vocab.add(word)
                self.unary_counts[(nt, word)] += tcount
            elif type  ==  "BINARYRULE":
                rule = (fields[2], fields[3], fields[4])
                self.binary_counts[rule] += tcount


    def unary_prob(self, X, w):
        """
        normalize unary rule counts to get probs
        :param X: non-terminal, LHS of rule
        :param w: word, RHS of rule
        :return:
        """
        return self.unary_counts[(X, w)]/float(self.nonterminal_counts[X])

    def binary_prob(self, X, Y1, Y2):
        """
        normalize binary rule counts to get probs
        :param X: rule LHS
        :param Y1: RHS, part1
        :param Y2: RHS, part2
        :return:
        """
        return self.binary_counts[(X, Y1, Y2)]/float(self.nonterminal_counts[X])


    def predict(self, sentence):
        """
        predict the best parse using the CKY algorithm
        :param sentence: list of words
        :return: tree (list of lists of ...)
        """
        ## rare word handling
        sentence = [word if word in self.vocab else self._RARE_ for word in sentence]
        N = len(sentence) #length of sentence

        # fixes the crappy notation:
        nonterminals_lookup = self.nonterminal_counts.keys()
        nonterminals = range(len(nonterminals_lookup))

        # Define the chart:
        pi = [[[0.0]*len(nonterminals) for _ in sentence] for _ in sentence]
        bp = dict()

        ##################
        # initialization #
        ##################
        for i in xrange(N):
            for X in nonterminals:
                pi[i][i][X] = self.unary_prob(nonterminals_lookup[X], sentence[i])

        ##############
        # fill chart #
        ##############

        for l in xrange(1, N): #Length:
            for i in xrange(0, N-l):
                j = i+l

                for X in nonterminals:
                    #Iterate through all possible separators:
                    for s in xrange(i, j):
                        for Y in nonterminals:
                            for Z in nonterminals:
                                q = self.binary_prob(nonterminals_lookup[X], nonterminals_lookup[Y], nonterminals_lookup[Z])*pi[i][s][Y]*pi[s+1][j][Z]

                                if q > pi[i][j][X]:
                                    pi[i][j][X] = q
                                    bp[(i,j,nonterminals_lookup[X])] = (nonterminals_lookup[Y],nonterminals_lookup[Z],s)


        # recovery of best parse; get max non-terminal from top-chart item
        best_score = 0
        best_arg = None
        for X in nonterminals:
            if pi[0][N-1][X] > best_score:
                best_score = pi[0][N-1][X]
                best_arg = nonterminals_lookup[X]

        # follow back pointers to get tree
        return self.get_tree(sentence, bp, 0, N-1, best_arg)


    def get_tree(self, sentence, bp, i, j, X):
        """
        return the best tree by following the back pointers
        :param sentence:
        :param bp: backpointer structure
        :param i: starting position
        :param j: end position
        :param X: non-terminal
        :return: tree (list of lists)
        """
        if i == j:
            return [X,  sentence[i]]
        else:
            Y1, Y2, s = bp[(i,j,X)]
            return [X,  self.get_tree(sentence, bp, i, s, Y1), 
                    self.get_tree(sentence, bp, s +1, j, Y2)]


    def get_sentence(self, tree, s):
        """
        returns the sentence (list of terminals) from a nested tree
        :param tree: list of lists of lists of...
        :param s: partial sentence (initially: empty list)
        :return:
        """
        if isinstance(tree,  basestring):
            return
        if len(tree)  ==  3:
            self.get_sentence(tree[1], s)
            self.get_sentence(tree[2], s)
        elif len(tree)  ==  2:
            pos, key  =  tree[0], tree[1]
            s.append(key)


#========================================================

def main():

    # parse command line options
    parser  =  argparse.ArgumentParser(description = """Parse a sentence with CYK for a PCFG""")
    parser.add_argument("--train",  help = "count file (produced by count_cfg_freq.py)", required = True)
    parser.add_argument("--test",  help = "test key file (parse trees as lists in JSON format)",  required = False)
    args  =  parser.parse_args()

    p = Parser()

    if args.train:
        p.fit(args.train)

    print p.predict(['I','eat', 'cake','with', 'strawberries'])



    if args.test:
        for l in open(args.test):
            tree  =  json.loads(l)
            sentence = []
            p.get_sentence(tree, sentence)
            # output predicted parse as json tree to STDOUT
            print json.dumps(p.predict(sentence))


if __name__ == "__main__":
    main()
