# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
from collections import defaultdict, Counter
import sys
import argparse


__author__ = 'Dirk Hovy, Barbara Plank'


class LanguageModel(object):

    def __init__(self):
        """
        initialize object variables
        trigram_counts: holds counts of trigrams, 
                 e.g. "u v w" -> trigram_counts["u v"]["w"]
        start and end symbol
        """
        self.trigram_counts = defaultdict(lambda: defaultdict(int))
        self.START_SYMBOL="*"
        self.STOP_SYMBOL="STOP"


    def fit(self, file_name):
        """
        collect counts from the input file and store them in trigram_counts
        :param file_name: the input file, should be one sentence per line
        :param lowercase: whether to lower-case the input
        :return: None
        """
        print >>sys.stderr, "Estimating language model"
        for line in codecs.open(file_name, encoding='utf-8'):
            line = line.strip()

            if line:
                words = line.split() # default is split by whitespace characters 
                words.append(self.STOP_SYMBOL)
                words = [self.START_SYMBOL, self.START_SYMBOL] + words
         
                ###################
                # Collect counts of all trigrams and store them in trigram_counts
                # hint: to understand the data structure, examine the function print_trigrams_count
                #
                ##########################
                # *** YOUR CODE HERE *** #
                ##########################



    def predict(self, sentence, smoothing=None):
        """
        compute the probability of the given sentence under the model
        :param sentence: the input sentence to estimate
        :param smoothing: type of smoothing to apply (optional)
        :return: the probability
        """
        words = sentence.split()
        words.append("STOP")
        probability = 1.0

        words = [self.START_SYMBOL, self.START_SYMBOL] + words
        ###################
        # Compute the probability of a sentence under the trigram model
        # p(x1,..,xn)= \prod {q(x_i| x_{i-2}x_{i-1}}
        ##########################
        # *** YOUR CODE HERE *** #
        ##########################

        return probability


    def trigram_prob(self,u,v,w):
        """
        Return the probability of a trigram
        :input: the tokens u v w 
        :return: the probability q(w|u,v)
        """
        ###################
        # Use the trigram_counts to get q(w|u,v) 
        ##########################
        # *** YOUR CODE HERE *** #
        ##########################
        trigram_probability = 0.0

        return trigram_probability


    def print_trigrams_count(self):
        """
        print trigrams and their counts
        """
        for u_v in self.trigram_counts:
            for w in self.trigram_counts[u_v]:
                count=self.trigram_counts[u_v][w]
                print "{2}\tc({0} {1})".format(u_v,w,count)

        


# if script is run from command line, automatically execute the following
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Creates a Language Model from text""")
    parser.add_argument("train", help="corpus to estimate LM from")
    parser.add_argument("-p","--predict",help="predict probabilities of sentences in a file (one sentence per line)",required=False)
    parser.add_argument("-s","--showtrigrams",help="show trigrams frequencies",required=False,default=False,action="store_true")
    args = parser.parse_args()

    # create new model
    lm = LanguageModel()

    # fir model to supplied file
    lm.fit(args.train)

    # check whether to show predictions
    if args.predict:
        for line in codecs.open(args.predict, encoding='utf-8'):
            line=line.strip()
            print line, lm.predict(line)

    # check whether to show counts
    if args.showtrigrams:
        lm.print_trigrams_count()
