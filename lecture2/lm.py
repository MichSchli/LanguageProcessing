# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import codecs
from collections import defaultdict, Counter
import sys
import argparse
import random


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
        self.word_counts = defaultdict(int)
        self.START_SYMBOL="*"
        self.STOP_SYMBOL="STOP"
        self.total_trigram_count = 0


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

                for word in words:
                    if self.word_counts[word] == 0:
                        self.word_counts[word] = 1

                #Collect trigram counts:
                for i in xrange(len(words)-2):
                    self.trigram_counts[(words[i], words[i+1])][words[i+2]] += 1


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
        for i in xrange(len(words)-2):
            probability *= self.trigram_prob(words[i], words[i+1], words[i+2])

        return probability


    def markov_walk(self, mode='greedy',smoothing='none'):
        if mode=='greedy':
            s = self.predict_next(self.START_SYMBOL,self.START_SYMBOL)
        else:
            s = self.weighted_markov_chain_next(self.START_SYMBOL,self.START_SYMBOL, smoothing=smoothing)

        prev = self.START_SYMBOL

        sentence = []
        while s != self.STOP_SYMBOL:
            sentence.append(s)
            temp = prev
            prev = s
            if mode=='greedy':
                s = self.predict_next(temp,s)
            else:
                s = self.weighted_markov_chain_next(temp,s,smoothing=smoothing)


        return sentence

    #Predict the next word given a bigram:
    def predict_next(self, u,v):
        max_count = 0
        word = "ERROR"
        for w,c in self.trigram_counts[(u,v)].iteritems():
            if c > max_count:
                word = w
                max_count = c
        return word

    def weighted_markov_chain_next(self,u,v,smoothing='none'):
        choices = self.trigram_counts[(u,v)]

        if smoothing == 'none':
            total = sum(choices.values())
        else:
            total = sum(choices.values())+sum(self.word_counts.values())
        r = random.uniform(0,total)

        upto = 0
        for word in self.word_counts.keys():
            if upto + choices[word]+ (0 if smoothing=='none' else 1) > r:
                return word
            upto += choices[word]+ (0 if smoothing=='none' else 1)

        return 'ERROR'

    def trigram_prob(self,u,v,w):
        """
        Return the probability of a trigram
        :input: the tokens u v w 
        :return: the probability q(w|u,v)
        """
        ###################
        # Use the trigram_counts to get q(w|u,v)
        choices = self.trigram_counts[(u,v)]
        total = sum(choices.values())+sum(self.word_counts.values())

        trigram_probability = (self.trigram_counts[(u,v)][w]+1)/float(total)

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

    print " ".join(lm.markov_walk(mode='markov', smoothing='laplace'))

    # check whether to show counts
    if args.showtrigrams:
        lm.print_trigrams_count()
