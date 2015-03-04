import argparse
import codecs
from collections import defaultdict
import numpy as np
import json
import sys
import math

np.set_printoptions(precision=4)

class HMM(object):


    def __init__(self):
        """
        initialize model parameters
        :return:
        """
        self.START = '_START_'
        self.UNK = 'UNK'
        ### for the optional exercise
        self.UNKed = 'UNK-ed'
        self.UNKly = 'UNK-ly'
        ####
        self.STOP = '_STOP_'
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))
        self.vocabulary = set()
        self.tags = set()

    def load(self, file_name):
        """
        load the model from a json file
        """
        model_data = codecs.open(file_name, 'r', encoding='utf-8').readline().strip()
        model = json.loads(model_data)
        self.emissions = model['emissions']
        self.transitions = model['transitions']
        self.tags = model['tags']
        self.vocabulary = model['vocabulary']


    def show_probabilities(self, searchstring, type="bigram"):
        """
        prints out the probabilities for a given search string
        :type:  bigram    print out all possible tags than can follow the given tag and its probabilities
        :type:  emission   print out all possible words that the tag can emit
        :type:  dictionary   print out all possible tags for a given word
        """
        if type=="bigram":
            prevtag = searchstring
            ##################
            # YOUR CODE HERE #
            ##################
            for tag in self.transitions[prevtag]:
                print prevtag, tag, self.transitions[prevtag][tag]
        elif type=="emission":
            tag = searchstring
            ##################
            # YOUR CODE HERE #
            ##################

        elif type=="dictionary":
            word = searchstring
            ##################
            # YOUR CODE HERE #
            ##################

        else:
            print>>sys.stderr, "not a valid type specified!"


    def predict(self, sentence, method='most_likely'):
        """
        predict the most likely tag sequence for a sentence using Viterbi

        :param sentence: a sentence
        :param method: Viterbi or most likely decoding
        :return: predicted tag sequence
        """
        #### optional: change the following to include advanced UNK word handling ####
        #### e.g. unknown words ending with 'ly' get mapped to self.UNKly  
        sentence = [self.UNK if w not in self.vocabulary else w for w in sentence]
        #### 

        if method == 'viterbi':
            return self.predict_viterbi(sentence)
        else:
            return self.predict_most_likely(sentence)


    def predict_most_likely(self,sentence):
        """
        predict the single most likely tag (from training data) for every token in sentence
        (i.e., just looks at a single tag at a time, no context)
        :sentence: list of tokens
        :return: list of tags
        """
        best_tag = None
        best_prob = float('-Inf')        

        predtags=[]
        for token in sentence:
            for tag in self.tags:
                ##### YOUR CODE FROM LAST LECTURE #####
                pass

            predtags.append(best_tag)

            # reset
            best_tag = None
            best_prob = float('-Inf')

        return (sentence,predtags)


    def predict_viterbi(self,sentence):
        """
        predict the most likely tag sequences using the Viterbi algorithm
        
        i.e. p(w1,..,wn,t1,..,tn) = argmax \prod transition(prevtag|tag) \prod emission(word|tag)


        :sentence: list of tokens
        :return: list of tags
        """

        N=len(sentence)
        M=len(self.tags) #number of tags


        # initialize matrices
        # create trellis of size M (number of tags) x N (sentence length) 
        Q = np.ones((len(self.tags), N)) * float('-Inf')
        backp = np.ones((len(self.tags), N), dtype=np.int16) * -1 #backpointers


        #### YOUR CODE HERE ####

        ### initialization step
        ## fill in first column

        tags = list(self.tags)
        n_of_tags = len(tags)

        for i in xrange(n_of_tags):
            Q[i,0] = self.transitions[self.START][tags[i]]*self.emissions[tags[i]].get(sentence[0], 0)

             
        # iteration step, filling the lattice
        # for every token in the sentence go over lattice
        for i in xrange(1,N):
            ###
            for current_tag in xrange(n_of_tags):
                Q[current_tag,i] = -1

                for previous_tag in xrange(n_of_tags):
                    t_prob = self.transitions[tags[previous_tag]].get(tags[current_tag],0)
                    e_prob = self.emissions[tags[current_tag]].get(sentence[i], 0)
                    to_prev_prob = Q[previous_tag, i-1]

                    combined_prob = t_prob*e_prob*to_prev_prob
                    if combined_prob > Q[current_tag,i]:
                        Q[current_tag,i] = combined_prob
                        backp[current_tag, i] = previous_tag

        # termination, final best
        final_best = None
        final_best_prob = -1
        for tag in xrange(n_of_tags):
            if Q[tag, N-1] > final_best_prob:
                final_best_prob = Q[tag, N-1]
                final_best = tag

        # print best tags in reverse order : hint: look at the [::-1] construction
        best_tags = []
        temp = final_best
        for i in range(N)[::-1]:
            best_tags.append(tags[temp])
            temp = backp[temp, i]

        #print best_tags[::-1]
        return (sentence, best_tags[::-1])


    def evaluate(self, file_name,method='most_likely',output=False):
        """
        evaluate model on a test file in CoNLL format
        :param file_name:
        :return: (sentence accuracy, word accuracy)
        """
        correct_words = 0
        correct_sentences = 0

        words_total = 0.0
        sentences_total = 0.0

        for sentence, tags in self.read_conll_file(file_name):

            # check whether entire tag sequence was correct
            sentences_total += 1
            sentenceUnk, predicted_sequence = self.predict(sentence,method=method)
            if output:
                for s,su,pred,gold in zip(sentence,sentenceUnk,tags,predicted_sequence):
                    print "{}\t{}\t{}\t{}".format(s,su,pred,gold)
                print ""
            if predicted_sequence == tags:
                correct_sentences += 1

                # add 1 for each correct tag
                words_total += len(sentence)
                correct_words += len(sentence)

            # check individual tags for correctness
            else:
                for predicted_tag, gold_tag in zip(predicted_sequence, tags):
                    words_total += 1
                    if predicted_tag == gold_tag:
                        correct_words += 1

        return (correct_sentences/sentences_total, correct_words/words_total)


    def read_conll_file(self, file_name):
        """
        read in a file with format:
        word1    tag1
        ...      ...
        wordN    tagN

        Sentences MUST be separated by newlines!

        :param file_name: file to read in
        :return: generator of instances ((list of  words, list of tags) pairs)
        """
        current_words = []
        current_tags = []

        for line in codecs.open(file_name, encoding='utf-8'):
            line = line.strip()

            if line:
                word, tag = line.split('\t')
                current_words.append(word)
                current_tags.append(tag)

            else:
                yield (current_words, current_tags)
                current_words = []
                current_tags = []

        # if file does not end in newline (it should...), check whether there is an instance in the buffer
        if current_tags != []:
            yield (current_words, current_tags)



# if script is run from command line, automatically execute the following
if __name__=="__main__":

    # parse command line options
    parser = argparse.ArgumentParser(description="""Creates a Naive Bayes Model from text""")
    parser.add_argument("--test", help="test model on a file (CoNLL format)", required=False)
    parser.add_argument("--output", help="output predictions to stdout", required=False,action="store_true")
    parser.add_argument("--load", help="load model from JSON file", required=False)
    parser.add_argument("-m","--method", help="predict method [most_likely,viterbi]", required=False,default='most_likely')
    args = parser.parse_args()

    # create new model
    hmm = HMM()

    if args.load:
        hmm.load(args.load)


    #### add code to test your code #### 
    hmm.predict_viterbi(['this','fight','is','lost'])

    # check whether to show predictions
    if args.test:
        print hmm.evaluate(args.test,output=args.output,method=args.method)
