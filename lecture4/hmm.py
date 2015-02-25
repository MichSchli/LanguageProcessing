import argparse
import codecs
from collections import defaultdict
import numpy as np
import json
import sys

# limit the number of floating point when printing numbers to make it easier to read
np.set_printoptions(precision=2)


class HMM(object):
    """
    Implements a Hidden Markov Model
    """

    def __init__(self):
        """
        initialize model parameters
        :return: None
        """
        self.START = '_START_'
        self.UNK = 'UNK'
        self.STOP = '_STOP_'

        # the main model parameters
        self.transitions = defaultdict(lambda: defaultdict(float))
        self.emissions = defaultdict(lambda: defaultdict(float))

        # convenience bookkeeping: helps to assign UNK
        self.vocabulary = set()
        # let's us check what tags we have and iterate over them
        self.tags = set()


    def load(self, file_name):
        """
        load the model from a JSON file
        :param file_name: file to load from. Must be one line JSON
        :return: None, just setting the params
        """
        model_data = codecs.open(file_name, 'r', encoding='utf-8').readline().strip()
        model = json.loads(model_data)
        self.emissions = model['emissions']
        self.transitions = model['transitions']
        self.tags = model['tags']
        self.vocabulary = model['vocabulary']


    def save(self, file_name):
        """
        save model to a JSON file
        :param file_name: filet to write to
        :return: None
        """
        with codecs.open(file_name, "w", encoding='utf-8') as model:
            model.write("%s\n" % json.dumps({'emissions': dict(self.emissions), 'transitions': dict(self.transitions), 'tags': list(self.tags), 'vocabulary': list(self.vocabulary)}))


    def fit(self, file_name):
        """
        fit model to a file in CoNLL format (not needed today since you already get the model)
        :param file_name: input file with two columns
        :return: None, set parameters
        """
        counts = defaultdict(int)

        # record all used tags and words
        for (words, tags) in self.read_conll_file(file_name):
            print "TODO"


    def predict(self, sentence, method='most_likely'):
        """
        predict the most likely tag sequence for a sentence using Viterbi

        :param sentence: a sentence
        :param method: Viterbi or most likely decoding
        :return: predicted tag sequence
        """
        if method == 'viterbi':
            return self.predict_viterbi(sentence)
        else:
            return self.predict_most_likely(sentence)


    def predict_most_likely(self, sentence):
        """
        predict the single most likely tag (from training data) for every token in sentence
        (i.e., just looks at a single tag at a time, no context)
        :sentence: list of tokens
        :return: list of tags
        """

        predicted_tags = [None]*len(sentence)

        #Iterate through the sentence word-by-word:
        for i in xrange(len(sentence)):
            #Get the possible tags for this word:
            possibles = self.tags_emitting_word(sentence[i])

            #Clear memory:
            best_tag = None
            best_prob = float('-Inf')

            #Find the most likely tag and add it to the list:
            for tag, prob in possibles.iteritems():
                if prob > best_prob:
                    best_prob = prob
                    best_tag = tag

            predicted_tags[i] = best_tag

        return sentence, predicted_tags


    def predict_viterbi(self,sentence):
        """
        predict the most likely tag sequences using the Viterbi algorithm
        
        i.e. p(w1,..,wn,t1,..,tn) = argmax \prod transition(prevtag|tag) \prod emission(word|tag)


        :sentence: list of tokens
        :return: list of tags
        """

        #######
        # TBA #
        #######

        return None

    '''
    WE DID IT AND WE DID IT AND WE DID IT... (ACTUALLY WE DID MORE BUT WHO CARES)
    '''
    def show_probabilities(self, searchstring, type="bigram"):
        """
        prints out the probabilities for a given search string
        :type:  bigram    print out all possible tags than can follow the given tag and its probabilities
        :type:  emission   print out all possible words that the tag can emit
        :type:  dictionary   print out all possible tags for a given word
        """
        if type=="bigram":
            prevtag = searchstring
            print self.transitions[prevtag]

        elif type=="emission":
            tag = searchstring
            print self.emissions[tag]

        elif type=="dictionary":
            word = searchstring
            print self.tags_emitting_word(word)
        else:
            print>>sys.stderr, "not a valid type specified!"


    #Returns the set of tags T that can emit the given word along with p(w|t) for each tag t in T.
    def tags_emitting_word(self, word):
        d = {}
        for tag,emission in self.emissions.iteritems():
            if word in emission.keys():
                d[tag] = emission[word]

        #If we have no data, return an unknown tag:
        if len(d) == 0:
            d[self.UNK] = 1.0
        return d

    def sample(self, value, parameter):
        """
        sample
        :param value: a given tag or word
        :param parameter: self.emissions or self.transitions
        :return: a tag samped from the respective conditional probability
        """
        # get probability distro given value
        keys, probabilities = zip(*list(parameter[value].iteritems()))
        # sample
        return np.random.choice(keys, p=probabilities)


    def evaluate(self, file_name,method='most_likely', output=False):
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
            sentenceUnk, predicted_sequence = self.predict(sentence, method=method)
            if output:
                # original sentence, sentence with UNK, predicted tags, gold tags
                for s, su, pred, gold in zip(sentence, sentenceUnk, tags, predicted_sequence):
                    print "{}\t{}\t{}\t{}".format(s, su, pred, gold)
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
    parser = argparse.ArgumentParser(description="""Creates an HMM from tagged text""")
    parser.add_argument("--train", help="corpus to estimate HMM from (CoNLL format)", required=False)
    parser.add_argument("--test", help="test model on a file (CoNLL format)", required=False)
    parser.add_argument("--output", help="output predictions to STDOUT", required=False,action="store_true")
    parser.add_argument("--load", help="load model from JSON file", required=False)
    parser.add_argument("--save", help="save model to JSON file", required=False)
    parser.add_argument("-m", "--method", help="prediction method", choices=['most_likely', 'viterbi'], required=False, default='most_likely')
    args = parser.parse_args()

    # create new model
    hmm = HMM()

    if args.load:
        hmm.load(args.load)

    # fit model to supplied file
    if args.train:
        hmm.fit(args.train)

    # save model
    if args.save:
        hmm.save(args.save)


    hmm.show_probabilities('NOUN', type='bigram')
    hmm.show_probabilities('.', type='emission')
    hmm.show_probabilities('over', type='dictionary')

    # print bigram probabilities for tags that can follow a NOUN

    # word that '.' can emit

    # possible tags and their probs

    print hmm.predict(["this", "can", "can", "fly"])
    print hmm.predict(["we", "are", "filing", "a", "report"])

    ################################
    # OPTIONAL: GENERATE SENTENCES #
    ################################
    #first = hmm.sample(hmm.START, hmm.transitions)


    # check whether to show predictions
    if args.test:
        print hmm.evaluate(args.test, output=args.output)
