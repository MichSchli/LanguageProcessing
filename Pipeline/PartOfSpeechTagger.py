import argparse
import codecs
from collections import defaultdict, Counter
import cPickle
import numpy as np
import sys
import networkx as nx

np.set_printoptions(precision=4)


def memoize(f):
    """
    helper function to be used as decorator to memoize features
    :param f:
    :return:
    """
    memo = {}
    def helper(*args):
        key = tuple(args[1:])
        try:
            return memo[key]
        except KeyError:
            memo[key] = f(*args)
            return memo[key]
    return helper



class StructuredPerceptron(object):
    """
    implements a structured perceptron as described in Collins 2002
    """

    def __init__(self):
        """
        initialize model
        :return:
        """
        self.feature_weights = defaultdict(float)
        self.tags = set()

        self.START = "__START__"
        self.END = "__END__"


    def fit(self, train_data, train_labels, iterations=5, learning_rate=0.2):
        """
        read in a CoNLL file, extract emission features iterate over instances to train weight vector
        :param file_name:
        :return:
        """
        averaged_weights = Counter()

        for iteration in xrange(iterations):
            correct = 0
            total = 0.0
            sys.stderr.write('iteration %s\n************\n' % (iteration+1))

            for i, words in enumerate(train_data):
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

                for tag in train_labels[i]:
                    self.tags.add(tag)

                # get prediction
                prediction = self.predict(words)

                # derive global features
                global_gold_features = self.get_global_features(words, train_labels[i])
                global_prediction_features = self.get_global_features(words, prediction)

                # update weight vector
                for fid, count in global_gold_features.iteritems():
                    self.feature_weights[fid] += learning_rate * count
                for fid, count in global_prediction_features.iteritems():
                    self.feature_weights[fid] -= learning_rate * count

                # compute training accuracy for this iteration
                correct += sum([1 for (predicted, gold) in zip(prediction, train_labels[i]) if predicted == gold])
                total += len(train_labels[i])

            sys.stderr.write('\n\t%s features\n' % (len(self.feature_weights)))
            averaged_weights.update(self.feature_weights)
            sys.stderr.write('\tTraining accuracy: %.4f\n\n' % (correct/total))

        self.feature_weights = averaged_weights


    def get_global_features(self, words, tags):
        """
        count how often each feature fired for the whole sentence
        :param words:
        :param tags:
        :return:
        """
        feature_counts = Counter()

        for i, (word, tag) in enumerate(zip(words, tags)):
            previous_tag = self.START if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag))

        return feature_counts


    @memoize
    def get_features(self, word, tag, previous_tag):
        """
        get all features that can be derived from the word and tags
        :param word:
        :param tag:
        :param previous_tag:
        :return:
        """
        word_lower = word.lower()
        prefix = word_lower[:3]
        suffix = word_lower[-3:]

        features = [
                    'TAG_%s' % (tag),                       # current tag
                    'TAG_BIGRAM_%s_%s' % (previous_tag, tag),  # tag bigrams
                    'WORD+TAG_%s_%s' % (word, tag),            # word-tag combination
                    'WORD_LOWER+TAG_%s_%s' % (word_lower, tag),# word-tag combination (lowercase)
                    'UPPER_%s_%s' % (word[0].isupper(), tag),  # word starts with uppercase letter
                    'DASH_%s_%s' % ('-' in word, tag),         # word contains a dash
                    'PREFIX+TAG_%s_%s' % (prefix, tag),        # prefix and tag
                    'SUFFIX+TAG_%s_%s' % (suffix, tag)        # suffix and tag

                    #########################
                    # ADD MOAAAAR FEATURES! #
                    #########################
        ]

        return features



    def predict(self, words):
        """
        use Viterbi to predict best sequence
        :param words:
        :param word_features:
        :return:
        """
        number_of_words = len(words)
        number_of_states = len(self.tags)
        tags = list(self.tags)

        V = nx.DiGraph()

        # initialize
        for j, tag in enumerate(tags):
            features = self.get_features(words[0], tag, self.START)
            feature_weights = sum((self.feature_weights[x] for x in features))

            V.add_edge(self.START, "%s_0" % (tags[j]), weight=-feature_weights)

        # iterate
        for i in xrange(1, number_of_words):
            for j, tag in enumerate(tags):
                for k, previous_tag in enumerate(tags):
                    features = self.get_features(words[i], tag, previous_tag)
                    feature_weights = sum((self.feature_weights[x] for x in features))

                    V.add_edge("%s_%s" % (tags[k], i-1), "%s_%s" % (tags[j], i), weight=-feature_weights)

        # add END node
        for j, tag in enumerate(tags):
            V.add_edge("%s_%s" % (tags[j], number_of_words-1), self.END, weight=1.0)

        # find shortest path
        predecessors, edge_weights = nx.bellman_ford(V, self.START)

        current = self.END
        best_path = []
        while current != self.START:
            best_path.append(predecessors[current])
            current = predecessors[current]

        return [node.split('_')[0] for node in best_path[::-1][1:]]


    def evaluate(self, test_data, test_labels, output=False):
        """
        compute accuracy on a test file
        :param file_name:
        :param output:
        :return:
        """
        correct = 0
        total = 0.0
        if output:
            sys.stderr.write('\nTesting\n')
            sys.stderr.write('*******\n')

        for i, (words) in enumerate(test_data):
            if output:
                if i%100==0:
                    sys.stderr.write('%s'%i)
                elif i%10==0:
                    sys.stderr.write('.')

            # get prediction
            prediction = self.predict(words)

            if output:
                print >> sys.stderr, "\t%s" % " ".join(words)
                print >> sys.stderr, "GOLD:\t%s" % " ".join(test_labels[i])
                print >> sys.stderr, "PRED.:\t%s" % " ".join(prediction)
                print >> sys.stderr, ""

            correct += sum([1 for (predicted, gold) in zip(prediction, test_labels[i]) if predicted == gold])
            total += len(test_labels[i])
        if output:
            print >> sys.stderr, "\nTest accuracy on %s items: %.4f" % (i+1, correct/total)

        return correct/total

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


    def save(self, file_name):
        """
        save model
        :param file_name:
        :return:
        """
        print >> sys.stderr, "saving model...",
        model = {'tags': list(self.tags), 'weights': dict(self.feature_weights)}
        cPickle.dump(model, open(file_name, "wb"))
        print >> sys.stderr, "done"


    def load(self, file_name):
        """
        load model from JSON file
        :param file_name:
        :return:
        """
        print >> sys.stderr, "loading model...",
        model = cPickle.load(open(file_name, 'rb'))
        self.tags = set(model['tags'])
        self.feature_weights = defaultdict(float, model['weights'])
        print >> sys.stderr, "done"


# if script is run from command line, automatically execute the following
if __name__=="__main__":

    '''
    # parse command line options
    parser = argparse.ArgumentParser(description="""Run a structured perceptron""")
    parser.add_argument("--train", help="train model on a file (CoNLL format)", required=False)
    parser.add_argument("--test", help="test model on a file (CoNLL format)", required=False)
    parser.add_argument("--output", help="output predictions to stdout", required=False,action="store_true")
    parser.add_argument("--load", help="load model from JSON file", required=False)
    parser.add_argument("--save", help="save model as JSON file", required=False)
    parser.add_argument("--iterations", help="number of training iterations", required=False, default=5, type=int)
    parser.add_argument("--learning_rate", help="learning rate during training", required=False, default=0.2, type=float)
    args = parser.parse_args()
    '''

    inp = [line.strip() for line in sys.stdin]

    sentences = inp[::2]
    languages = inp[1::2]

    #Create a model for each language
    en_sp = StructuredPerceptron()
    nl_sp = StructuredPerceptron()
    pt_sp = StructuredPerceptron()

    en_sp.load('models/postagger.model')
    nl_sp.load('models/sp-nl.model')
    pt_sp.load('models/sp-pt.model')

    tagged = [None]*len(languages)
    #Run through sentences, tagging with the right model
    for i,s in enumerate(sentences):
        sentence = s.split(' ')
        if languages[i] == 'EN':
            tagged[i] = en_sp.predict(sentence)
        elif languages[i] == 'NL':
            tagged[i] = nl_sp.predict(sentence)
        else:
            tagged[i] = pt_sp.predict(sentence)

        for j, tag in enumerate(tagged[i]):
            print sentence[j], tag
        print ''

    #print tagged

    '''
    # create new model
    sp = StructuredPerceptron()

    if args.load:
        sp.load(args.load)

    if args.train:
        sp.fit(args.train, iterations=args.iterations, learning_rate=args.learning_rate)

    if args.save:
        sp.save(args.save)

    # check whether to show predictions
    if args.test:
        sp.evaluate(args.test, output=args.output)
    '''